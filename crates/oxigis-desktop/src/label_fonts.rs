// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Supplying the label rasteriser with fonts.
//!
//! `oxigis-render` never reads a font itself — finding the bytes is a shell
//! responsibility. This shell hands over the bundled Latin face immediately
//! (it is already in the binary) and then walks the OS font directories on a
//! background thread for a CJK fallback chain, streaming each face onto the map
//! as it is read rather than making the first labelled frame wait for the whole
//! scan. The scan itself lives in `font_scan`; this module is the GPU-side glue
//! and the thread that drives it.
//!
//! Split out of `main.rs` under the 2000-line rule.

use crate::font_scan;

/// One message from the background font scan.
///
/// Two variants because the two chains install differently: fallbacks are
/// APPENDED, so streaming them one at a time gets the best-ranked face onto
/// the map as soon as its bytes are read; the bold chain REPLACES whatever
/// bold chain there was, so sending it entry by entry would invalidate every
/// shaped label once per face for no benefit.
pub(crate) enum ScannedFont {
    /// A regular-chain fallback face, in chain order.
    Fallback(Vec<u8>),
    /// The whole bold chain, sent once after the regular chain (print/text
    /// v1.4, D-W4). Empty is never sent — no bold face simply means no
    /// message, and Bold labels keep drawing Regular.
    BoldChain(Vec<Vec<u8>>),
}

/// Installs the primary label font into the GPU map.
///
/// The bundled Noto Sans Regular (Latin, OFL-1.1) — the same 431 KB the web
/// shell ships, and free to hand over because it is already in the binary. The
/// CJK fallback arrives later, through [`start_cjk_font_scan`].
///
/// Returns whether it landed: `false` means no `MapGpuState` is installed yet,
/// which is the normal state on the first frame, so the caller retries.
pub(crate) fn install_label_fonts(render_state: &eframe::egui_wgpu::RenderState) -> bool {
    let installed = oxigis_ui::map_gpu::set_label_fonts(
        render_state,
        oxifont_bundled::NOTO_SANS_REGULAR.to_vec(),
        Vec::new(),
    );
    if installed {
        tracing::info!("OxiGIS desktop: label font installed (Noto Sans, Latin)");
    }
    installed
}

/// The CJK fallback chain, resolved once per process.
static CJK_REGULAR_PATHS: std::sync::OnceLock<std::sync::Arc<[std::path::PathBuf]>> =
    std::sync::OnceLock::new();

/// The CJK bold chain, resolved once per process.
static CJK_BOLD_PATHS: std::sync::OnceLock<std::sync::Arc<[std::path::PathBuf]>> =
    std::sync::OnceLock::new();

/// The CJK fallback chain's paths, walking the OS font tree at most once.
///
/// The scan reads every candidate's header to rank it, so on Windows it is
/// tens of megabytes of I/O — and a PDF export needs the same answer the
/// startup scan already computed. Memoised for the process: installing a font
/// while OxiGIS runs takes effect on the next launch, which is the documented
/// cost of not re-walking four directory trees per export.
pub(crate) fn cjk_regular_paths() -> std::sync::Arc<[std::path::PathBuf]> {
    std::sync::Arc::clone(CJK_REGULAR_PATHS.get_or_init(|| font_scan::find_cjk_font_paths().into()))
}

/// The CJK bold chain's paths — [`cjk_regular_paths`]'s twin.
pub(crate) fn cjk_bold_paths() -> std::sync::Arc<[std::path::PathBuf]> {
    std::sync::Arc::clone(
        CJK_BOLD_PATHS.get_or_init(|| font_scan::find_cjk_bold_font_paths().into()),
    )
}

/// Starts the system CJK font scan on a background thread.
///
/// Off the render thread on purpose: the scan walks every OS font directory
/// and then reads a fallback chain — faces or whole TrueType Collections —
/// that can run tens of MB, which on a cold file-system cache is seconds of
/// I/O. Doing that inline would stall the very first frame — and the map does
/// not need it to draw, since Latin labels work from the bundled face alone.
///
/// Genuinely streamed: each chain entry is read
/// ([`font_scan::read_cjk_font`]) and sent as its own message, so the
/// best-ranked face reaches the map as soon as *its* bytes are off the disk
/// rather than after the whole chain is. This is the native twin of the
/// browser shell's asynchronous font fetch, landing through the
/// [`oxigis_ui::map_gpu::add_label_fallback_fonts`] seam.
///
/// Returns [`None`] if the OS refuses the thread, in which case there is simply
/// no CJK fallback.
pub(crate) fn start_cjk_font_scan(
    ctx: &egui::Context,
) -> Option<std::sync::mpsc::Receiver<ScannedFont>> {
    let (tx, rx) = std::sync::mpsc::channel();
    let ctx = ctx.clone();
    match std::thread::Builder::new()
        .name("oxigis-font-scan".to_owned())
        .spawn(move || {
            for path in cjk_regular_paths().iter() {
                let Some(bytes) = font_scan::read_cjk_font(path) else {
                    continue;
                };
                tracing::debug!(
                    path = %path.display(),
                    "OxiGIS desktop: CJK fallback queued for install",
                );
                // A send failure means the app is already gone; nothing to do.
                if tx.send(ScannedFont::Fallback(bytes)).is_err() {
                    return;
                }
                // The map may well be idle by now, so ask for the frame that
                // will pick the bytes up.
                ctx.request_repaint();
            }
            // The bold chain last: labels draw at Regular until it lands, and
            // installing it invalidates the cache, so it must not interleave
            // with the regular chain's own invalidations.
            let bold: Vec<Vec<u8>> = cjk_bold_paths()
                .iter()
                .filter_map(|path| font_scan::read_cjk_font(path))
                .collect();
            if !bold.is_empty() && tx.send(ScannedFont::BoldChain(bold)).is_ok() {
                ctx.request_repaint();
            }
        }) {
        Ok(_handle) => Some(rx),
        Err(error) => {
            tracing::warn!(%error, "OxiGIS desktop: could not start the CJK font scan");
            None
        }
    }
}

/// Installs every CJK fallback font the background scan has produced so far.
///
/// Returns whether the receiver is now spent (the scan thread hung up), so
/// the caller can drop it and stop polling. Fonts arrive one message per
/// chain entry, in chain order; everything available this frame is appended
/// in a single batch, so a warm-cache burst costs one label invalidation (and
/// one hand-off copy of the chain) instead of one per font.
pub(crate) fn drain_cjk_font(
    render_state: &eframe::egui_wgpu::RenderState,
    rx: &std::sync::mpsc::Receiver<ScannedFont>,
    ctx: &egui::Context,
) -> bool {
    use std::sync::mpsc::TryRecvError;
    let mut batch = Vec::new();
    let mut bold: Option<Vec<Vec<u8>>> = None;
    let spent = loop {
        match rx.try_recv() {
            Ok(ScannedFont::Fallback(bytes)) => batch.push(bytes),
            // Last one wins: the scan sends the bold chain exactly once.
            Ok(ScannedFont::BoldChain(chain)) => bold = Some(chain),
            // The scan thread is done: whatever is batched is the rest.
            Err(TryRecvError::Disconnected) => break true,
            // More may still come; keep the receiver and poll next frame.
            Err(TryRecvError::Empty) => break false,
        }
    };
    if !batch.is_empty() {
        let fonts = batch.len();
        let bytes: usize = batch.iter().map(Vec::len).sum();
        if oxigis_ui::map_gpu::add_label_fallback_fonts(render_state, batch) {
            tracing::info!(
                fonts,
                bytes,
                "OxiGIS desktop: CJK label fallbacks installed; labels will re-shape",
            );
            // Installing fallbacks invalidates every shaped label, so the map
            // must draw at least one more frame to show the new glyphs.
            ctx.request_repaint();
        } else {
            tracing::warn!(
                fonts,
                "OxiGIS desktop: CJK fonts arrived before the map; discarded",
            );
        }
    }
    if let Some(chain) = bold {
        let fonts = chain.len();
        if oxigis_ui::map_gpu::set_label_bold_fonts(render_state, chain) {
            tracing::info!(
                fonts,
                "OxiGIS desktop: bold label chain installed; Bold labels draw bold",
            );
            ctx.request_repaint();
        } else {
            tracing::warn!(
                fonts,
                "OxiGIS desktop: bold fonts arrived before the map; discarded",
            );
        }
    }
    spent
}
