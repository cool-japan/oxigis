// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Driving [`PmtilesOpen`], and the refusals that belong to the codec side.
//!
//! `oxigis-render` keeps a no-codec contract, so its open state machine hands
//! the root directory and the metadata block *back* — tagged with the archive's
//! `internal_compression` — and waits for the plain bytes
//! ([`PmtilesOpenProgress::NeedPlain`]). This crate already owns a gzip
//! implementation (`oxiarc-deflate`, the same one
//! [`crate::vector_provider`] gunzips tile bodies with), so answering that
//! hand-back is a three-line loop — [`advance_open`] — and every consumer in
//! this module drives the open through it rather than repeating the dance.
//!
//! The other half of this file is [`check_archive`]: the once-at-open refusal
//! of everything the header can declare that this build will not read. Doing it
//! here, once, is what turns "the map is blank" into a sentence naming the
//! archive's own bytes.

use oxigis_render::ByteRange;
use oxigis_render::pmtiles::{
    Compression, PmtilesArchive, PmtilesError, PmtilesOpen, PmtilesOpenProgress, TileType,
};

use crate::tile_provider::TileError;

/// Whether an archive holds raster tiles or vector ones.
///
/// Decided **once, at open**, from the header's `tile_type` byte: an archive
/// cannot change its mind halfway through, and routing per tile would mean
/// discovering a mismatch after a layer had already been created for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchiveContent {
    /// Image tiles — PNG, JPEG or WebP — drawn by
    /// [`crate::archive::ArchiveTileProvider`].
    Raster,
    /// Mapbox Vector Tiles, fed to
    /// [`crate::vector_provider::VectorTileProvider`] through
    /// [`crate::archive::ArchiveTileTransport`].
    Vector,
}

impl ArchiveContent {
    /// A lowercase name for status lines and refusals.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Raster => "raster",
            Self::Vector => "vector",
        }
    }
}

/// What [`advance_open`] wants next.
pub(crate) enum OpenStep {
    /// These archive bytes are needed; hand them to
    /// [`PmtilesOpen::supply`] and call [`advance_open`] again.
    Need(ByteRange),
    /// The archive is open.
    Ready(Box<PmtilesArchive>),
    /// The archive will never open, for this named reason.
    Failed(String),
}

/// Polls `open` until it needs bytes, is done, or refuses — inflating any
/// `internal_compression`-coded block the state machine hands back.
///
/// The gzip step lives here rather than in `oxigis-render` because that crate
/// keeps a no-codec contract; see [`oxigis_render::pmtiles`]'s module docs. An
/// archive stored with `internal_compression = None` never reaches the inflate
/// arm at all, which is why the offline fixtures exercise the whole parse with
/// no codec involved.
pub(crate) fn advance_open(open: &mut PmtilesOpen) -> OpenStep {
    loop {
        match open.poll() {
            Ok(PmtilesOpenProgress::Need(range)) => return OpenStep::Need(range),
            Ok(PmtilesOpenProgress::NeedPlain {
                slot,
                compression,
                raw,
            }) => {
                let plain = match plain_directory(compression, raw) {
                    Ok(plain) => plain,
                    Err(reason) => {
                        return OpenStep::Failed(format!(
                            "the archive's {} could not be decoded: {reason}",
                            slot.name()
                        ));
                    }
                };
                if let Err(error) = open.supply_plain(slot, plain) {
                    return OpenStep::Failed(error.to_string());
                }
            }
            Ok(PmtilesOpenProgress::Ready(archive)) => return OpenStep::Ready(archive),
            Err(error) => return OpenStep::Failed(error.to_string()),
        }
    }
}

/// Decodes one `internal_compression`-coded block: a directory or the metadata.
///
/// The twin of [`plain_tile`], and deliberately a *separate* function: the two
/// honour different header fields, and conflating them is the mistake a
/// measured archive (gzip directories, uncompressed PNG tiles) punishes
/// immediately.
///
/// Brotli and zstd never reach here through the open path: [`PmtilesOpen`]
/// refuses an archive declaring either, by name, before it hands any block
/// back.
pub(crate) fn plain_directory(compression: Compression, raw: Vec<u8>) -> Result<Vec<u8>, String> {
    match compression {
        Compression::None => Ok(raw),
        Compression::Gzip => {
            oxiarc_deflate::gzip_decompress(&raw).map_err(|error| error.to_string())
        }
        Compression::Unknown | Compression::Brotli | Compression::Zstd => Err(format!(
            "it is coded with {}, which OxiGIS does not read",
            compression.name()
        )),
    }
}

/// Refuses, by name, every archive this build will not read, and reports what
/// kind of tiles the rest hold.
///
/// `expected` is the content the layer asking for the archive was built for;
/// a mismatch is refused rather than drawn empty, because the two paths draw
/// through entirely different providers and there is no honest way to show a
/// vector archive through the raster one.
pub(crate) fn check_archive(
    archive: &PmtilesArchive,
    expected: Option<ArchiveContent>,
) -> Result<ArchiveContent, String> {
    let header = archive.header();
    let content = match header.tile_type {
        TileType::Mvt => ArchiveContent::Vector,
        TileType::Png | TileType::Jpeg | TileType::Webp => ArchiveContent::Raster,
        TileType::Avif => {
            return Err(
                "this PMTiles archive stores AVIF tiles, and no Pure-Rust AVIF decoder is in \
                 OxiGIS's dependency graph"
                    .to_owned(),
            );
        }
        TileType::Unknown => {
            return Err(
                "this PMTiles archive does not declare what its tiles are, which OxiGIS will \
                 not guess"
                    .to_owned(),
            );
        }
    };
    match header.tile_compression {
        Compression::None | Compression::Gzip => {}
        Compression::Unknown => {
            return Err(
                "this PMTiles archive does not declare how its tiles are compressed, which \
                 OxiGIS will not guess"
                    .to_owned(),
            );
        }
        refused @ (Compression::Brotli | Compression::Zstd) => {
            return Err(format!(
                "this PMTiles archive uses {} for its tiles, which OxiGIS does not read",
                refused.name()
            ));
        }
    }
    if let Some(expected) = expected
        && expected != content
    {
        return Err(format!(
            "this archive holds {} tiles, so it cannot be drawn as a {} layer",
            content.name(),
            expected.name()
        ));
    }
    Ok(content)
}

/// Decodes one tile body per the header's `tile_compression`.
///
/// The bytes are honoured, never sniffed: `internal_compression` and
/// `tile_compression` are independent header fields and genuinely differ in the
/// wild (a measured raster archive is gzip-internal with uncompressed PNG
/// tiles), so a blind gunzip would break it and a blind magic sniff would break
/// a gzip-magic-prefixed image.
pub(crate) fn plain_tile(compression: Compression, raw: Vec<u8>) -> Result<Vec<u8>, TileError> {
    match compression {
        Compression::None => Ok(raw),
        Compression::Gzip => oxiarc_deflate::gzip_decompress(&raw)
            .map_err(|error| TileError::permanent(format!("gzip decode failed: {error}"))),
        // Unreachable: `check_archive` refuses these when the header lands.
        Compression::Unknown | Compression::Brotli | Compression::Zstd => {
            Err(TileError::permanent(format!(
                "the archive's tiles are coded with {}, which OxiGIS does not read",
                compression.name()
            )))
        }
    }
}

/// Classifies a PMTiles refusal for the shared retry policy.
///
/// Every one is permanent, following [`crate::cog_provider`]'s rule verbatim:
/// nothing the parser reports about an archive's own bytes gets better on a
/// retry. Only the transport classifies transient failures, because only it
/// knows an HTTP status or an IO error.
pub(crate) fn classify(error: &PmtilesError) -> TileError {
    TileError::permanent(error.to_string())
}
