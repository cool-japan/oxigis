//! Tile-byte acquisition — **without** performing any I/O.
//!
//! `oxigis-render` must build for `wasm32-unknown-unknown` as well as for
//! native targets, so it deliberately contains no HTTP client, no async
//! runtime and no filesystem access. Instead the shells inject a fetcher:
//!
//! * `oxigis-desktop` implements [`TileFetch`] over whatever native HTTP or
//!   filesystem stack it prefers;
//! * `oxigis-web` implements it over `fetch()` via `wasm-bindgen-futures`.
//!
//! What lives here is the part that is pure computation and therefore worth
//! sharing: the fetcher traits, the XYZ URL template expander and the byte
//! range type. The Cloud-Optimized GeoTIFF reader built on [`RangeFetch`] lives
//! in [`crate::cog`].

use crate::error::RenderError;
use crate::mercator::TileId;
use core::future::Future;
use core::pin::Pin;

/// Boxed future returned by the fetcher traits in this module.
///
/// On native targets this is `Pin<Box<dyn Future<Output = T> + Send + 'a>>`,
/// i.e. the boxed future is `Send` and a fetcher may be driven from a worker
/// thread. Spelled out rather than borrowed from `futures`'
/// `BoxFuture`/`LocalBoxFuture`, which are aliases for exactly these two types:
/// this crate's only other use of that dependency is in its tests, so writing
/// them here is what keeps `futures` out of a consumer's build graph.
///
/// On `wasm32` the `Send` bound is dropped, because the browser's `fetch()`
/// bridges through `wasm_bindgen_futures::JsFuture`, which is **not** `Send`;
/// requiring it there would make the web shell unable to implement these traits
/// at all. wasm32 is single-threaded, so dropping the bound costs nothing.
///
/// A caller builds one with `Box::pin(async move { .. })`.
#[cfg(not(target_arch = "wasm32"))]
pub type TileFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Boxed future returned by the fetcher traits in this module.
///
/// See the native definition for the rationale behind the `Send`-less
/// `wasm32` variant.
#[cfg(target_arch = "wasm32")]
pub type TileFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// Marker bound that is `Sync` on native targets and vacuous on `wasm32`.
///
/// A future built on top of [`TileFuture`] must be `Send` on native, which means
/// everything it borrows has to be `Sync`; on `wasm32` [`TileFuture`] is not
/// `Send` at all, so requiring `Sync` there would exclude the browser's
/// `fetch()`-backed fetchers for no benefit. Generic readers therefore bound
/// their fetcher with `RangeFetch + MaybeSync` rather than with a bare `Sync`.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSync: Sync {}

/// Blanket implementation: every `Sync` type qualifies on native targets.
#[cfg(not(target_arch = "wasm32"))]
impl<T: Sync + ?Sized> MaybeSync for T {}

/// Marker bound that is `Sync` on native targets and vacuous on `wasm32`.
///
/// See the native definition for the rationale.
#[cfg(target_arch = "wasm32")]
pub trait MaybeSync {}

/// Blanket implementation: every type qualifies on `wasm32`.
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSync for T {}

/// Something that can deliver the encoded bytes of a map tile.
///
/// "Encoded" means whatever the source serves — PNG/JPEG/WEBP for raster
/// sources, protobuf for vector ones. Decoding is the caller's job; the
/// renderer only ever ingests already-decoded RGBA through
/// [`MapRenderer::accept_tile`].
///
/// Implementations are free to be backed by the network, by a bundled asset
/// map, or by a test fixture. No bound beyond the future type is imposed, so a
/// shell can decide for itself whether its fetcher needs to be shareable.
///
/// # Shell note: keep the fetcher out of `CallbackResources`
///
/// [`TileFuture`] is `Send` on native and deliberately not on `wasm32`, so a
/// fetcher written against this trait is *not* portable into
/// `egui_wgpu::CallbackResources`: on native that map is a
/// `type_map::concurrent::TypeMap` and requires `Send + Sync`, which the wasm
/// implementation cannot provide (`survey-oxiui.md` §6). Keep the fetcher in
/// the app state that owns the fetch loop and hand results to
/// [`MapRenderer::accept_tile`]; only the [`MapRenderer`] itself belongs in the
/// callback resources.
///
/// [`MapRenderer`]: crate::renderer::MapRenderer
/// [`MapRenderer::accept_tile`]: crate::renderer::MapRenderer::accept_tile
pub trait TileFetch {
    /// Fetches the encoded bytes for `tile`.
    ///
    /// # Errors
    ///
    /// Implementations should report transport and status failures as
    /// [`RenderError::Fetch`].
    fn fetch_tile(&self, tile: TileId) -> TileFuture<'_, Result<Vec<u8>, RenderError>>;
}

/// A pure `{z}/{x}/{y}` URL template with optional subdomain rotation.
///
/// Recognised placeholders:
///
/// | Placeholder | Replaced with |
/// |---|---|
/// | `{z}` | zoom level |
/// | `{x}` | column index |
/// | `{y}` | row index, counted from the north (XYZ / Google scheme) |
/// | `{-y}` | row index counted from the south (TMS scheme), i.e. `2^z - 1 - y` |
/// | `{s}` | one of the configured subdomains |
///
/// Expansion is a plain textual substitution: no URL parsing, no escaping and
/// no network access, which keeps this usable identically on both targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XyzTemplate {
    template: String,
    subdomains: Vec<String>,
}

impl XyzTemplate {
    /// Creates a template.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTemplate`] unless the string contains
    /// `{z}`, `{x}`, and either `{y}` or `{-y}`.
    pub fn new(template: impl Into<String>) -> Result<Self, RenderError> {
        let template = template.into();
        if !template.contains("{z}") {
            return Err(RenderError::InvalidTemplate(format!(
                "missing {{z}} placeholder in {template:?}"
            )));
        }
        if !template.contains("{x}") {
            return Err(RenderError::InvalidTemplate(format!(
                "missing {{x}} placeholder in {template:?}"
            )));
        }
        if !template.contains("{y}") && !template.contains("{-y}") {
            return Err(RenderError::InvalidTemplate(format!(
                "missing {{y}} or {{-y}} placeholder in {template:?}"
            )));
        }
        Ok(Self {
            template,
            subdomains: Vec::new(),
        })
    }

    /// Attaches the subdomains that `{s}` rotates through.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTemplate`] if the list is empty or if the
    /// template has no `{s}` placeholder to fill.
    pub fn with_subdomains<I, S>(mut self, subdomains: I) -> Result<Self, RenderError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let subdomains: Vec<String> = subdomains.into_iter().map(Into::into).collect();
        if subdomains.is_empty() {
            return Err(RenderError::InvalidTemplate(
                "subdomain list must not be empty".to_owned(),
            ));
        }
        if !self.template.contains("{s}") {
            return Err(RenderError::InvalidTemplate(format!(
                "subdomains given but no {{s}} placeholder in {:?}",
                self.template
            )));
        }
        self.subdomains = subdomains;
        Ok(self)
    }

    /// The raw template string.
    #[must_use]
    pub fn template(&self) -> &str {
        &self.template
    }

    /// The configured subdomains, empty when none were set.
    #[must_use]
    pub fn subdomains(&self) -> &[String] {
        &self.subdomains
    }

    /// The subdomain `tile` maps to, or `None` when none are configured.
    ///
    /// Rotation is `(x + y) % subdomains.len()`, the scheme used by Leaflet and
    /// OpenLayers: it spreads a row of tiles across hosts while staying a pure
    /// function of the tile address, so retries hit the same host and browser
    /// caches stay warm.
    #[must_use]
    pub fn subdomain_for(&self, tile: TileId) -> Option<&str> {
        if self.subdomains.is_empty() {
            return None;
        }
        let index = (tile.x.wrapping_add(tile.y) as usize) % self.subdomains.len();
        self.subdomains.get(index).map(String::as_str)
    }

    /// Expands the template for `tile`.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidTemplate`] if the template needs a
    /// subdomain but none were configured.
    pub fn expand(&self, tile: TileId) -> Result<String, RenderError> {
        let mut url = self.template.clone();
        if url.contains("{s}") {
            let Some(subdomain) = self.subdomain_for(tile) else {
                return Err(RenderError::InvalidTemplate(format!(
                    "template {:?} uses {{s}} but no subdomains were configured",
                    self.template
                )));
            };
            url = url.replace("{s}", subdomain);
        }
        if url.contains("{-y}") {
            url = url.replace("{-y}", &tile.tms_row().to_string());
        }
        Ok(url
            .replace("{z}", &tile.z.to_string())
            .replace("{x}", &tile.x.to_string())
            .replace("{y}", &tile.y.to_string()))
    }
}

/// The TMS row for an XYZ row at zoom `z`: `2^z − 1 − y`.
///
/// Shared by the `{-y}` template expansion and by the MBTiles reader
/// (tiles v1.3 — MBTiles rows are TMS), so the two can never disagree.
///
/// # Out-of-range input
///
/// The pair `(z, y)` carries no invariant of its own, and an MBTiles file is
/// untrusted input, so both bounds are absorbed rather than trusted: `z` is
/// clamped to [`MAX_ZOOM`] the way [`TileId::tiles_per_axis`] clamps it, and a
/// `y` at or past `2^z` saturates at row `0` instead of wrapping around `u32`.
/// Callers that hold a validated tile should use [`TileId::tms_row`], where
/// neither case can arise.
///
/// [`MAX_ZOOM`]: crate::mercator::MAX_ZOOM
#[must_use]
pub const fn tms_row(z: u8, y: u32) -> u32 {
    let last = TileId::tiles_per_axis(z) - 1;
    last.saturating_sub(y)
}

/// A half-open byte range `[start, end)` for an HTTP `Range` request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteRange {
    /// First byte of the range, inclusive.
    pub start: u64,
    /// One past the last byte of the range.
    pub end: u64,
}

impl ByteRange {
    /// Creates a non-empty range.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidRange`] if `end` is not greater than
    /// `start`.
    pub fn new(start: u64, end: u64) -> Result<Self, RenderError> {
        if end <= start {
            return Err(RenderError::InvalidRange(format!(
                "end ({end}) must be greater than start ({start})"
            )));
        }
        Ok(Self { start, end })
    }

    /// Creates a range of `len` bytes beginning at `start`.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::InvalidRange`] if `len` is zero or the range
    /// would overflow `u64`.
    pub fn with_len(start: u64, len: u64) -> Result<Self, RenderError> {
        let Some(end) = start.checked_add(len) else {
            return Err(RenderError::InvalidRange(format!(
                "range {start}+{len} overflows u64"
            )));
        };
        Self::new(start, end)
    }

    /// Number of bytes covered.
    #[must_use]
    pub const fn len(&self) -> u64 {
        self.end - self.start
    }

    /// Always `false`: [`ByteRange::new`] rejects empty ranges.
    ///
    /// Provided so the type reads naturally next to `len()`.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// The value of the HTTP `Range` header for this range, e.g.
    /// `bytes=0-15`. The header is inclusive on both ends, hence the `- 1`.
    #[must_use]
    pub fn header_value(&self) -> String {
        format!("bytes={}-{}", self.start, self.end.saturating_sub(1))
    }
}

/// Something that can read arbitrary byte ranges out of a remote resource.
///
/// This is the Cloud-Optimized GeoTIFF counterpart of [`TileFetch`]: a COG is
/// read by issuing HTTP `Range` requests against a single URL rather than by
/// fetching one URL per tile.
pub trait RangeFetch {
    /// Total size of the resource in bytes, as reported by `Content-Length`.
    ///
    /// # Errors
    ///
    /// Implementations should report transport and status failures as
    /// [`RenderError::Fetch`].
    fn content_length(&self) -> TileFuture<'_, Result<u64, RenderError>>;

    /// Reads `range` from the resource.
    ///
    /// Implementations must treat a short read as an error rather than
    /// returning fewer bytes than requested — CORS-opaque responses and
    /// misbehaving proxies are exactly the failure mode this guards against
    /// (see `survey-oxigeo.md` §4.3).
    ///
    /// # Errors
    ///
    /// Implementations should report transport, status and short-read failures
    /// as [`RenderError::Fetch`].
    fn fetch_range(&self, range: ByteRange) -> TileFuture<'_, Result<Vec<u8>, RenderError>>;
}

#[cfg(test)]
mod tests {
    use super::{ByteRange, RangeFetch, TileFetch, TileFuture, XyzTemplate};
    use crate::error::RenderError;
    use crate::mercator::TileId;

    fn tile(z: u8, x: u32, y: u32) -> TileId {
        match TileId::new(z, x, y) {
            Ok(tile) => tile,
            Err(err) => panic!("tile construction failed: {err}"),
        }
    }

    fn expand(template: &XyzTemplate, tile: TileId) -> String {
        match template.expand(tile) {
            Ok(url) => url,
            Err(err) => panic!("expansion failed: {err}"),
        }
    }

    fn template(pattern: &str) -> XyzTemplate {
        match XyzTemplate::new(pattern) {
            Ok(template) => template,
            Err(err) => panic!("template rejected: {err}"),
        }
    }

    #[test]
    fn template_requires_placeholders() {
        assert!(XyzTemplate::new("https://tiles/{z}/{x}/{y}.png").is_ok());
        assert!(XyzTemplate::new("https://tiles/{z}/{x}/{-y}.png").is_ok());
        for bad in [
            "https://tiles/{x}/{y}.png",
            "https://tiles/{z}/{y}.png",
            "https://tiles/{z}/{x}.png",
        ] {
            assert!(
                matches!(XyzTemplate::new(bad), Err(RenderError::InvalidTemplate(_))),
                "{bad} should be rejected"
            );
        }
    }

    #[test]
    fn expansion_substitutes_every_placeholder() {
        let template = template("https://tiles.example/{z}/{x}/{y}.png");
        assert_eq!(
            expand(&template, tile(3, 5, 2)),
            "https://tiles.example/3/5/2.png"
        );
        assert_eq!(
            expand(&template, tile(0, 0, 0)),
            "https://tiles.example/0/0/0.png"
        );
        assert_eq!(template.template(), "https://tiles.example/{z}/{x}/{y}.png");
        assert!(template.subdomains().is_empty());
    }

    #[test]
    fn tms_row_is_flipped() {
        let template = template("https://tiles.example/{z}/{x}/{-y}.png");
        // At z=3 there are 8 rows, so row 2 from the north is row 5 from the south.
        assert_eq!(
            expand(&template, tile(3, 5, 2)),
            "https://tiles.example/3/5/5.png"
        );
        assert_eq!(
            expand(&template, tile(0, 0, 0)),
            "https://tiles.example/0/0/0.png"
        );
    }

    #[test]
    fn tms_row_absorbs_out_of_range_input() {
        use super::tms_row;
        use crate::mercator::MAX_ZOOM;

        // The in-range rule, and the agreement with the tile-level form that
        // makes the MBTiles reader and the `{-y}` expansion one implementation.
        for (z, y) in [(0u8, 0u32), (3, 2), (10, 1023), (MAX_ZOOM, 0)] {
            assert_eq!(tms_row(z, y), tile(z, 0, y).tms_row(), "z{z} y{y}");
        }

        // A row at or past `2^z` would underflow `2^z - 1 - y`: a debug-build
        // panic and a wrapped garbage row in release. It saturates instead.
        assert_eq!(tms_row(0, 1), 0);
        assert_eq!(tms_row(3, 8), 0);
        assert_eq!(tms_row(3, u32::MAX), 0);

        // A zoom past `MAX_ZOOM` answers against the clamped grid, exactly as
        // `TileId::tiles_per_axis` does — no overflow, no shift by 255.
        assert_eq!(tms_row(MAX_ZOOM + 1, 0), tms_row(MAX_ZOOM, 0));
        assert_eq!(tms_row(u8::MAX, 0), tms_row(MAX_ZOOM, 0));
    }

    #[test]
    fn subdomains_rotate_deterministically() {
        let Ok(template) =
            template("https://{s}.tiles.example/{z}/{x}/{y}.png").with_subdomains(["a", "b", "c"])
        else {
            panic!("subdomains rejected");
        };
        assert_eq!(
            template.subdomains().to_vec(),
            vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]
        );

        // (x + y) % 3
        assert_eq!(template.subdomain_for(tile(4, 0, 0)), Some("a"));
        assert_eq!(template.subdomain_for(tile(4, 1, 0)), Some("b"));
        assert_eq!(template.subdomain_for(tile(4, 0, 1)), Some("b"));
        assert_eq!(template.subdomain_for(tile(4, 1, 1)), Some("c"));
        assert_eq!(template.subdomain_for(tile(4, 3, 0)), Some("a"));

        assert_eq!(
            expand(&template, tile(4, 1, 1)),
            "https://c.tiles.example/4/1/1.png"
        );
        // The same tile always maps to the same host.
        assert_eq!(
            expand(&template, tile(4, 1, 1)),
            expand(&template, tile(4, 1, 1))
        );
    }

    #[test]
    fn subdomain_configuration_is_validated() {
        let empty: [&str; 0] = [];
        assert!(matches!(
            template("https://{s}.tiles/{z}/{x}/{y}.png").with_subdomains(empty),
            Err(RenderError::InvalidTemplate(_))
        ));
        assert!(matches!(
            template("https://tiles/{z}/{x}/{y}.png").with_subdomains(["a"]),
            Err(RenderError::InvalidTemplate(_))
        ));
        assert!(matches!(
            template("https://{s}.tiles/{z}/{x}/{y}.png").expand(tile(1, 0, 0)),
            Err(RenderError::InvalidTemplate(_))
        ));
        assert_eq!(
            template("https://tiles/{z}/{x}/{y}.png").subdomain_for(tile(1, 0, 0)),
            None
        );
    }

    #[test]
    fn byte_ranges_are_validated() {
        let Ok(range) = ByteRange::new(0, 16) else {
            panic!("range rejected");
        };
        assert_eq!(range.len(), 16);
        assert!(!range.is_empty());
        assert_eq!(range.header_value(), "bytes=0-15");

        let Ok(range) = ByteRange::with_len(1_024, 256) else {
            panic!("range rejected");
        };
        assert_eq!(
            range,
            ByteRange {
                start: 1024,
                end: 1280
            }
        );
        assert_eq!(range.header_value(), "bytes=1024-1279");

        assert!(matches!(
            ByteRange::new(10, 10),
            Err(RenderError::InvalidRange(_))
        ));
        assert!(matches!(
            ByteRange::new(10, 4),
            Err(RenderError::InvalidRange(_))
        ));
        assert!(matches!(
            ByteRange::with_len(0, 0),
            Err(RenderError::InvalidRange(_))
        ));
        assert!(matches!(
            ByteRange::with_len(u64::MAX, 2),
            Err(RenderError::InvalidRange(_))
        ));
    }

    /// A fetcher backed by an in-memory buffer — the shape a shell provides,
    /// with the I/O replaced by a slice copy.
    struct MemoryFetch {
        bytes: Vec<u8>,
    }

    impl TileFetch for MemoryFetch {
        fn fetch_tile(&self, tile: TileId) -> TileFuture<'_, Result<Vec<u8>, RenderError>> {
            Box::pin(async move {
                if tile.z == 0 {
                    Ok(self.bytes.clone())
                } else {
                    Err(RenderError::Fetch(format!("no tile {tile:?}")))
                }
            })
        }
    }

    impl RangeFetch for MemoryFetch {
        fn content_length(&self) -> TileFuture<'_, Result<u64, RenderError>> {
            Box::pin(async move { Ok(self.bytes.len() as u64) })
        }

        fn fetch_range(&self, range: ByteRange) -> TileFuture<'_, Result<Vec<u8>, RenderError>> {
            Box::pin(async move {
                let end = range.end as usize;
                let start = range.start as usize;
                self.bytes
                    .get(start..end)
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| RenderError::Fetch(format!("short read for {range:?}")))
            })
        }
    }

    #[test]
    fn injected_fetchers_drive_to_completion() {
        let fetch = MemoryFetch {
            bytes: (0u8..64).collect(),
        };
        let bytes = futures::executor::block_on(fetch.fetch_tile(tile(0, 0, 0)));
        assert_eq!(bytes.ok(), Some((0u8..64).collect::<Vec<u8>>()));
        assert!(futures::executor::block_on(fetch.fetch_tile(tile(1, 0, 0))).is_err());

        assert_eq!(
            futures::executor::block_on(fetch.content_length()).ok(),
            Some(64)
        );
        let Ok(range) = ByteRange::new(8, 12) else {
            panic!("range rejected");
        };
        assert_eq!(
            futures::executor::block_on(fetch.fetch_range(range)).ok(),
            Some(vec![8, 9, 10, 11])
        );
        let Ok(beyond) = ByteRange::new(60, 128) else {
            panic!("range rejected");
        };
        assert!(futures::executor::block_on(fetch.fetch_range(beyond)).is_err());
    }
}
