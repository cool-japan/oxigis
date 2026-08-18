//! The layer model: [`Layer`], [`LayerKind`] and its sources, and
//! [`LayerStack`] — the ordered collection the layer-tree panel binds to.

use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};

use crate::error::CoreError;
use crate::style::LayerStyle;
use crate::util::deserialize_clamped_unit;

/// Counter backing [`LayerId::new`]. Module-scoped (rather than a
/// function-local `static` inside `new`) so [`LayerId`]'s `Deserialize` impl
/// can also advance it — see [`reserve_at_least`].
static NEXT_LAYER_ID: AtomicU64 = AtomicU64::new(1);

/// Ceiling every deserialized [`LayerId`] must stay strictly under — see
/// `LayerId`'s `Deserialize` impl, which rejects anything at or above it.
///
/// A value has no successor once it reaches `u64::MAX`, so
/// [`reserve_at_least`] could not reseed the counter *past* a loaded id that
/// large; the counter would have nowhere to go but back to (or below) ids
/// already handed out, and the next [`LayerId::new`] would collide with the
/// very layer that id names. Refusing to load such a file, rather than
/// accepting it and risking that collision, is the only sound option — and
/// it is what keeps every *accepted* id comfortably below this ceiling, so
/// `reserve_at_least` can stay a simple, always-correct `saturating_add(1)`.
///
/// Set to half of `u64::MAX` for a wide, round margin: no real session
/// mints anywhere close to that many layers.
const LAYER_ID_CEILING: u64 = u64::MAX / 2;

/// Ensures a subsequently minted [`LayerId::new`] will not collide with
/// `value`. Called whenever a `LayerId` is deserialized (see `LayerId`'s
/// manual `Deserialize` impl below), so ids loaded from a project file
/// written by a prior process — whose counter starts over at 1 in this
/// process — can never be handed out again to a *new* layer.
///
/// Every caller has already rejected `value >= LAYER_ID_CEILING`, so the
/// `saturating_add` here never actually needs to saturate — it stays purely
/// so this can never panic regardless. `fetch_max` makes this safe to call
/// in any order for any number of loaded ids: the counter only ever moves
/// forward.
fn reserve_at_least(value: u64) {
    NEXT_LAYER_ID.fetch_max(value.saturating_add(1), Ordering::Relaxed);
}

/// Stable identifier for a [`Layer`].
///
/// `LayerId` is a process-unique integer handle: once assigned to a layer it
/// never changes for the lifetime of that layer, even as the layer is
/// reordered, restyled, or toggled. It serializes as a bare JSON integer
/// (matching `#[serde(transparent)]`'s wire shape) so project files stay
/// stable and human-readable across re-saves that don't touch the
/// identified layer.
///
/// Deserializing a `LayerId` (anywhere — a [`Layer::id`] field, a
/// `BTreeMap<LayerId, _>` key such as [`crate::project::Project::styles`],
/// or a bare `LayerId`) reserves its value against future collisions with
/// [`LayerId::new`], via a module-private reservation helper — and rejects
/// (with a `Deserialize` error) a value too large to be safely reserved
/// past, rather than accepting it and risking a future collision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct LayerId(u64);

impl LayerId {
    /// Allocates a new, process-unique `LayerId`.
    ///
    /// IDs are handed out from a monotonically increasing counter starting
    /// at 1 (`0` is never returned by `new`, so it is available to callers
    /// as a sentinel if one is needed). The counter is also advanced by
    /// deserializing any `LayerId` (see the type docs), so a freshly minted
    /// id never collides with one loaded from a project file.
    ///
    /// Saturates instead of wrapping if the counter is ever driven all the
    /// way to `u64::MAX`: a plain `fetch_add` would wrap to `0` there,
    /// breaking the "never returns 0" guarantee above and reissuing ids
    /// already handed out. `LAYER_ID_CEILING` keeps this branch unreachable
    /// in practice — this is defense in depth for the case where it isn't.
    pub fn new() -> Self {
        let previous = NEXT_LAYER_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(1))
            })
            .unwrap_or(u64::MAX);
        Self(previous)
    }

    /// Returns the raw integer value of this id.
    ///
    /// Exposed for persistence/debugging. Prefer [`LayerId::new`] to mint
    /// ids in application code; use [`LayerId::from_raw`] only to
    /// reconstruct an id previously obtained from `get` (e.g. to look up a
    /// layer by an id captured earlier), not to mint fresh ids — unlike
    /// deserialization, `from_raw` does not reserve the value.
    pub fn get(self) -> u64 {
        self.0
    }

    /// Reconstructs a `LayerId` from a raw integer previously obtained via
    /// [`LayerId::get`].
    pub fn from_raw(value: u64) -> Self {
        Self(value)
    }
}

impl Default for LayerId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for LayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'de> Deserialize<'de> for LayerId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        if value >= LAYER_ID_CEILING {
            return Err(D::Error::custom(format!(
                "layer id {value} is at or above the reserved ceiling \
                 {LAYER_ID_CEILING} and cannot be loaded safely"
            )));
        }
        reserve_at_least(value);
        Ok(Self(value))
    }
}

/// Where a single-file tile archive lives.
///
/// Serializes as `{"at":"url","url":"https://…"}` /
/// `{"at":"path","path":"C:\\…"}`, following the internally-tagged idiom
/// [`RasterSource`] and [`VectorSource`] already use so an archive reference
/// reads the same way as every other source in a project file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "at", rename_all = "snake_case")]
pub enum ArchiveRef {
    /// Served over HTTP with `Range` requests.
    Url {
        /// The archive's URL. The server must allow the `Range` request
        /// header and, for a browser client, expose `Content-Range`.
        url: String,
    },
    /// A file on disk.
    ///
    /// Native shells stream it with byte-range reads; a browser has no
    /// filesystem, so there a dropped archive lives only for the session and
    /// this variant is what a project *records* rather than what it reads.
    Path {
        /// Filesystem path to the archive.
        path: String,
    },
}

impl ArchiveRef {
    /// The string a range transport addresses — a URL or a path.
    #[must_use]
    pub fn location(&self) -> &str {
        match self {
            Self::Url { url } => url,
            Self::Path { path } => path,
        }
    }

    /// The last non-empty path/URL segment, for a layer name.
    ///
    /// Falls back to the whole location when there is no separator, so a
    /// reference always names *something* the user typed.
    #[must_use]
    pub fn file_name(&self) -> &str {
        let location = self.location();
        location
            .rsplit(['/', '\\'])
            .find(|segment| !segment.is_empty())
            .unwrap_or(location)
    }
}

/// Which single-file container a [`ArchiveRef`] points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveFormat {
    /// PMTiles v3 — a tile pyramid addressed by a Hilbert index, designed to
    /// be read with byte-range requests.
    PmTiles,
    /// MBTiles — a SQLite database of tiles.
    MbTiles,
}

impl ArchiveFormat {
    /// The file extension the format is written with, without the dot.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::PmTiles => "pmtiles",
            Self::MbTiles => "mbtiles",
        }
    }

    /// The format a file name announces, if it announces one.
    #[must_use]
    pub fn from_file_name(name: &str) -> Option<Self> {
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".pmtiles") {
            Some(Self::PmTiles)
        } else if lower.ends_with(".mbtiles") {
            Some(Self::MbTiles)
        } else {
            None
        }
    }
}

/// Why a `(reference, format)` pair cannot be read, when it cannot.
///
/// The single rule both the add seam and project load consult, so a
/// combination refused at add time cannot arrive through a hand-edited project
/// file instead.
///
/// **Currently always returns [`None`]: no `(reference, format)` pair is
/// refused here today.** That is a deliberate, recorded decision, not an
/// unfinished stub — read on for why, and for where the refusals that used
/// to live here actually moved.
///
/// # No pair is refusable any more
///
/// Until tiles v1.4 this refused `Url + MbTiles` on the grounds that a SQLite
/// database "cannot be read over HTTP Range requests". That was measured and
/// found false: a paged reader opens an MBTiles archive in one 16 KiB read and
/// then costs **2.33 range requests per tile cold and 0.33 warm** (2.96 / 0.42
/// for the normalized schema) — within ~2.3× of PMTiles, and moving *less* data
/// per tile than a single planet-scale PMTiles leaf-directory miss.
///
/// The real refusals were never about the *pair*; they are about the archive's
/// own bytes, and they now land where those bytes are — at survey time, before a
/// layer exists, each by name:
///
/// * no index on `(zoom_level, tile_column, tile_row)`, which is the one thing
///   the old refusal was right about: without it, finding one tile means reading
///   the whole archive. That message names both ways out;
/// * a normalized archive with no index on `images.tile_id`;
/// * a key column with a non-`BINARY` collation, which a byte-comparing descent
///   would walk past into the wrong subtree;
/// * a `WITHOUT ROWID` table;
/// * a header this build cannot trust — an illegal page size, a reserved area
///   that leaves the payload arithmetic meaningless, UTF-16 text.
///
/// See `oxigis-ui`'s `mbtiles::paged::survey`. This function is kept — rather
/// than deleted — because it is the seam a *future* unreadable pair would be
/// refused through, and both the add seam and project load already consult it.
///
/// # Errors
///
/// [`Some`] carries the sentence to show the user; [`None`] means the pair is
/// readable.
#[must_use]
pub fn archive_refusal(archive: &ArchiveRef, format: ArchiveFormat) -> Option<String> {
    let _ = (archive, format);
    None
}

/// Where raster pixel data for a [`LayerKind::Raster`] layer comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RasterSource {
    /// XYZ/slippy-map tile URL template containing `{z}`/`{x}`/`{y}`
    /// placeholders, e.g. `https://tile.example/{z}/{x}/{y}.png`.
    Xyz {
        /// The URL template.
        url_template: String,
        /// Credit line the service's terms require, carried on the layer
        /// because the service — not its host name — is what the licence
        /// names: OpenStreetMap tiles require "© OpenStreetMap
        /// contributors", which no amount of URL parsing produces.
        ///
        /// Skipped when empty, so a project whose XYZ layer declares no
        /// attribution serializes byte-identically to one written before this
        /// field existed. Build one with [`RasterSource::xyz`] when there is
        /// no credit to carry.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        attribution: String,
    },
    /// A Cloud-Optimized GeoTIFF served over HTTP range requests.
    Cog {
        /// URL of the COG file.
        url: String,
    },
    /// A local GeoTIFF file on disk (native shells only).
    LocalGeoTiff {
        /// Filesystem path to the `.tif`/`.tiff` file.
        path: String,
    },
    /// A single-file tile archive whose tiles are images (PNG / JPEG / WebP).
    ///
    /// Which of the two it is, is decided when the archive's header is read —
    /// see `oxigis-ui`'s `archive` module — so a layer of this kind is only
    /// ever created after that answer has landed.
    TileArchive {
        /// Where the archive is.
        archive: ArchiveRef,
        /// Which container format it is.
        format: ArchiveFormat,
        /// Credit line the archive's own metadata asked for.
        ///
        /// Skipped when empty, so a project whose archive declares no
        /// attribution serializes byte-identically to one written before this
        /// field existed.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        attribution: String,
    },
}

impl RasterSource {
    /// An XYZ tile layer with no credit line — the one-line form for the
    /// construction sites that have no attribution to carry (tests, and any
    /// caller that has only a URL). Sites that DO know the service's required
    /// credit write the variant out, so the credit is never lost by defaulting.
    #[must_use]
    pub fn xyz(url_template: impl Into<String>) -> Self {
        Self::Xyz {
            url_template: url_template.into(),
            attribution: String::new(),
        }
    }
}

/// One paint rule of a [`VectorSource::MvtTiles`] layer: which layer *inside*
/// the vector tile it matches, and how that layer's features are drawn.
///
/// A vector tile is itself a bundle of named layers (`countries`, `water`,
/// `roads`, …), so a single OxiGIS layer needs a *list* of styles rather than
/// the one [`crate::style::LayerStyle`] a raster or file-backed vector layer
/// carries in [`crate::project::Project::styles`]. The rules are ordered and
/// the first match for a given source layer wins, matching the render crate's
/// `PaintTable` lookup.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorTilePaint {
    /// Name of the layer inside the vector tile this rule styles.
    pub source_layer: String,
    /// How features of that source layer are drawn.
    pub style: LayerStyle,
}

impl VectorTilePaint {
    /// Pairs `source_layer` with `style`.
    pub fn new(source_layer: impl Into<String>, style: LayerStyle) -> Self {
        Self {
            source_layer: source_layer.into(),
            style,
        }
    }
}

/// Where feature data for a [`LayerKind::Vector`] layer comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VectorSource {
    /// Mapbox Vector Tiles (MVT) served from an XYZ `{z}/{x}/{y}.pbf` URL
    /// template, styled per source layer by [`VectorTilePaint`] rules.
    MvtTiles {
        /// `{z}/{x}/{y}` URL template of the `.pbf` tiles.
        url_template: String,
        /// Per-source-layer paint rules; the first match wins.
        paints: Vec<VectorTilePaint>,
    },
    /// A local GeoJSON file on disk.
    LocalGeoJson {
        /// Filesystem path to the `.geojson`/`.json` file.
        path: String,
    },
    /// A local Shapefile on disk, referenced by its `.shp` path (sibling
    /// `.dbf`/`.shx`/`.prj`/`.cpg` files are implied).
    LocalShapefile {
        /// Filesystem path to the `.shp` file.
        path: String,
    },
    /// One feature table of a local GeoPackage on disk.
    ///
    /// A `.gpkg` holds *several* feature tables and each becomes its own
    /// layer, so — unlike every other file-backed source — the path alone does
    /// not identify the data: `table` names which one this layer was read
    /// from, and a reload that no longer finds it reports so rather than
    /// silently picking another.
    LocalGpkg {
        /// Filesystem path to the `.gpkg` file.
        path: String,
        /// Name of the feature table inside it.
        table: String,
    },
    /// A local GeoParquet file on disk, referenced by its `.parquet` /
    /// `.geoparquet` path.
    ///
    /// Reading one back needs the `geoparquet` Cargo feature (native-only —
    /// see `oxigis-ui`'s `geoparquet_input` module docs); this variant itself
    /// is unconditional so a project file that references one still
    /// round-trips through a build without the feature, reporting the layer
    /// as unavailable instead of failing to load the whole project.
    LocalGeoParquet {
        /// Filesystem path to the `.parquet` / `.geoparquet` file.
        path: String,
    },
    /// GeoJSON held entirely in memory (e.g. drag-and-drop, paste, or a
    /// project file that embeds a small dataset inline).
    InlineGeoJson {
        /// Raw GeoJSON text.
        geojson: String,
    },
    /// A single-file tile archive whose tiles are Mapbox Vector Tiles.
    ///
    /// Provider-drawn, exactly like [`VectorSource::MvtTiles`]: the tiles are
    /// styled per *source layer* by [`VectorTilePaint`] rules rather than by
    /// the single [`crate::style::LayerStyleSet`] a file-backed layer carries,
    /// so this is deliberately **not** one of the sources `oxigis-ui`'s
    /// `local_input::is_local_vector_source` recognises.
    TileArchive {
        /// Where the archive is.
        archive: ArchiveRef,
        /// Which container format it is.
        format: ArchiveFormat,
        /// Per-source-layer paint rules; the first match wins. Seeded from the
        /// archive's declared `vector_layers` when the layer is created.
        paints: Vec<VectorTilePaint>,
        /// Credit line the archive's own metadata asked for.
        ///
        /// Skipped when empty, so a project whose archive declares no
        /// attribution serializes byte-identically to one written before this
        /// field existed.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        attribution: String,
    },
}

/// What a layer renders and where its data comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "source", rename_all = "snake_case")]
pub enum LayerKind {
    /// Raster data (XYZ tiles or a Cloud-Optimized GeoTIFF).
    Raster(RasterSource),
    /// Vector data (GeoJSON or Shapefile, from a file, URL, or in memory).
    Vector(VectorSource),
}

impl LayerKind {
    /// Whether a [`crate::project::Project::styles`] entry for a layer of
    /// this kind is ever consulted by a renderer.
    ///
    /// `false` for every [`LayerKind::Raster`] (no raster source carries
    /// per-feature style) and for the two provider-drawn [`VectorSource`]
    /// variants — [`VectorSource::MvtTiles`] and [`VectorSource::TileArchive`]
    /// — which paint from their own [`VectorTilePaint`] list instead;
    /// `true` for every other (file-backed) `VectorSource`. Consulted by
    /// [`crate::project::Project::set_style`] to refuse creating a style
    /// entry that could only ever sit inert.
    #[must_use]
    pub fn accepts_layer_style(&self) -> bool {
        match self {
            LayerKind::Raster(_) => false,
            LayerKind::Vector(source) => !matches!(
                source,
                VectorSource::MvtTiles { .. } | VectorSource::TileArchive { .. }
            ),
        }
    }
}

/// The highest zoom level a scale range may name.
///
/// 24 is MapLibre's own ceiling for a style layer's `minzoom` / `maxzoom`, so a
/// range imported from a GeoLibre/MapLibre style needs no rescaling on the way
/// in. It is also what makes [`sanitize_zoom_bound`] total: a hand-edited project
/// file naming `1e30` — or an overflowing literal serde hands over as `inf` —
/// is clamped to a bound a camera can actually reach, rather than left as a
/// value no zoom will ever satisfy and no UI can display.
pub const MAX_ZOOM_LEVEL: f32 = 24.0;

/// Normalizes one end of a scale range -- the ONE rule every writer of a
/// bound goes through: the model, a loaded file, and the layer panel's range
/// editor, which asks this before deciding whether a drag changed anything.
///
/// A non-finite bound is dropped entirely rather than clamped: NaN compares
/// `false` against everything, so a NaN `min_zoom` would hide the layer at
/// every zoom for ever, and it would do it silently — the panel's range editor
/// would show a bound that reads as satisfied. A finite bound is clamped into
/// `0.0..=`[`MAX_ZOOM_LEVEL`], which is the whole domain of the field.
pub fn sanitize_zoom_bound(value: Option<f32>) -> Option<f32> {
    value
        .filter(|bound| bound.is_finite())
        .map(|bound| bound.clamp(0.0, MAX_ZOOM_LEVEL))
}

/// A `serde(deserialize_with = ...)` helper applying [`sanitize_zoom_bound`], so a
/// hand-edited or machine-generated project file cannot load a bound the rest
/// of the crate would have to keep re-checking.
fn deserialize_zoom_bound<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    Ok(sanitize_zoom_bound(Option::<f32>::deserialize(
        deserializer,
    )?))
}

/// A single map layer: a named, orderable, independently visible unit of
/// data plus the minimal rendering settings that apply to it as a whole.
///
/// Per-geometry styling (fill/line/circle/symbol) lives separately in
/// [`crate::style::LayerStyle`], keyed by [`LayerId`] at the
/// [`crate::project::Project`] level, so this struct doesn't need to know
/// about style at all.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    /// Stable identifier, assigned at creation and never reused.
    pub id: LayerId,
    /// Human-readable display name shown in the layer tree.
    pub name: String,
    /// Whether the layer is currently rendered.
    pub visible: bool,
    /// Rendering opacity. Always kept within `0.0..=1.0` — set it via
    /// [`Layer::set_opacity`], which clamps, rather than by editing the
    /// (private) field directly. A malformed value loaded from a project
    /// file is likewise clamped on deserialization.
    #[serde(deserialize_with = "deserialize_clamped_unit")]
    opacity: f32,
    /// What kind of data this layer sources and how to render it.
    pub kind: LayerKind,
    /// The coordinate reference system the layer's *source data* is stored in,
    /// when it is not WGS 84.
    ///
    /// Every coordinate that reaches the renderer is WGS 84 lon/lat — a
    /// shell reprojects at ingest, through [`crate::crs::Reprojector`] — so
    /// this is a record of **provenance, not an instruction**: it is what a
    /// layer panel shows the user ("this shapefile was in JGD2011 / Japan
    /// Plane Rectangular CS IX"), and what a Save-As or an export needs in
    /// order to say what the data originally was.
    ///
    /// Reloading a path-referenced layer deliberately does *not* consult this
    /// field: the reader re-reads the file's own `.prj` / `gpkg_spatial_ref_sys`
    /// row / GeoParquet `crs` and reprojects from that, so a file whose CRS
    /// declaration was corrected on disk loads correctly rather than being
    /// forced back into the CRS a stale project file remembers.
    ///
    /// **[`None`] means WGS 84**, and is skipped on serialization, so every
    /// `.oxigis.json` written before CRSs existed in the model round-trips
    /// byte-identically — the crate has whole-document byte-identity tests
    /// that depend on it. Declared after every field that predates it for the
    /// same reason: serde emits fields in declaration order, so a new one
    /// anywhere but at the END would move the bytes of a document that does
    /// carry it. The scale range below follows exactly the same rule, which is
    /// why it is declared after this and not before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crs: Option<crate::crs::Crs>,
    /// Lowest zoom at which the layer draws, **inclusive**, or [`None`] for no
    /// lower bound. See [`Layer::in_zoom_range`] for the interval convention
    /// and [`Layer::set_zoom_range`] for the one writer.
    ///
    /// Private, like [`Layer::opacity`], because the value has a domain: a NaN
    /// or an infinity here would hide the layer at every zoom with nothing on
    /// screen to say why. Every write goes through [`Layer::set_zoom_range`],
    /// and a value loaded from a file goes through [`deserialize_zoom_bound`].
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_zoom_bound"
    )]
    min_zoom: Option<f32>,
    /// Zoom at which the layer stops drawing, **exclusive**, or [`None`] for no
    /// upper bound. Same domain and same one writer as [`Layer::min_zoom`].
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_zoom_bound"
    )]
    max_zoom: Option<f32>,
}

impl Layer {
    /// Creates a new, visible layer at full opacity with a fresh
    /// [`LayerId`], no recorded source CRS (i.e. WGS 84) and no scale range
    /// (i.e. it draws at every zoom).
    pub fn new(name: impl Into<String>, kind: LayerKind) -> Self {
        Self {
            id: LayerId::new(),
            name: name.into(),
            visible: true,
            opacity: 1.0,
            kind,
            crs: None,
            min_zoom: None,
            max_zoom: None,
        }
    }

    /// The same layer with `crs` recorded as its source CRS.
    ///
    /// WGS 84 is recorded as [`None`] rather than as `EPSG:4326`: the two mean
    /// the same thing, and keeping the absent form for the default is what
    /// keeps a project file byte-identical to one written before this field
    /// existed.
    #[must_use]
    pub fn with_crs(mut self, crs: crate::crs::Crs) -> Self {
        self.crs = (!crs.is_wgs84()).then_some(crs);
        self
    }

    /// The layer's source CRS — the recorded one, or WGS 84 when none was.
    #[must_use]
    pub fn source_crs(&self) -> crate::crs::Crs {
        self.crs.clone().unwrap_or_default()
    }

    /// Current opacity, guaranteed to be within `0.0..=1.0`.
    pub fn opacity(&self) -> f32 {
        self.opacity
    }

    /// Sets the opacity, clamping the input into `0.0..=1.0`.
    pub fn set_opacity(&mut self, value: f32) {
        self.opacity = crate::util::clamp_unit(value);
    }

    /// Lowest zoom at which the layer draws (inclusive), or [`None`] for no
    /// lower bound. Guaranteed finite and within `0.0..=`[`MAX_ZOOM_LEVEL`].
    #[must_use]
    pub fn min_zoom(&self) -> Option<f32> {
        self.min_zoom
    }

    /// Zoom at which the layer stops drawing (exclusive), or [`None`] for no
    /// upper bound. Same guarantee as [`Layer::min_zoom`].
    #[must_use]
    pub fn max_zoom(&self) -> Option<f32> {
        self.max_zoom
    }

    /// Sets the scale range, sanitizing both ends (see [`sanitize_zoom_bound`]).
    ///
    /// Both ends move together deliberately: they are one user-facing fact
    /// ("this layer draws between these zooms"), and a setter per end would
    /// let a caller record half of a change on the undo stack — the same
    /// argument [`crate::project::Project`]'s basemap pair is built on.
    ///
    /// An inverted range (`min >= max`) is stored as given rather than
    /// swapped or refused: it means "never draws", which is a state the user
    /// can see in the panel and drag straight back out of, whereas a silent
    /// swap would move a bound they did not touch.
    pub fn set_zoom_range(&mut self, min: Option<f32>, max: Option<f32>) {
        self.min_zoom = sanitize_zoom_bound(min);
        self.max_zoom = sanitize_zoom_bound(max);
    }

    /// The same layer with `min`/`max` recorded as its scale range — the
    /// builder twin of [`Layer::set_zoom_range`], matching [`Layer::with_crs`].
    #[must_use]
    pub fn with_zoom_range(mut self, min: Option<f32>, max: Option<f32>) -> Self {
        self.set_zoom_range(min, max);
        self
    }

    /// Whether `zoom` falls inside this layer's scale range, **ignoring**
    /// [`Layer::visible`].
    ///
    /// The interval is half-open — `min <= zoom < max` — which is MapLibre's
    /// convention and, more importantly, the only one that makes the standard
    /// pair of layers correct: a generalized outline with `max_zoom = 14` and
    /// the detailed cadastre it hands over to with `min_zoom = 14` must draw
    /// exactly one of themselves at z14, not both. An absent bound is no bound.
    #[must_use]
    pub fn in_zoom_range(&self, zoom: f64) -> bool {
        self.min_zoom.is_none_or(|min| zoom >= f64::from(min))
            && self.max_zoom.is_none_or(|max| zoom < f64::from(max))
    }

    /// Whether the layer draws at `zoom` — [`Layer::visible`] **and**
    /// [`Layer::in_zoom_range`].
    ///
    /// This is the predicate a renderer's layer filter wants: the checkbox and
    /// the scale range are two ways of saying the same thing to the map, and a
    /// filter that consults only one of them draws a layer the project says is
    /// off.
    #[must_use]
    pub fn visible_at(&self, zoom: f64) -> bool {
        self.visible && self.in_zoom_range(zoom)
    }
}

/// Ordered collection of [`Layer`]s, back-to-front (`layers()[0]` is painted
/// first / at the bottom; the last entry is painted last / on top).
///
/// This is the model a layer-tree panel binds to: every mutation that can
/// fail (an unknown [`LayerId`]) returns [`CoreError::LayerNotFound`]
/// instead of panicking.
///
/// Serializes transparently as a plain JSON array of [`Layer`] (no wrapper
/// object), since the struct exists only to carry ordering behavior, not
/// extra wire fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LayerStack {
    layers: Vec<Layer>,
}

impl LayerStack {
    /// Creates an empty stack.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Appends a new layer on top (painted last / frontmost) and returns
    /// its id.
    pub fn add(&mut self, layer: Layer) -> LayerId {
        let id = layer.id;
        self.layers.push(layer);
        id
    }

    /// Inserts a new layer at `index` (clamped to `0..=len()`) and returns
    /// its id. `index == len()` behaves like [`Self::add`] (append on top,
    /// i.e. frontmost); `index == 0` inserts at the very back.
    pub fn insert(&mut self, index: usize, layer: Layer) -> LayerId {
        let id = layer.id;
        let index = index.min(self.layers.len());
        self.layers.insert(index, layer);
        id
    }

    /// Removes the layer with the given id, returning it if it was present.
    pub fn remove(&mut self, id: LayerId) -> Option<Layer> {
        let index = self.layers.iter().position(|l| l.id == id)?;
        Some(self.layers.remove(index))
    }

    /// Returns a reference to the layer with the given id, if present.
    pub fn get(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    /// Returns a mutable reference to the layer with the given id, if
    /// present.
    ///
    /// [`Layer::id`] is a public field reachable through this reference —
    /// nothing here stops a caller from overwriting it to collide with
    /// another layer already in the stack (or with a future
    /// [`LayerId::new`]). Never reassign a `Layer::id`; construct one only
    /// via [`Layer::new`], or accept a value already reserved by
    /// deserializing.
    pub fn get_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        self.layer_mut(id).ok()
    }

    /// All layers, back-to-front (see [`LayerStack`] docs for ordering).
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Number of layers currently in the stack.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the stack has no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Moves the layer one step toward the front (painted later, i.e. on
    /// top of layers below it). Returns `Ok(true)` if it moved, `Ok(false)`
    /// if it was already frontmost.
    pub fn move_up(&mut self, id: LayerId) -> Result<bool, CoreError> {
        let index = self.index_of(id)?;
        if index + 1 >= self.layers.len() {
            return Ok(false);
        }
        self.layers.swap(index, index + 1);
        Ok(true)
    }

    /// Moves the layer one step toward the back (painted earlier, i.e.
    /// underneath layers above it). Returns `Ok(true)` if it moved,
    /// `Ok(false)` if it was already at the back.
    pub fn move_down(&mut self, id: LayerId) -> Result<bool, CoreError> {
        let index = self.index_of(id)?;
        if index == 0 {
            return Ok(false);
        }
        self.layers.swap(index, index - 1);
        Ok(true)
    }

    /// Moves the layer to `index`, preserving the relative order of every
    /// other layer. `index` addresses the stack *after* the layer is lifted
    /// out, so it is clamped to `0..=len() - 1`: `index == 0` sends the
    /// layer to the back (painted first); `index >= len() - 1` — including
    /// `usize::MAX` — brings it all the way to the front (painted last / on
    /// top). Moving a layer to its own current index is a no-op.
    pub fn move_to(&mut self, id: LayerId, index: usize) -> Result<(), CoreError> {
        let current = self.index_of(id)?;
        let layer = self.layers.remove(current);
        let index = index.min(self.layers.len());
        self.layers.insert(index, layer);
        Ok(())
    }

    /// Flips the layer's visibility flag, returning the new value.
    pub fn toggle_visibility(&mut self, id: LayerId) -> Result<bool, CoreError> {
        let layer = self.layer_mut(id)?;
        layer.visible = !layer.visible;
        Ok(layer.visible)
    }

    /// Sets the layer's visibility to an ABSOLUTE value, returning whether it
    /// actually changed.
    ///
    /// The absolute form is what a reversible operation needs:
    /// [`Self::toggle_visibility`] applied twice is a no-op only if nothing
    /// else moved in between, so an undo/redo applier built on it would flip
    /// the wrong way the moment two writers disagreed. This one is idempotent
    /// — the same rule the stack's whole-order reorder follows.
    pub fn set_visibility(&mut self, id: LayerId, value: bool) -> Result<bool, CoreError> {
        let layer = self.layer_mut(id)?;
        let changed = layer.visible != value;
        layer.visible = value;
        Ok(changed)
    }

    /// Renames the layer, returning whether the name actually changed.
    pub fn rename(&mut self, id: LayerId, name: impl Into<String>) -> Result<bool, CoreError> {
        let name = name.into();
        let layer = self.layer_mut(id)?;
        let changed = layer.name != name;
        layer.name = name;
        Ok(changed)
    }

    /// Sets the layer's scale range, sanitized (see [`Layer::set_zoom_range`]).
    pub fn set_zoom_range(
        &mut self,
        id: LayerId,
        min: Option<f32>,
        max: Option<f32>,
    ) -> Result<(), CoreError> {
        self.layer_mut(id)?.set_zoom_range(min, max);
        Ok(())
    }

    /// Sets the layer's opacity, clamped into `0.0..=1.0`.
    pub fn set_opacity(&mut self, id: LayerId, value: f32) -> Result<(), CoreError> {
        self.layer_mut(id)?.set_opacity(value);
        Ok(())
    }

    fn index_of(&self, id: LayerId) -> Result<usize, CoreError> {
        self.layers
            .iter()
            .position(|l| l.id == id)
            .ok_or(CoreError::LayerNotFound(id))
    }

    fn layer_mut(&mut self, id: LayerId) -> Result<&mut Layer, CoreError> {
        self.layers
            .iter_mut()
            .find(|l| l.id == id)
            .ok_or(CoreError::LayerNotFound(id))
    }

    /// Checks that every layer in the stack has a distinct [`LayerId`].
    ///
    /// Every id-keyed lookup here (`get`, `get_mut`, and the private
    /// `index_of`/`layer_mut` helpers) resolves only the *first* match for a
    /// given id, so a second layer sharing that id would be permanently
    /// unreachable through this API even though it is still in the stack —
    /// every remove/toggle/restyle meant for it would silently land on its
    /// earlier twin instead. Called by
    /// [`crate::project::Project::from_json_string`] so a hand-edited or
    /// corrupt project file is refused at load rather than loaded into that
    /// state.
    pub fn validate_unique_ids(&self) -> Result<(), CoreError> {
        let mut seen = HashSet::with_capacity(self.layers.len());
        for layer in &self.layers {
            if !seen.insert(layer.id) {
                return Err(CoreError::DuplicateLayerId(layer.id));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xyz_layer(name: &str) -> Layer {
        Layer::new(
            name,
            LayerKind::Raster(RasterSource::xyz("https://tile.example/{z}/{x}/{y}.png")),
        )
    }

    #[test]
    fn layer_ids_are_unique_and_stable() {
        let a = LayerId::new();
        let b = LayerId::new();
        assert_ne!(a, b);
        assert_eq!(a, a);
    }

    #[test]
    fn deserializing_a_layer_id_reserves_it_against_future_collisions() {
        // Simulates loading a project written by a prior process, whose
        // `LayerId` counter had already advanced well past this process's
        // fresh `NEXT_LAYER_ID` (which starts at 1 every run).
        let loaded: LayerId = serde_json::from_str("500").expect("deserialize");
        let fresh = LayerId::new();
        assert_ne!(fresh, loaded);
        assert!(
            fresh.get() > loaded.get(),
            "fresh id {fresh} did not advance past loaded id {loaded}"
        );
    }

    #[test]
    fn stack_deserialized_from_a_prior_process_avoids_id_collisions_on_add() {
        // `LayerStack` is `#[serde(transparent)]`: a plain JSON array.
        let json = r#"[
            {
                "id": 1000,
                "name": "loaded-a",
                "visible": true,
                "opacity": 1.0,
                "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}
            },
            {
                "id": 1002,
                "name": "loaded-b",
                "visible": true,
                "opacity": 1.0,
                "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}
            }
        ]"#;
        let mut stack: LayerStack = serde_json::from_str(json).expect("deserialize");
        let new_id = stack.add(xyz_layer("new-in-this-process"));

        let ids: Vec<u64> = stack.layers().iter().map(|l| l.id.get()).collect();
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), ids.len(), "duplicate LayerId in {ids:?}");
        assert!(new_id.get() > 1002);
    }

    #[test]
    fn a_layer_id_at_or_above_the_ceiling_is_rejected_rather_than_risking_a_collision() {
        // These values have no (or barely any) room to reseed the counter
        // past them — see `LAYER_ID_CEILING`'s docs — so they must be
        // refused rather than accepted and left to wrap `NEXT_LAYER_ID`
        // into collision with an id already handed out.
        for hostile in [u64::MAX, u64::MAX - 1, LAYER_ID_CEILING] {
            let result: Result<LayerId, _> = serde_json::from_str(&hostile.to_string());
            assert!(result.is_err(), "{hostile} must be rejected");
        }
    }

    #[test]
    fn a_layer_id_just_under_the_ceiling_loads_and_reserves_without_colliding() {
        // Deliberately advances the process-wide `NEXT_LAYER_ID` counter far
        // past every other test's range; safe under `cargo nextest` (each
        // test its own process) per this crate's verification policy. Only
        // relative properties are asserted, never an absolute minted value.
        let loaded: LayerId = serde_json::from_str(&(LAYER_ID_CEILING - 1).to_string())
            .expect("just under the ceiling must still load");
        let fresh = LayerId::new();
        assert_ne!(fresh, loaded);
        assert!(
            fresh.get() > loaded.get(),
            "fresh id {fresh} did not advance past loaded id {loaded}"
        );
    }

    #[test]
    fn layer_id_serde_is_a_bare_integer() {
        let id = LayerId::from_raw(7);
        let json = serde_json::to_string(&id).expect("serialize");
        assert_eq!(json, "7");
        let back: LayerId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, id);
    }

    #[test]
    fn new_layer_is_visible_and_opaque() {
        let layer = xyz_layer("basemap");
        assert!(layer.visible);
        assert_eq!(layer.opacity(), 1.0);
    }

    #[test]
    fn set_opacity_clamps_out_of_range_values() {
        let mut layer = xyz_layer("basemap");
        layer.set_opacity(5.0);
        assert_eq!(layer.opacity(), 1.0);
        layer.set_opacity(-3.0);
        assert_eq!(layer.opacity(), 0.0);
        layer.set_opacity(0.42);
        assert_eq!(layer.opacity(), 0.42);
    }

    #[test]
    fn deserializing_a_layer_clamps_out_of_range_opacity() {
        let json = r#"{
            "id": 1,
            "name": "basemap",
            "visible": true,
            "opacity": 42.0,
            "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}
        }"#;
        let layer: Layer = serde_json::from_str(json).expect("deserialize");
        assert_eq!(layer.opacity(), 1.0);
    }

    #[test]
    fn stack_add_get_remove_roundtrip() {
        let mut stack = LayerStack::new();
        let id = stack.add(xyz_layer("basemap"));
        assert_eq!(stack.len(), 1);
        assert!(!stack.is_empty());
        assert_eq!(stack.get(id).map(|l| l.name.as_str()), Some("basemap"));

        let removed = stack.remove(id).expect("layer present");
        assert_eq!(removed.id, id);
        assert!(stack.is_empty());
        assert!(stack.get(id).is_none());
    }

    #[test]
    fn insert_places_a_layer_at_the_given_index_and_clamps_out_of_range() {
        let mut stack = LayerStack::new();
        stack.add(xyz_layer("bottom"));
        stack.add(xyz_layer("top"));

        let middle_id = stack.insert(1, xyz_layer("middle"));
        let names: Vec<&str> = stack.layers().iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["bottom", "middle", "top"]);
        assert_eq!(
            stack.get(middle_id).map(|l| l.name.as_str()),
            Some("middle")
        );

        // An out-of-range index clamps to the end, same as `add`.
        stack.insert(100, xyz_layer("overflow"));
        let names: Vec<&str> = stack.layers().iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["bottom", "middle", "top", "overflow"]);
    }

    #[test]
    fn get_mut_allows_mutating_a_layer_in_place_and_reports_missing_ids() {
        let mut stack = LayerStack::new();
        let id = stack.add(xyz_layer("basemap"));
        stack.get_mut(id).expect("layer present").name = "renamed".to_string();
        assert_eq!(stack.get(id).map(|l| l.name.as_str()), Some("renamed"));
        assert!(stack.get_mut(LayerId::from_raw(999)).is_none());
    }

    #[test]
    fn validate_unique_ids_accepts_a_stack_with_no_duplicates() {
        let mut stack = LayerStack::new();
        stack.add(xyz_layer("a"));
        stack.add(xyz_layer("b"));
        assert!(stack.validate_unique_ids().is_ok());
    }

    #[test]
    fn validate_unique_ids_rejects_a_repeated_id() {
        // `LayerStack`'s own deserialize is structural only (no uniqueness
        // check), so this loads successfully and produces exactly the
        // corrupt state `validate_unique_ids` exists to catch.
        let json = r#"[
            {"id": 5, "name": "a", "visible": true, "opacity": 1.0,
             "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}},
            {"id": 5, "name": "b", "visible": true, "opacity": 1.0,
             "kind": {"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}}
        ]"#;
        let stack: LayerStack = serde_json::from_str(json).expect("structurally valid JSON");
        assert_eq!(
            stack.validate_unique_ids(),
            Err(CoreError::DuplicateLayerId(LayerId::from_raw(5)))
        );
    }

    #[test]
    fn stack_operations_report_missing_layer() {
        let mut stack = LayerStack::new();
        let ghost = LayerId::from_raw(999);
        assert_eq!(stack.move_up(ghost), Err(CoreError::LayerNotFound(ghost)));
        assert_eq!(stack.move_down(ghost), Err(CoreError::LayerNotFound(ghost)));
        assert_eq!(
            stack.move_to(ghost, 0),
            Err(CoreError::LayerNotFound(ghost))
        );
        assert_eq!(
            stack.toggle_visibility(ghost),
            Err(CoreError::LayerNotFound(ghost))
        );
        assert_eq!(
            stack.set_visibility(ghost, true),
            Err(CoreError::LayerNotFound(ghost))
        );
        assert_eq!(
            stack.rename(ghost, "gone"),
            Err(CoreError::LayerNotFound(ghost))
        );
        assert_eq!(
            stack.set_zoom_range(ghost, Some(4.0), None),
            Err(CoreError::LayerNotFound(ghost))
        );
        assert_eq!(
            stack.set_opacity(ghost, 0.5),
            Err(CoreError::LayerNotFound(ghost))
        );
        assert!(stack.remove(ghost).is_none());
    }

    #[test]
    fn move_up_and_down_reorder_and_report_boundary() {
        let mut stack = LayerStack::new();
        let bottom = stack.add(xyz_layer("bottom"));
        let _middle = stack.add(xyz_layer("middle"));
        let top = stack.add(xyz_layer("top"));

        // Already frontmost: no-op.
        assert_eq!(stack.move_up(top), Ok(false));
        // Already at the back: no-op.
        assert_eq!(stack.move_down(bottom), Ok(false));

        // Move "bottom" to the front.
        assert_eq!(stack.move_up(bottom), Ok(true));
        let names: Vec<&str> = stack.layers().iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["middle", "bottom", "top"]);

        // Move "top" toward the back.
        assert_eq!(stack.move_down(top), Ok(true));
        let names: Vec<&str> = stack.layers().iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["middle", "top", "bottom"]);
    }

    #[test]
    fn move_to_relocates_a_layer_and_clamps_the_target_index() {
        let mut stack = LayerStack::new();
        let bottom = stack.add(xyz_layer("bottom"));
        let middle = stack.add(xyz_layer("middle"));
        let _top = stack.add(xyz_layer("top"));

        // `usize::MAX` clamps to the front.
        stack.move_to(bottom, usize::MAX).expect("layer present");
        let names: Vec<&str> = stack.layers().iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["middle", "top", "bottom"]);

        // 0 sends the (now frontmost) layer all the way to the back.
        stack.move_to(bottom, 0).expect("layer present");
        let names: Vec<&str> = stack.layers().iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["bottom", "middle", "top"]);

        // Moving a layer to its own current index leaves everyone in place.
        stack.move_to(middle, 1).expect("layer present");
        let names: Vec<&str> = stack.layers().iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["bottom", "middle", "top"]);
    }

    #[test]
    fn toggle_visibility_flips_and_returns_new_state() {
        let mut stack = LayerStack::new();
        let id = stack.add(xyz_layer("basemap"));
        assert_eq!(stack.toggle_visibility(id), Ok(false));
        assert!(!stack.get(id).expect("present").visible);
        assert_eq!(stack.toggle_visibility(id), Ok(true));
        assert!(stack.get(id).expect("present").visible);
    }

    #[test]
    fn set_opacity_through_stack_clamps() {
        let mut stack = LayerStack::new();
        let id = stack.add(xyz_layer("basemap"));
        stack.set_opacity(id, -1.0).expect("layer present");
        assert_eq!(stack.get(id).expect("present").opacity(), 0.0);
    }

    #[test]
    fn mvt_tile_layer_round_trips_with_its_paint_rules() {
        use crate::style::{Color, FillStyle, LayerStyle, LineStyle};

        let kind = LayerKind::Vector(VectorSource::MvtTiles {
            url_template: "https://demotiles.maplibre.org/tiles/{z}/{x}/{y}.pbf".to_string(),
            paints: vec![
                VectorTilePaint::new(
                    "countries",
                    LayerStyle::Fill(FillStyle::new(Color::from_rgb(0xE8, 0xDF, 0xC8))),
                ),
                VectorTilePaint::new(
                    "geolines",
                    LayerStyle::Line(LineStyle::new(Color::from_rgb(0x88, 0x88, 0x88), 0.6)),
                ),
            ],
        });
        let value: serde_json::Value = serde_json::to_value(&kind).expect("serialize to Value");
        assert_eq!(value["kind"], "vector");
        assert_eq!(value["source"]["type"], "mvt_tiles");
        assert_eq!(value["source"]["paints"][0]["source_layer"], "countries");
        assert_eq!(value["source"]["paints"][0]["style"]["type"], "fill");
        let round_tripped: LayerKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(round_tripped, kind);
    }

    #[test]
    fn a_gpkg_layer_round_trips_and_older_project_files_still_load() {
        let kind = LayerKind::Vector(VectorSource::LocalGpkg {
            path: "/data/tokyo.gpkg".to_string(),
            table: "cities".to_string(),
        });
        let value: serde_json::Value = serde_json::to_value(&kind).expect("serialize to Value");
        assert_eq!(value["kind"], "vector");
        assert_eq!(value["source"]["type"], "local_gpkg");
        assert_eq!(value["source"]["path"], "/data/tokyo.gpkg");
        assert_eq!(value["source"]["table"], "cities");
        let round_tripped: LayerKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(round_tripped, kind);

        // The variant is additive: a project file written before it existed
        // must keep loading unchanged.
        let older = r#"{"kind":"vector","source":{"type":"local_shapefile","path":"/d/a.shp"}}"#;
        let loaded: LayerKind = serde_json::from_str(older).expect("deserialize");
        assert!(matches!(
            loaded,
            LayerKind::Vector(VectorSource::LocalShapefile { .. })
        ));
    }

    #[test]
    fn a_geoparquet_layer_round_trips_and_older_project_files_still_load() {
        let kind = LayerKind::Vector(VectorSource::LocalGeoParquet {
            path: "/data/tokyo.parquet".to_string(),
        });
        let value: serde_json::Value = serde_json::to_value(&kind).expect("serialize to Value");
        assert_eq!(value["kind"], "vector");
        assert_eq!(value["source"]["type"], "local_geo_parquet");
        assert_eq!(value["source"]["path"], "/data/tokyo.parquet");
        let round_tripped: LayerKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(round_tripped, kind);

        // The variant is additive: a project file written before it existed
        // must keep loading unchanged.
        let older = r#"{"kind":"vector","source":{"type":"local_gpkg","path":"/d/a.gpkg","table":"cities"}}"#;
        let loaded: LayerKind = serde_json::from_str(older).expect("deserialize");
        assert!(matches!(
            loaded,
            LayerKind::Vector(VectorSource::LocalGpkg { .. })
        ));
    }

    #[test]
    fn a_raster_tile_archive_round_trips_with_its_reference_and_format() {
        let kind = LayerKind::Raster(RasterSource::TileArchive {
            archive: ArchiveRef::Url {
                url: "https://example.test/basemap.pmtiles".to_string(),
            },
            format: ArchiveFormat::PmTiles,
            attribution: String::new(),
        });
        let value: serde_json::Value = serde_json::to_value(&kind).expect("serialize to Value");
        assert_eq!(value["kind"], "raster");
        assert_eq!(value["source"]["type"], "tile_archive");
        assert_eq!(value["source"]["format"], "pm_tiles");
        assert_eq!(value["source"]["archive"]["at"], "url");
        assert_eq!(
            value["source"]["archive"]["url"],
            "https://example.test/basemap.pmtiles"
        );
        assert!(
            value["source"].get("attribution").is_none(),
            "an empty credit line is skipped, so old and new files agree byte for byte"
        );
        let round_tripped: LayerKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(round_tripped, kind);
    }

    #[test]
    fn a_vector_tile_archive_round_trips_with_its_paints_and_credit() {
        use crate::style::{Color, FillStyle, LayerStyle};

        let kind = LayerKind::Vector(VectorSource::TileArchive {
            archive: ArchiveRef::Path {
                path: "C:\\data\\tokyo.mbtiles".to_string(),
            },
            format: ArchiveFormat::MbTiles,
            paints: vec![VectorTilePaint::new(
                "earth",
                LayerStyle::Fill(FillStyle::new(Color::from_rgb(0xE9, 0xE4, 0xD8))),
            )],
            attribution: "\u{a9} Example".to_string(),
        });
        let value: serde_json::Value = serde_json::to_value(&kind).expect("serialize to Value");
        assert_eq!(value["kind"], "vector");
        assert_eq!(value["source"]["type"], "tile_archive");
        assert_eq!(value["source"]["format"], "mb_tiles");
        assert_eq!(value["source"]["archive"]["at"], "path");
        assert_eq!(
            value["source"]["archive"]["path"],
            "C:\\data\\tokyo.mbtiles"
        );
        assert_eq!(value["source"]["paints"][0]["source_layer"], "earth");
        assert_eq!(value["source"]["attribution"], "\u{a9} Example");
        let round_tripped: LayerKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(round_tripped, kind);
    }

    #[test]
    fn the_archive_variants_are_additive_so_older_files_still_load() {
        // Written before either variant existed; both enums must still parse
        // every one of their prior shapes unchanged.
        for older in [
            r#"{"kind":"raster","source":{"type":"cog","url":"https://x/a.tif"}}"#,
            r#"{"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}"#,
            r#"{"kind":"vector","source":{"type":"local_gpkg","path":"/d/a.gpkg","table":"cities"}}"#,
            r#"{"kind":"vector","source":{"type":"mvt_tiles","url_template":"https://x/{z}/{x}/{y}.pbf","paints":[]}}"#,
        ] {
            let loaded: LayerKind = serde_json::from_str(older).expect("deserialize");
            let again = serde_json::to_string(&loaded).expect("serialize");
            assert_eq!(
                again, older,
                "an untouched layer must re-save byte-identically"
            );
        }
    }

    #[test]
    fn an_xyz_layer_round_trips_with_its_credit_and_skips_an_empty_one() {
        let credited = LayerKind::Raster(RasterSource::Xyz {
            url_template: "https://tile.example/{z}/{x}/{y}.png".to_string(),
            attribution: "\u{a9} OpenStreetMap contributors".to_string(),
        });
        let value: serde_json::Value = serde_json::to_value(&credited).expect("serialize to Value");
        assert_eq!(value["source"]["type"], "xyz");
        assert_eq!(
            value["source"]["attribution"],
            "\u{a9} OpenStreetMap contributors"
        );
        let round_tripped: LayerKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(round_tripped, credited);

        // The empty credit is the byte-compatibility case: `xyz()` builds it,
        // and it must serialize exactly as the pre-field shape did.
        let plain = LayerKind::Raster(RasterSource::xyz("https://x/{z}/{x}/{y}.png"));
        assert_eq!(
            serde_json::to_string(&plain).expect("serialize"),
            r#"{"kind":"raster","source":{"type":"xyz","url_template":"https://x/{z}/{x}/{y}.png"}}"#
        );
    }

    #[test]
    fn an_archive_written_without_the_credit_field_still_loads() {
        // The exact shape the plan specified before `attribution` was added:
        // `serde(default)` is what keeps it loadable.
        let json = r#"{"kind":"raster","source":{"type":"tile_archive","archive":{"at":"url","url":"https://x/a.pmtiles"},"format":"pm_tiles"}}"#;
        let loaded: LayerKind = serde_json::from_str(json).expect("deserialize");
        assert_eq!(serde_json::to_string(&loaded).expect("serialize"), json);
    }

    #[test]
    fn an_archive_reference_reports_its_location_and_file_name() {
        let url = ArchiveRef::Url {
            url: "https://example.test/tiles/basemap.pmtiles".to_string(),
        };
        assert_eq!(url.location(), "https://example.test/tiles/basemap.pmtiles");
        assert_eq!(url.file_name(), "basemap.pmtiles");

        let path = ArchiveRef::Path {
            path: "C:\\data\\tokyo.mbtiles".to_string(),
        };
        assert_eq!(path.file_name(), "tokyo.mbtiles");

        // No separator at all: the whole thing is the name.
        let bare = ArchiveRef::Path {
            path: "tiles.pmtiles".to_string(),
        };
        assert_eq!(bare.file_name(), "tiles.pmtiles");

        // A trailing separator must not produce an empty name.
        let trailing = ArchiveRef::Url {
            url: "https://example.test/a.pmtiles/".to_string(),
        };
        assert_eq!(trailing.file_name(), "a.pmtiles");
    }

    #[test]
    fn a_format_is_recognised_from_a_file_name_case_insensitively() {
        assert_eq!(
            ArchiveFormat::from_file_name("Basemap.PMTiles"),
            Some(ArchiveFormat::PmTiles)
        );
        assert_eq!(
            ArchiveFormat::from_file_name("tokyo.mbtiles"),
            Some(ArchiveFormat::MbTiles)
        );
        assert_eq!(ArchiveFormat::from_file_name("scene.tif"), None);
        assert_eq!(ArchiveFormat::PmTiles.extension(), "pmtiles");
        assert_eq!(ArchiveFormat::MbTiles.extension(), "mbtiles");
    }

    #[test]
    fn no_reference_and_format_pair_is_refusable_any_more() {
        // Tiles v1.4 flipped this: a `.mbtiles` URL is READ, a page at a time,
        // and the refusals that matter moved to survey time where the archive's
        // own bytes answer. `oxigis-ui`'s `mbtiles::paged::tests` pins each of
        // them by name — no index, no `images.tile_id` index, a non-BINARY
        // collation, WITHOUT ROWID, an untrustworthy header — each *before* a
        // layer exists, which is what this seam used to buy at the pair level.
        let remote = ArchiveRef::Url {
            url: "https://example.test/tokyo.mbtiles".to_string(),
        };
        let local = ArchiveRef::Path {
            path: "/data/tokyo.mbtiles".to_string(),
        };
        for (archive, format) in [
            (&remote, ArchiveFormat::MbTiles),
            (&remote, ArchiveFormat::PmTiles),
            (&local, ArchiveFormat::MbTiles),
            (&local, ArchiveFormat::PmTiles),
        ] {
            assert!(
                archive_refusal(archive, format).is_none(),
                "{archive:?} + {format:?} is readable"
            );
        }
    }

    #[test]
    fn layer_kind_serializes_with_expected_tag_shape() {
        let kind = LayerKind::Vector(VectorSource::InlineGeoJson {
            geojson: "{\"type\":\"FeatureCollection\",\"features\":[]}".to_string(),
        });
        let value: serde_json::Value = serde_json::to_value(&kind).expect("serialize to Value");
        assert_eq!(value["kind"], "vector");
        assert_eq!(value["source"]["type"], "inline_geo_json");
        let round_tripped: LayerKind = serde_json::from_value(value).expect("deserialize");
        assert_eq!(round_tripped, kind);
    }

    #[test]
    fn accepts_layer_style_is_true_only_for_file_backed_vector_sources() {
        // No raster source has per-feature style, tile archive included.
        assert!(
            !LayerKind::Raster(RasterSource::xyz("https://x/{z}/{x}/{y}.png"))
                .accepts_layer_style()
        );
        assert!(
            !LayerKind::Raster(RasterSource::TileArchive {
                archive: ArchiveRef::Url {
                    url: "https://x/a.pmtiles".to_string()
                },
                format: ArchiveFormat::PmTiles,
                attribution: String::new(),
            })
            .accepts_layer_style()
        );

        // File-backed vector sources resolve through `Project::styles`.
        assert!(
            LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "a.geojson".to_string()
            })
            .accepts_layer_style()
        );
        assert!(
            LayerKind::Vector(VectorSource::InlineGeoJson {
                geojson: "{}".to_string()
            })
            .accepts_layer_style()
        );

        // Provider-drawn vector sources paint from their own list instead.
        assert!(
            !LayerKind::Vector(VectorSource::MvtTiles {
                url_template: "https://x/{z}/{x}/{y}.pbf".to_string(),
                paints: vec![],
            })
            .accepts_layer_style()
        );
        assert!(
            !LayerKind::Vector(VectorSource::TileArchive {
                archive: ArchiveRef::Path {
                    path: "a.mbtiles".to_string()
                },
                format: ArchiveFormat::MbTiles,
                paints: vec![],
                attribution: String::new(),
            })
            .accepts_layer_style()
        );
    }

    // ---- the source CRS field ---------------------------------------------

    #[test]
    fn a_fresh_layer_records_no_crs_and_reads_back_as_wgs84() {
        let layer = Layer::new(
            "Roads",
            LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "roads.geojson".to_string(),
            }),
        );
        assert_eq!(layer.crs, None);
        assert!(layer.source_crs().is_wgs84());
    }

    #[test]
    fn with_crs_records_a_projected_source_and_still_drops_wgs84() {
        let kind = LayerKind::Vector(VectorSource::LocalShapefile {
            path: "tokyo.shp".to_string(),
        });
        let projected =
            Layer::new("Tokyo", kind.clone()).with_crs(crate::crs::Crs::from_epsg(6677));
        assert_eq!(
            projected.crs.as_ref().map(crate::crs::Crs::epsg),
            Some(6677)
        );
        assert_eq!(projected.source_crs().epsg(), 6677);

        // WGS 84 is the absent form, so a layer that WAS in WGS 84 serializes
        // exactly as one written before the field existed.
        let plain = Layer::new("Plain", kind).with_crs(crate::crs::Crs::wgs84());
        assert_eq!(plain.crs, None);
        assert!(plain.source_crs().is_wgs84());
    }

    #[test]
    fn a_layer_without_a_crs_serializes_byte_identically_to_a_pre_crs_document() {
        // THE gate for the new field: a document written before CRSs existed in
        // the model must come back out of this build with the same bytes.
        // `crs` is last in the struct AND skipped when absent, so nothing moves.
        //
        // The fixture is built from a layer with no CRS rather than written out
        // by hand, so it stays correct if any *other* field of `Layer` changes
        // shape — what is being measured here is that the round trip moves no
        // byte and that the `crs` key is absent, not the spelling of `kind`.
        let layer = Layer {
            id: LayerId::from_raw(7),
            name: "Roads".to_string(),
            visible: true,
            opacity: 1.0,
            kind: LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "roads.geojson".to_string(),
            }),
            crs: None,
            min_zoom: None,
            max_zoom: None,
        };
        let older = serde_json::to_string(&layer).expect("serialize");
        assert!(!older.contains("crs"), "{older}");

        let loaded: Layer = serde_json::from_str(&older).expect("a pre-CRS layer loads");
        assert_eq!(loaded.crs, None);
        let again = serde_json::to_string(&loaded).expect("re-serialize");
        assert_eq!(
            again, older,
            "an untouched layer must re-save byte-identically",
        );

        // And through a whole project document, which is where a `#[serde]`
        // field-ordering mistake actually shows up.
        let mut project = crate::project::Project::new("P");
        project.layers.add(layer);
        let pretty = project.to_json_string().expect("pretty");
        assert!(!pretty.contains("\"crs\""), "{pretty}");
        let reloaded =
            crate::project::Project::from_json_string(&pretty).expect("the document parses");
        assert_eq!(
            reloaded.to_json_string().expect("re-serialize"),
            pretty,
            "adding the CRS field must not move one byte of a WGS 84 project",
        );
    }

    #[test]
    fn a_layer_with_a_crs_round_trips_with_the_key_last() {
        let layer = Layer {
            id: LayerId::from_raw(3),
            name: "Tokyo".to_string(),
            visible: true,
            opacity: 1.0,
            kind: LayerKind::Vector(VectorSource::LocalShapefile {
                path: "tokyo.shp".to_string(),
            }),
            crs: Some(crate::crs::Crs::from_epsg(6677)),
            min_zoom: None,
            max_zoom: None,
        };
        let json = serde_json::to_string(&layer).expect("serialize");
        assert!(json.ends_with(r#""crs":{"epsg":6677}}"#), "{json}");
        let back: Layer = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, layer);
        assert_eq!(
            serde_json::to_string(&back).expect("re-serialize"),
            json,
            "re-saving is byte-stable",
        );
    }

    // ---- the per-layer scale range ---------------------------------------

    #[test]
    fn a_fresh_layer_has_no_scale_range_and_draws_at_every_zoom() {
        let layer = xyz_layer("Roads");
        assert_eq!(layer.min_zoom(), None);
        assert_eq!(layer.max_zoom(), None);
        for zoom in [0.0, 2.0, 13.9, 22.0] {
            assert!(layer.in_zoom_range(zoom), "unbounded must draw at {zoom}");
            assert!(layer.visible_at(zoom));
        }
    }

    #[test]
    fn the_scale_range_is_half_open_so_a_handover_pair_never_double_draws() {
        // THE reason for the convention: a generalized outline that stops at
        // z14 and the detailed cadastre that starts there must draw exactly
        // one of themselves at z14.
        let outline = xyz_layer("Outline").with_zoom_range(None, Some(14.0));
        let cadastre = xyz_layer("Cadastre").with_zoom_range(Some(14.0), None);
        for zoom in [0.0, 13.0, 13.999] {
            assert!(outline.in_zoom_range(zoom), "outline at {zoom}");
            assert!(!cadastre.in_zoom_range(zoom), "cadastre at {zoom}");
        }
        for zoom in [14.0, 14.001, 22.0] {
            assert!(!outline.in_zoom_range(zoom), "outline at {zoom}");
            assert!(cadastre.in_zoom_range(zoom), "cadastre at {zoom}");
        }
        // At the handover zoom exactly one of the two draws — never both,
        // never neither.
        assert_ne!(outline.in_zoom_range(14.0), cadastre.in_zoom_range(14.0));
    }

    #[test]
    fn visible_at_is_the_checkbox_and_the_range_together() {
        let mut layer = xyz_layer("Cadastre").with_zoom_range(Some(14.0), Some(18.0));
        assert!(layer.visible_at(15.0));
        assert!(!layer.visible_at(10.0), "out of range, though visible");
        layer.visible = false;
        assert!(!layer.visible_at(15.0), "hidden, though in range");
        assert!(
            layer.in_zoom_range(15.0),
            "the range half must stay readable on its own \u{2014} the panel \
             greys an out-of-range row without touching the checkbox"
        );
    }

    #[test]
    fn an_inverted_range_means_never_draws_rather_than_being_silently_swapped() {
        let layer = xyz_layer("Nothing").with_zoom_range(Some(18.0), Some(14.0));
        assert_eq!(layer.min_zoom(), Some(18.0));
        assert_eq!(layer.max_zoom(), Some(14.0));
        for zoom in [0.0, 14.0, 16.0, 18.0, 22.0] {
            assert!(!layer.in_zoom_range(zoom), "inverted must refuse {zoom}");
        }
    }

    #[test]
    fn a_bound_is_sanitized_on_the_way_in_so_no_layer_hides_itself_invisibly() {
        let mut layer = xyz_layer("Roads");
        // NaN would compare false against every zoom for ever.
        layer.set_zoom_range(Some(f32::NAN), Some(f32::INFINITY));
        assert_eq!(layer.min_zoom(), None);
        assert_eq!(layer.max_zoom(), None);
        // Finite values are clamped into the field's whole domain.
        layer.set_zoom_range(Some(-5.0), Some(1.0e9));
        assert_eq!(layer.min_zoom(), Some(0.0));
        assert_eq!(layer.max_zoom(), Some(MAX_ZOOM_LEVEL));
        // And through the stack's mutator, which is the same rule.
        let mut stack = LayerStack::new();
        let id = stack.add(xyz_layer("Through the stack"));
        stack
            .set_zoom_range(id, Some(f32::NEG_INFINITY), Some(12.5))
            .expect("layer present");
        let stored = stack.get(id).expect("present");
        assert_eq!(stored.min_zoom(), None);
        assert_eq!(stored.max_zoom(), Some(12.5));
    }

    #[test]
    fn a_layer_without_a_scale_range_serializes_byte_identically_to_an_older_document() {
        // THE gate for the two new fields, exactly as the CRS field has: both
        // are declared last and skipped when absent, so a project written
        // before scale ranges existed re-saves without one byte moving.
        let layer = Layer {
            id: LayerId::from_raw(11),
            name: "Roads".to_string(),
            visible: true,
            opacity: 1.0,
            kind: LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "roads.geojson".to_string(),
            }),
            crs: None,
            min_zoom: None,
            max_zoom: None,
        };
        let older = serde_json::to_string(&layer).expect("serialize");
        assert!(!older.contains("zoom"), "{older}");
        let loaded: Layer = serde_json::from_str(&older).expect("a pre-range layer loads");
        assert_eq!(loaded.min_zoom(), None);
        assert_eq!(loaded.max_zoom(), None);
        assert_eq!(
            serde_json::to_string(&loaded).expect("re-serialize"),
            older,
            "an untouched layer must re-save byte-identically",
        );

        // And through a whole project document, where a field-ordering
        // mistake actually shows up.
        let mut project = crate::project::Project::new("P");
        project.layers.add(layer);
        let pretty = project.to_json_string().expect("pretty");
        assert!(
            !pretty.contains("min_zoom") && !pretty.contains("max_zoom"),
            "{pretty}"
        );
        let reloaded =
            crate::project::Project::from_json_string(&pretty).expect("the document parses");
        assert_eq!(
            reloaded.to_json_string().expect("re-serialize"),
            pretty,
            "adding the scale range must not move one byte of a rangeless project",
        );
    }

    #[test]
    fn a_layer_with_a_scale_range_round_trips_with_the_keys_last_and_in_order() {
        let layer = Layer {
            id: LayerId::from_raw(12),
            name: "Cadastre".to_string(),
            visible: true,
            opacity: 1.0,
            kind: LayerKind::Vector(VectorSource::LocalShapefile {
                path: "cadastre.shp".to_string(),
            }),
            crs: Some(crate::crs::Crs::from_epsg(6677)),
            min_zoom: Some(14.0),
            max_zoom: Some(18.0),
        };
        let json = serde_json::to_string(&layer).expect("serialize");
        assert!(
            json.ends_with(r#""crs":{"epsg":6677},"min_zoom":14.0,"max_zoom":18.0}"#),
            "{json}"
        );
        let back: Layer = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, layer);
        assert_eq!(
            serde_json::to_string(&back).expect("re-serialize"),
            json,
            "re-saving is byte-stable",
        );

        // Only one end declared is the common import shape, and it must skip
        // the other key rather than write a null.
        let one_ended = xyz_layer("Detail").with_zoom_range(Some(14.0), None);
        let json = serde_json::to_string(&one_ended).expect("serialize");
        assert!(json.contains(r#""min_zoom":14.0"#), "{json}");
        assert!(!json.contains("max_zoom"), "{json}");
    }

    #[test]
    fn a_hostile_zoom_bound_in_a_file_is_sanitized_at_load_rather_than_stored() {
        // `null` is the absent form; an out-of-domain number is clamped; a
        // huge literal serde hands over as an infinity is dropped entirely.
        let base = serde_json::to_string(&xyz_layer("Roads")).expect("serialize");
        let hostile = base.replace(r#""kind":"#, r#""min_zoom":-40.0,"max_zoom":1e40,"kind":"#);
        assert_ne!(hostile, base, "the fixture really did gain the keys");
        let loaded: Layer = serde_json::from_str(&hostile).expect("loads");
        assert_eq!(loaded.min_zoom(), Some(0.0));
        assert!(
            matches!(loaded.max_zoom(), None | Some(MAX_ZOOM_LEVEL)),
            "an over-range upper bound is clamped or dropped, never stored: {:?}",
            loaded.max_zoom()
        );

        let nulled = base.replace(r#""kind":"#, r#""min_zoom":null,"kind":"#);
        let loaded: Layer = serde_json::from_str(&nulled).expect("loads");
        assert_eq!(loaded.min_zoom(), None);
    }

    #[test]
    fn set_visibility_is_absolute_and_reports_whether_it_moved() {
        let mut stack = LayerStack::new();
        let id = stack.add(xyz_layer("basemap"));
        // Idempotent, which `toggle_visibility` is not — this is exactly the
        // property an undo/redo applier is built on.
        assert_eq!(stack.set_visibility(id, false), Ok(true));
        assert_eq!(stack.set_visibility(id, false), Ok(false));
        assert!(!stack.get(id).expect("present").visible);
        assert_eq!(stack.set_visibility(id, true), Ok(true));
        assert!(stack.get(id).expect("present").visible);
    }

    #[test]
    fn rename_replaces_the_name_and_reports_whether_it_moved() {
        let mut stack = LayerStack::new();
        let id = stack.add(xyz_layer("before"));
        assert_eq!(stack.rename(id, "after"), Ok(true));
        assert_eq!(stack.get(id).map(|l| l.name.as_str()), Some("after"));
        assert_eq!(stack.rename(id, "after"), Ok(false));
        // An empty name is the caller's business to refuse: the model stores
        // what it is told, so an undo can restore a name a future import
        // decides is legal.
        assert_eq!(stack.rename(id, ""), Ok(true));
        assert_eq!(stack.get(id).map(|l| l.name.as_str()), Some(""));
    }

    #[test]
    fn a_crs_key_with_no_epsg_loads_as_wgs84_because_the_field_defaults() {
        // The additive rule, exercised through a `Layer`: a `crs` object with
        // no `epsg` key means WGS 84.
        let layer = Layer {
            id: LayerId::from_raw(9),
            name: "L".to_string(),
            visible: true,
            opacity: 1.0,
            kind: LayerKind::Vector(VectorSource::LocalGeoJson {
                path: "a.geojson".to_string(),
            }),
            crs: Some(crate::crs::Crs::from_epsg(6677)),
            min_zoom: None,
            max_zoom: None,
        };
        let json = serde_json::to_string(&layer).expect("serialize");
        let stripped = json.replace(r#""crs":{"epsg":6677}"#, r#""crs":{}"#);
        assert_ne!(stripped, json, "the fixture really did lose its epsg key");
        let loaded: Layer = serde_json::from_str(&stripped).expect("loads");
        assert_eq!(
            loaded.crs.as_ref().map(crate::crs::Crs::epsg),
            Some(crate::crs::EPSG_WGS84),
        );
        assert!(loaded.source_crs().is_wgs84());
    }
}
