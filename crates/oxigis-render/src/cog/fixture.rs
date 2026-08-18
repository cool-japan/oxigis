//! Hand-built TIFF/COG fixtures for the tests in this module.
//!
//! Test-only: there is no encoder in `oxigis-render`, and depending on one just
//! to exercise the parser would pull a whole writing stack into the graph. This
//! assembles the bytes directly instead, which also makes it possible to
//! produce files a real encoder would refuse to write — a self-referential IFD
//! chain, an internal mask in the overview position, a tile directory parked
//! past the header prefetch window.

use crate::cog::meta::{
    COMPRESSION_DEFLATE, COMPRESSION_LZW, COMPRESSION_NONE, COMPRESSION_PACKBITS,
};

/// TIFF field type `ASCII`.
const TYPE_ASCII: u16 = 2;
/// TIFF field type `SHORT`.
const TYPE_SHORT: u16 = 3;
/// TIFF field type `LONG`.
const TYPE_LONG: u16 = 4;
/// TIFF field type `DOUBLE`.
const TYPE_DOUBLE: u16 = 12;
/// BigTIFF field type `LONG8`.
const TYPE_LONG8: u16 = 16;

/// One IFD field: its tag, type, value count and already-encoded value bytes.
struct Field {
    /// TIFF tag number.
    tag: u16,
    /// TIFF field type.
    field_type: u16,
    /// Number of values.
    count: u32,
    /// Values, encoded in the fixture's byte order.
    data: Vec<u8>,
}

impl Field {
    /// Whether the values live in another part of the file.
    ///
    /// BigTIFF's inline value field is eight bytes rather than four, so the
    /// same tag can be inline in one dialect and out of line in the other.
    fn is_external(&self, big_tiff: bool) -> bool {
        self.data.len() > if big_tiff { 8 } else { 4 }
    }
}

/// A generated TIFF file plus the pixel data that went into it.
pub(crate) struct TiffFixture {
    /// The complete file.
    pub bytes: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Tile edge in pixels (square tiles).
    pub tile_edge: u32,
    /// Level-0 pixel values, row-major, `width * height` bytes.
    pub pixels: Vec<u8>,
}

impl TiffFixture {
    /// A builder with the default 8×8, 4×4-tiled, EPSG:4326 layout.
    pub(crate) fn builder() -> TiffFixtureBuilder {
        TiffFixtureBuilder::default()
    }

    /// Level-0 sample value at a pixel position.
    pub(crate) fn pixel(&self, x: u32, y: u32) -> Option<u8> {
        let index = usize::try_from(u64::from(y) * u64::from(self.width) + u64::from(x)).ok()?;
        self.pixels.get(index).copied()
    }
}

/// Options for [`TiffFixture`].
pub(crate) struct TiffFixtureBuilder {
    /// Emit an `MM` file rather than `II`.
    big_endian: bool,
    /// Write `ModelPixelScale`/`ModelTiepoint`/`GeoKeyDirectory` at all.
    georeference: bool,
    /// EPSG code to declare in the GeoKeys.
    epsg: u32,
    /// CRS coordinate of the level-0 pixel-grid origin.
    origin: (f64, f64),
    /// Level-0 pixel size in CRS units.
    pixel_size: f64,
    /// Append a half-resolution overview IFD.
    overview: bool,
    /// Mark the second IFD as an internal transparency mask.
    mask_overview: bool,
    /// Omit the tile directory tags entirely.
    drop_tile_directory: bool,
    /// Point the first IFD's next pointer back at itself.
    self_referential_chain: bool,
    /// Padding inserted between the header and the first external value block.
    directory_gap: u64,
    /// TIFF `Compression` code the tile payloads are written with.
    compression: u16,
    /// Whether tile payloads are horizontally differenced.
    predictor: bool,
    /// Lay the overview IFD before the main one, so the chain points backwards.
    overview_first: bool,
    /// Emit a BigTIFF (magic 43) rather than a classic TIFF.
    big_tiff: bool,
    /// Write the image as strips of this many rows instead of as tiles.
    rows_per_strip: Option<u32>,
    /// `GDAL_NODATA` (tag 42113) text, when the fixture declares one.
    nodata: Option<String>,
    /// Image height in pixels, so a striped fixture can be given a height that
    /// is not a multiple of its strip height.
    height: u32,
}

impl Default for TiffFixtureBuilder {
    fn default() -> Self {
        Self {
            big_endian: false,
            georeference: true,
            epsg: 4326,
            origin: (10.0, 50.0),
            pixel_size: 0.5,
            overview: true,
            mask_overview: false,
            drop_tile_directory: false,
            self_referential_chain: false,
            directory_gap: 0,
            compression: COMPRESSION_NONE,
            predictor: false,
            overview_first: false,
            big_tiff: false,
            rows_per_strip: None,
            nodata: None,
            height: 8,
        }
    }
}

impl TiffFixtureBuilder {
    /// Emits a big-endian (`MM`) file.
    pub(crate) fn big_endian(mut self, value: bool) -> Self {
        self.big_endian = value;
        self
    }

    /// Writes (or omits) the GeoTIFF georeference tags.
    pub(crate) fn georeference(mut self, value: bool) -> Self {
        self.georeference = value;
        self
    }

    /// Sets the EPSG code declared in the GeoKeys.
    pub(crate) fn epsg(mut self, value: u32) -> Self {
        self.epsg = value;
        self
    }

    /// Sets the CRS origin of the pixel grid.
    pub(crate) fn origin(mut self, x: f64, y: f64) -> Self {
        self.origin = (x, y);
        self
    }

    /// Sets the level-0 pixel size in CRS units.
    pub(crate) fn pixel_size(mut self, value: f64) -> Self {
        self.pixel_size = value;
        self
    }

    /// Appends (or omits) the half-resolution overview IFD.
    pub(crate) fn overview(mut self, value: bool) -> Self {
        self.overview = value;
        self
    }

    /// Marks the second IFD as an internal transparency mask.
    pub(crate) fn mask_overview(mut self, value: bool) -> Self {
        self.mask_overview = value;
        self
    }

    /// Omits the tile directory tags.
    pub(crate) fn drop_tile_directory(mut self, value: bool) -> Self {
        self.drop_tile_directory = value;
        self
    }

    /// Points the first IFD's next pointer back at itself.
    pub(crate) fn self_referential_chain(mut self, value: bool) -> Self {
        self.self_referential_chain = value;
        self
    }

    /// Inserts padding before the first external value block, pushing the tile
    /// directory out of the header prefetch window.
    pub(crate) fn directory_gap(mut self, value: u64) -> Self {
        self.directory_gap = value;
        self
    }

    /// Compresses the tile payloads with a TIFF `Compression` code.
    pub(crate) fn compression(mut self, value: u16) -> Self {
        self.compression = value;
        self
    }

    /// Horizontally differences the tile payloads (`Predictor` 2).
    pub(crate) fn predictor(mut self, value: bool) -> Self {
        self.predictor = value;
        self
    }

    /// Lays the overview IFD before the main one, making the next-IFD pointer
    /// go backwards — legal TIFF, and a case a naive chain walk drops.
    pub(crate) fn overview_first(mut self, value: bool) -> Self {
        self.overview_first = value;
        self
    }

    /// Declares `GDAL_NODATA` (tag 42113) with this ASCII text.
    pub(crate) fn nodata(mut self, value: &str) -> Self {
        self.nodata = Some(value.to_owned());
        self
    }

    /// Emits a BigTIFF (magic 43, 8-byte offsets, 20-byte IFD entries).
    pub(crate) fn big_tiff(mut self, value: bool) -> Self {
        self.big_tiff = value;
        self
    }

    /// Writes the image as strips of `rows` rows — `StripOffsets`/
    /// `StripByteCounts`/`RowsPerStrip` instead of the tile tags — with no
    /// overview, which is what a plain (non-COG) GeoTIFF looks like.
    ///
    /// Strips are *not* padded: a height that is not a multiple of `rows`
    /// leaves the final strip short, which is the common case.
    pub(crate) fn striped(mut self, rows: u32, height: u32) -> Self {
        self.rows_per_strip = Some(rows);
        self.height = height;
        self.overview = false;
        self
    }

    /// Assembles the file.
    pub(crate) fn build(self) -> TiffFixture {
        const WIDTH: u32 = 8;
        const TILE: u32 = 4;

        let big_endian = self.big_endian;
        let big_tiff = self.big_tiff;
        let height = self.height;
        // A deterministic gradient, distinct per pixel so a composed map tile
        // can be traced back to the source pixel it sampled.
        let pixels: Vec<u8> = (0..WIDTH * height)
            .map(|index| {
                let value = index * 3 + 7;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a deliberately wrapping test gradient"
                )]
                let byte = value as u8;
                byte
            })
            .collect();

        // A strip spans the full image width and `RowsPerStrip` rows, and —
        // unlike a tile — the last one is *not* padded to a full block.
        let (block_width, block_height) = match self.rows_per_strip {
            Some(rows) => (WIDTH, rows),
            None => (TILE, TILE),
        };
        let base_payloads = self.tile_payloads(&pixels, WIDTH, height, block_width, block_height);
        let overview_pixels = downsample(&pixels, WIDTH, height);
        let overview_payloads =
            self.tile_payloads(&overview_pixels, WIDTH / 2, height / 2, TILE, TILE);

        let base_count = base_payloads.len();
        let overview_count = overview_payloads.len();
        let want_second_ifd = self.overview || self.mask_overview;
        let base_lengths: Vec<u64> = base_payloads
            .iter()
            .map(|payload| payload.len() as u64)
            .collect();
        let overview_lengths: Vec<u64> = overview_payloads
            .iter()
            .map(|payload| payload.len() as u64)
            .collect();

        // ---- Pass 1: lay the file out ------------------------------------
        let mut base_fields =
            self.image_fields(WIDTH, height, &vec![0u64; base_count], &base_lengths, 0);
        let mut overview_fields = self.overview_fields(
            WIDTH / 2,
            height / 2,
            TILE,
            &vec![0u64; overview_count],
            &overview_lengths,
        );

        let header_bytes = if big_tiff { 16u64 } else { 8 };
        let mut position = header_bytes + self.directory_gap;
        let mut overview_ext_start = 0;
        let mut overview_ifd_pos = 0;
        // `overview_first` lays the overview's directory *before* the main one,
        // so the main IFD's next pointer goes backwards — legal TIFF that a
        // parser requiring monotonic offsets would silently truncate.
        if want_second_ifd && self.overview_first {
            overview_ext_start = position;
            position += external_bytes(&overview_fields, big_tiff);
            overview_ifd_pos = position;
            position += ifd_bytes(overview_fields.len(), big_tiff);
        }
        let base_ext_start = position;
        position += external_bytes(&base_fields, big_tiff);
        let base_ifd_pos = position;
        position += ifd_bytes(base_fields.len(), big_tiff);
        if want_second_ifd && !self.overview_first {
            overview_ext_start = position;
            position += external_bytes(&overview_fields, big_tiff);
            overview_ifd_pos = position;
            position += ifd_bytes(overview_fields.len(), big_tiff);
        }

        let mut base_offsets = Vec::with_capacity(base_count);
        for payload in &base_payloads {
            base_offsets.push(position);
            position += payload.len() as u64;
        }
        let mut overview_offsets = Vec::with_capacity(overview_count);
        if want_second_ifd {
            for payload in &overview_payloads {
                overview_offsets.push(position);
                position += payload.len() as u64;
            }
        }

        // ---- Pass 2: rebuild the fields with real block offsets -----------
        base_fields = self.image_fields(WIDTH, height, &base_offsets, &base_lengths, 0);
        overview_fields = self.overview_fields(
            WIDTH / 2,
            height / 2,
            TILE,
            &overview_offsets,
            &overview_lengths,
        );

        let next_from_base = if self.self_referential_chain {
            base_ifd_pos
        } else if want_second_ifd {
            overview_ifd_pos
        } else {
            0
        };

        let mut bytes = Vec::with_capacity(position as usize);
        bytes.extend_from_slice(if big_endian { b"MM" } else { b"II" });
        if big_tiff {
            bytes.extend_from_slice(&encode_short(43, big_endian));
            bytes.extend_from_slice(&encode_short(8, big_endian));
            bytes.extend_from_slice(&encode_short(0, big_endian));
            bytes.extend_from_slice(&encode_long8(base_ifd_pos, big_endian));
        } else {
            bytes.extend_from_slice(&encode_short(42, big_endian));
            bytes.extend_from_slice(&encode_long(
                u32::try_from(base_ifd_pos).unwrap_or(0),
                big_endian,
            ));
        }
        bytes.resize(
            usize::try_from(header_bytes + self.directory_gap).unwrap_or(8),
            0,
        );

        let serialize_overview = || {
            serialize_ifd(
                &overview_fields,
                overview_ext_start,
                0,
                big_endian,
                big_tiff,
            )
        };
        if want_second_ifd && self.overview_first {
            let (ext, ifd) = serialize_overview();
            bytes.extend_from_slice(&ext);
            bytes.extend_from_slice(&ifd);
        }
        let (ext, ifd) = serialize_ifd(
            &base_fields,
            base_ext_start,
            next_from_base,
            big_endian,
            big_tiff,
        );
        bytes.extend_from_slice(&ext);
        bytes.extend_from_slice(&ifd);
        if want_second_ifd && !self.overview_first {
            let (ext, ifd) = serialize_overview();
            bytes.extend_from_slice(&ext);
            bytes.extend_from_slice(&ifd);
        }
        for payload in &base_payloads {
            bytes.extend_from_slice(payload);
        }
        if want_second_ifd {
            for payload in &overview_payloads {
                bytes.extend_from_slice(payload);
            }
        }

        TiffFixture {
            bytes,
            width: WIDTH,
            height,
            tile_edge: block_width.min(block_height),
            pixels,
        }
    }

    /// Cuts an image into blocks and encodes each one.
    ///
    /// Tiles are padded to their full height; strips are not, because TIFF says
    /// so — the last strip of an image whose height is not a multiple of
    /// `RowsPerStrip` carries only the rows that are left.
    fn tile_payloads(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        tile_width: u32,
        tile_height: u32,
    ) -> Vec<Vec<u8>> {
        if tile_width == 0 || tile_height == 0 {
            return Vec::new();
        }
        let across = width.div_ceil(tile_width);
        let down = height.div_ceil(tile_height);
        let striped = self.rows_per_strip.is_some();
        let mut payloads = Vec::new();
        for tile_y in 0..down {
            let rows = if striped {
                tile_height.min(height - tile_y * tile_height)
            } else {
                tile_height
            };
            for tile_x in 0..across {
                let mut raw = Vec::with_capacity((tile_width * rows) as usize);
                for row in 0..rows {
                    for column in 0..tile_width {
                        let x = tile_x * tile_width + column;
                        let y = tile_y * tile_height + row;
                        let value = if x < width && y < height {
                            let index = (y * width + x) as usize;
                            pixels.get(index).copied().unwrap_or(0)
                        } else {
                            0
                        };
                        raw.push(value);
                    }
                }
                if self.predictor {
                    for row in 0..rows as usize {
                        let base = row * tile_width as usize;
                        for column in (1..tile_width as usize).rev() {
                            raw[base + column] =
                                raw[base + column].wrapping_sub(raw[base + column - 1]);
                        }
                    }
                }
                payloads.push(self.encode_payload(raw));
            }
        }
        payloads
    }

    /// Applies the fixture's compression to one raw tile.
    fn encode_payload(&self, raw: Vec<u8>) -> Vec<u8> {
        match self.compression {
            COMPRESSION_DEFLATE => oxiarc_deflate::zlib_compress(&raw, 6)
                .expect("zlib compression must succeed in a fixture"),
            COMPRESSION_LZW => {
                oxiarc_lzw::compress_tiff(&raw).expect("LZW compression must succeed in a fixture")
            }
            COMPRESSION_PACKBITS => pack_bits(&raw),
            _ => raw,
        }
    }

    /// Fields of the full-resolution IFD.
    fn image_fields(
        &self,
        width: u32,
        height: u32,
        offsets: &[u64],
        lengths: &[u64],
        subfile_type: u32,
    ) -> Vec<Field> {
        let big_endian = self.big_endian;
        let mut fields = vec![Field {
            tag: 254,
            field_type: TYPE_LONG,
            count: 1,
            data: encode_long(subfile_type, big_endian).to_vec(),
        }];
        fields.push(long_field(256, width, big_endian));
        fields.push(long_field(257, height, big_endian));
        fields.push(short_field(258, 8, big_endian));
        fields.push(short_field(259, u32::from(self.compression), big_endian));
        fields.push(short_field(262, 1, big_endian));
        fields.push(short_field(277, 1, big_endian));
        if self.predictor {
            fields.push(short_field(317, 2, big_endian));
        }
        match self.rows_per_strip {
            Some(rows) => {
                fields.push(long_field(278, rows, big_endian));
                if !self.drop_tile_directory {
                    fields.push(offset_array_field(273, offsets, big_endian, self.big_tiff));
                    fields.push(offset_array_field(279, lengths, big_endian, self.big_tiff));
                }
            }
            None => {
                fields.push(short_field(322, self.tile_edge(), big_endian));
                fields.push(short_field(323, self.tile_edge(), big_endian));
                if !self.drop_tile_directory {
                    fields.push(offset_array_field(324, offsets, big_endian, self.big_tiff));
                    fields.push(offset_array_field(325, lengths, big_endian, self.big_tiff));
                }
            }
        }
        fields.push(short_field(339, 1, big_endian));
        if let Some(nodata) = self.nodata.as_deref() {
            let mut data = nodata.as_bytes().to_vec();
            data.push(0);
            fields.push(Field {
                tag: 42_113,
                field_type: TYPE_ASCII,
                count: u32::try_from(data.len()).unwrap_or(0),
                data,
            });
        }
        if self.georeference {
            let (origin_x, origin_y) = self.origin;
            fields.push(double_array_field(
                33_550,
                &[self.pixel_size, self.pixel_size, 0.0],
                big_endian,
            ));
            fields.push(double_array_field(
                33_922,
                &[0.0, 0.0, 0.0, origin_x, origin_y, 0.0],
                big_endian,
            ));
            let key = if self.epsg == 4326 { 2048 } else { 3072 };
            let epsg = u16::try_from(self.epsg).unwrap_or(0);
            fields.push(short_array_field(
                34_735,
                &[1, 1, 0, 1, key, 0, 1, epsg],
                big_endian,
            ));
        }
        fields
    }

    /// The tile edge a tiled fixture declares.
    const fn tile_edge(&self) -> u32 {
        4
    }

    /// Fields of the second IFD: an overview, or an internal mask.
    fn overview_fields(
        &self,
        width: u32,
        height: u32,
        tile: u32,
        offsets: &[u64],
        lengths: &[u64],
    ) -> Vec<Field> {
        let subfile_type = if self.mask_overview { 4 } else { 1 };
        let mut fields = self.image_fields(width, height, offsets, lengths, subfile_type);
        if self.mask_overview {
            // A GDAL internal mask declares PhotometricInterpretation 4.
            for field in &mut fields {
                if field.tag == 262 {
                    field.data = encode_short(4, self.big_endian).to_vec();
                    field.data.extend_from_slice(&[0, 0]);
                    field.data.truncate(4);
                }
            }
        }
        // The overview's tile edge matches the base fixture's.
        debug_assert_eq!(tile, self.tile_edge());
        // Overviews carry no georeference or nodata of their own.
        fields.retain(|field| !matches!(field.tag, 33_550 | 33_922 | 34_735 | 42_113));
        fields
    }
}

/// The default fixture: an 8×8 EPSG:4326 COG, 4×4 tiles, one overview.
pub(crate) fn tiled_geo_tiff() -> TiffFixture {
    TiffFixture::builder().build()
}

/// The same 8×8 COG declared in **EPSG:32654** (WGS 84 / UTM zone 54N).
///
/// Georeferenced at 380 000 E / 3 950 000 N with 10 m pixels, i.e. an 80 m
/// square just west of Tokyo — inside zone 54 by a wide margin, so the whole
/// image projects. Because the EPSG code goes into `ProjectedCSTypeGeoKey`
/// (3072) rather than `GeographicTypeGeoKey` (2048), this also exercises the
/// projected branch of the GeoKey reader.
pub(crate) fn utm_geo_tiff() -> TiffFixture {
    TiffFixture::builder()
        .epsg(32_654)
        .origin(380_000.0, 3_950_000.0)
        .pixel_size(10.0)
        .build()
}

/// Encodes bytes as PackBits using literal runs only.
///
/// A valid (if unambitious) encoding: the decoder's repeat path is exercised
/// separately by the hand-written vectors in `codec`'s tests.
fn pack_bits(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 128 + 1);
    for chunk in raw.chunks(128) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a chunk of at most 128 bytes"
        )]
        let control = (chunk.len() - 1) as u8;
        out.push(control);
        out.extend_from_slice(chunk);
    }
    out
}

/// Halves an 8-bit image by taking the top-left pixel of each 2×2 block.
fn downsample(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(((width / 2) * (height / 2)) as usize);
    for y in 0..height / 2 {
        for x in 0..width / 2 {
            let index = ((y * 2) * width + x * 2) as usize;
            out.push(pixels.get(index).copied().unwrap_or(0));
        }
    }
    out
}

/// Encodes a 16-bit value.
fn encode_short(value: u16, big_endian: bool) -> [u8; 2] {
    if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    }
}

/// Encodes a 32-bit value.
fn encode_long(value: u32, big_endian: bool) -> [u8; 4] {
    if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    }
}

/// Encodes a 64-bit value.
fn encode_long8(value: u64, big_endian: bool) -> [u8; 8] {
    if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    }
}

/// Encodes a double.
fn encode_double(value: f64, big_endian: bool) -> [u8; 8] {
    if big_endian {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    }
}

/// A single-value `SHORT` field. TIFF left-justifies a short value inside the
/// four-byte value field, in both byte orders.
fn short_field(tag: u16, value: u32, big_endian: bool) -> Field {
    let short = u16::try_from(value).unwrap_or(u16::MAX);
    let mut data = encode_short(short, big_endian).to_vec();
    data.extend_from_slice(&[0, 0]);
    Field {
        tag,
        field_type: TYPE_SHORT,
        count: 1,
        data,
    }
}

/// A single-value `LONG` field.
fn long_field(tag: u16, value: u32, big_endian: bool) -> Field {
    Field {
        tag,
        field_type: TYPE_LONG,
        count: 1,
        data: encode_long(value, big_endian).to_vec(),
    }
}

/// A block-offset array: `LONG` in a classic TIFF, `LONG8` in a BigTIFF.
///
/// BigTIFF's whole point is that these do not fit in 32 bits, and GDAL writes
/// them as `LONG8`, so the fixture does too.
fn offset_array_field(tag: u16, values: &[u64], big_endian: bool, big_tiff: bool) -> Field {
    let mut data = Vec::with_capacity(values.len() * if big_tiff { 8 } else { 4 });
    for value in values {
        if big_tiff {
            data.extend_from_slice(&encode_long8(*value, big_endian));
        } else {
            data.extend_from_slice(&encode_long(
                u32::try_from(*value).unwrap_or(u32::MAX),
                big_endian,
            ));
        }
    }
    Field {
        tag,
        field_type: if big_tiff { TYPE_LONG8 } else { TYPE_LONG },
        count: u32::try_from(values.len()).unwrap_or(0),
        data,
    }
}

/// A `SHORT` array field.
fn short_array_field(tag: u16, values: &[u16], big_endian: bool) -> Field {
    let mut data = Vec::with_capacity(values.len() * 2);
    for value in values {
        data.extend_from_slice(&encode_short(*value, big_endian));
    }
    Field {
        tag,
        field_type: TYPE_SHORT,
        count: u32::try_from(values.len()).unwrap_or(0),
        data,
    }
}

/// A `DOUBLE` array field.
fn double_array_field(tag: u16, values: &[f64], big_endian: bool) -> Field {
    let mut data = Vec::with_capacity(values.len() * 8);
    for value in values {
        data.extend_from_slice(&encode_double(*value, big_endian));
    }
    Field {
        tag,
        field_type: TYPE_DOUBLE,
        count: u32::try_from(values.len()).unwrap_or(0),
        data,
    }
}

/// Bytes the external value blocks of `fields` occupy.
fn external_bytes(fields: &[Field], big_tiff: bool) -> u64 {
    fields
        .iter()
        .filter(|field| field.is_external(big_tiff))
        .map(|field| field.data.len() as u64)
        .sum()
}

/// Bytes one IFD of `count` entries occupies, next-IFD pointer included.
fn ifd_bytes(count: usize, big_tiff: bool) -> u64 {
    if big_tiff {
        8 + count as u64 * 20 + 8
    } else {
        2 + count as u64 * 12 + 4
    }
}

/// Serializes `fields` into `(external value block, directory)`.
fn serialize_ifd(
    fields: &[Field],
    ext_start: u64,
    next: u64,
    big_endian: bool,
    big_tiff: bool,
) -> (Vec<u8>, Vec<u8>) {
    let inline_bytes = if big_tiff { 8 } else { 4 };
    let mut ext = Vec::new();
    let mut directory = Vec::new();
    if big_tiff {
        directory.extend_from_slice(&encode_long8(fields.len() as u64, big_endian));
    } else {
        let count = u16::try_from(fields.len()).unwrap_or(u16::MAX);
        directory.extend_from_slice(&encode_short(count, big_endian));
    }
    for field in fields {
        directory.extend_from_slice(&encode_short(field.tag, big_endian));
        directory.extend_from_slice(&encode_short(field.field_type, big_endian));
        if big_tiff {
            directory.extend_from_slice(&encode_long8(u64::from(field.count), big_endian));
        } else {
            directory.extend_from_slice(&encode_long(field.count, big_endian));
        }
        if field.is_external(big_tiff) {
            let offset = ext_start + ext.len() as u64;
            if big_tiff {
                directory.extend_from_slice(&encode_long8(offset, big_endian));
            } else {
                directory.extend_from_slice(&encode_long(
                    u32::try_from(offset).unwrap_or(u32::MAX),
                    big_endian,
                ));
            }
            ext.extend_from_slice(&field.data);
        } else {
            let mut inline = field.data.clone();
            inline.resize(inline_bytes, 0);
            directory.extend_from_slice(&inline);
        }
    }
    if big_tiff {
        directory.extend_from_slice(&encode_long8(next, big_endian));
    } else {
        directory.extend_from_slice(&encode_long(u32::try_from(next).unwrap_or(0), big_endian));
    }
    (ext, directory)
}

#[cfg(test)]
mod tests {
    use super::{TiffFixture, tiled_geo_tiff};

    #[test]
    fn the_default_fixture_is_a_plausible_tiff() {
        let fixture = tiled_geo_tiff();
        assert_eq!(&fixture.bytes[..2], b"II");
        assert_eq!(fixture.width, 8);
        assert_eq!(fixture.height, 8);
        assert_eq!(fixture.tile_edge, 4);
        assert_eq!(fixture.pixels.len(), 64);
        assert_eq!(fixture.pixel(0, 0), Some(7));
        assert_eq!(fixture.pixel(1, 0), Some(10));
        assert_eq!(fixture.pixel(0, 1), Some(31));
        assert_eq!(fixture.pixel(9, 9), None);
    }

    #[test]
    fn a_gap_pushes_the_directory_out() {
        let plain = tiled_geo_tiff();
        let gapped = TiffFixture::builder().directory_gap(1_000).build();
        assert!(gapped.bytes.len() > plain.bytes.len() + 900);
    }
}
