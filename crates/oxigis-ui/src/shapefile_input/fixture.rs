// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Test-only shapefile fixtures, synthesised in memory.
//!
//! Shared by [`super::tests`], `crate::local_input`'s tests and `crate::app`'s,
//! so all three exercise the *same* bytes. Built with `oxigeo-shapefile`'s own
//! `ShpWriter`/`DbfWriter` over a `Cursor<Vec<u8>>` — those are generic over
//! `Write`/`Seek`, unlike the path-based `ShapefileWriter` — which means a
//! format change breaks the fixtures rather than letting reader and writer
//! drift apart. No filesystem is touched, so the fixtures work on `wasm32` too.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::{Cursor, Write};

use oxigeo::shapefile::dbf::{DbfRecord, DbfWriter, FieldDescriptor, FieldType, FieldValue};
use oxigeo::shapefile::shp::shapes::{MultiPartShape, Point as ShpPoint, ShapeType};
use oxigeo::shapefile::shp::{BoundingBox, Shape, ShpWriter};

/// Serialises a `.shp` holding `shapes`, with `bbox` as its header extent.
pub(crate) fn shp_bytes(
    shape_type: ShapeType,
    bbox: (f64, f64, f64, f64),
    shapes: Vec<Shape>,
) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    let header_bbox = BoundingBox::new_2d(bbox.0, bbox.1, bbox.2, bbox.3).expect("bbox");
    {
        let mut writer = ShpWriter::new(&mut cursor, shape_type, header_bbox);
        writer.write_header().expect("shp header");
        for shape in shapes {
            writer.write_record(shape).expect("shp record");
        }
        writer.update_file_length().expect("shp length");
    }
    cursor.into_inner()
}

/// Serialises a `.dbf` with the given schema and rows.
pub(crate) fn dbf_bytes(fields: Vec<FieldDescriptor>, rows: Vec<Vec<FieldValue>>) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = DbfWriter::new(&mut cursor, fields).expect("dbf writer");
        writer.write_header().expect("dbf header");
        for row in rows {
            writer
                .write_record(&DbfRecord::new(row))
                .expect("dbf record");
        }
        writer.update_record_count().expect("dbf count");
    }
    // The 0x1A file terminator `finalize` would append; the reader stops at the
    // header's record count either way, but a well-formed table has it.
    cursor.write_all(&[0x1A]).expect("terminator");
    cursor.into_inner()
}

/// Serialises a `.dbf` like [`dbf_bytes`], but lets each row set its own DBF
/// deletion flag (marker byte `0x2A`) — the fixture the "deleted records are
/// skipped" tests build against.
pub(crate) fn dbf_bytes_with_deletions(
    fields: Vec<FieldDescriptor>,
    rows: Vec<(Vec<FieldValue>, bool)>,
) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = DbfWriter::new(&mut cursor, fields).expect("dbf writer");
        writer.write_header().expect("dbf header");
        for (values, deleted) in rows {
            writer
                .write_record(&DbfRecord { values, deleted })
                .expect("dbf record");
        }
        writer.update_record_count().expect("dbf count");
    }
    cursor.write_all(&[0x1A]).expect("terminator");
    cursor.into_inner()
}

/// A run of points from `(lon, lat)` pairs.
pub(crate) fn ring(points: &[(f64, f64)]) -> Vec<ShpPoint> {
    points.iter().map(|(x, y)| ShpPoint::new(*x, *y)).collect()
}

/// A two-ring donut: a clockwise exterior over 138..140 °E / 35..37 °N with a
/// counter-clockwise hole inside it.
pub(crate) fn donut() -> Shape {
    let mut points = ring(&[
        (138.0, 35.0),
        (138.0, 37.0),
        (140.0, 37.0),
        (140.0, 35.0),
        (138.0, 35.0),
    ]);
    points.extend(ring(&[
        (138.5, 35.5),
        (139.5, 35.5),
        (139.5, 36.5),
        (138.5, 36.5),
        (138.5, 35.5),
    ]));
    Shape::Polygon(MultiPartShape::new(vec![0, 5], points).expect("donut"))
}

/// The donut's exterior ring alone — the control for the hole test.
pub(crate) fn donut_exterior() -> Shape {
    Shape::Polygon(
        MultiPartShape::new(
            vec![0],
            ring(&[
                (138.0, 35.0),
                (138.0, 37.0),
                (140.0, 37.0),
                (140.0, 35.0),
                (138.0, 35.0),
            ]),
        )
        .expect("exterior"),
    )
}

/// The donut with every vertex re-expressed in EPSG:3857 metres, for the
/// "does the CRS conversion preserve ring winding?" test.
pub(crate) fn donut_web_mercator() -> Shape {
    let project = |(x, y): &(f64, f64)| {
        const R: f64 = 6_378_137.0;
        (
            x.to_radians() * R,
            R * (45.0 + y / 2.0).to_radians().tan().ln(),
        )
    };
    let exterior = [
        (138.0, 35.0),
        (138.0, 37.0),
        (140.0, 37.0),
        (140.0, 35.0),
        (138.0, 35.0),
    ];
    let hole = [
        (138.5, 35.5),
        (139.5, 35.5),
        (139.5, 36.5),
        (138.5, 36.5),
        (138.5, 35.5),
    ];
    let mut points: Vec<ShpPoint> = exterior
        .iter()
        .map(project)
        .map(|(x, y)| ShpPoint::new(x, y))
        .collect();
    points.extend(hole.iter().map(project).map(|(x, y)| ShpPoint::new(x, y)));
    Shape::Polygon(MultiPartShape::new(vec![0, 5], points).expect("donut"))
}

/// The canonical two-feature point set: Tokyo and Osaka with a `NAME` and a
/// `POP` column. Returns `(shp, dbf)`.
pub(crate) fn cities() -> (Vec<u8>, Vec<u8>) {
    let shp = shp_bytes(
        ShapeType::Point,
        (135.502, 34.702, 139.767, 35.681),
        vec![
            Shape::Point(ShpPoint::new(139.767, 35.681)),
            Shape::Point(ShpPoint::new(135.502, 34.702)),
        ],
    );
    let dbf = dbf_bytes(
        vec![
            FieldDescriptor::new("NAME".into(), FieldType::Character, 12, 0).expect("field"),
            FieldDescriptor::new("POP".into(), FieldType::Number, 10, 0).expect("field"),
        ],
        vec![
            vec![
                FieldValue::String("Tokyo".into()),
                FieldValue::Integer(13_960_000),
            ],
            vec![
                FieldValue::String("Osaka".into()),
                FieldValue::Integer(2_750_000),
            ],
        ],
    );
    (shp, dbf)
}

/// The `.prj` text a GDAL/ogr2ogr export of a JGD2011 plane-rectangular
/// dataset carries — WKT1 with the CRS's own `AUTHORITY["EPSG","6677"]` last,
/// after four nested ones.
pub(crate) const JGD2011_ZONE9_PRJ: &str = r#"PROJCS["JGD2011 / Japan Plane Rectangular CS IX",GEOGCS["JGD2011",DATUM["Japanese_Geodetic_Datum_2011",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","1128"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","6668"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",36],PARAMETER["central_meridian",139.833333333333],PARAMETER["scale_factor",0.9999],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Northing",NORTH],AXIS["Easting",EAST],AUTHORITY["EPSG","6677"]]"#;

/// The ESRI-flavoured `.prj` for the same zone: no `AUTHORITY` clause anywhere,
/// which is what a good deal of Japanese municipal open data actually ships.
pub(crate) const JGD2011_ZONE9_ESRI_PRJ: &str = r#"PROJCS["JGD_2011_Japan_Zone_9",GEOGCS["GCS_JGD_2011",DATUM["D_JGD_2011",SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",139.8333333333333],PARAMETER["Scale_Factor",0.9999],PARAMETER["Latitude_Of_Origin",36.0],UNIT["Meter",1.0]]"#;

/// The `.prj` GDAL writes for a **Tokyo Datum** geographic dataset, carrying
/// the `TOWGS84[…]` Helmert clause that used to make it classify as WGS 84
/// (finding 73). Kept as a fixture because that clause is the whole point.
pub(crate) const TOKYO_DATUM_PRJ: &str = r#"GEOGCS["Tokyo",DATUM["Tokyo",SPHEROID["Bessel 1841",6377397.155,299.1528128,AUTHORITY["EPSG","7004"]],TOWGS84[-146.414,507.337,680.507,0,0,0,0],AUTHORITY["EPSG","6301"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4301"]]"#;

/// Longitude of the JGD2011 zone IX origin (139°50'E).
pub(crate) const ZONE9_CENTRAL_MERIDIAN: f64 = 139.0 + 50.0 / 60.0;

/// Latitude of the JGD2011 zone IX origin (36°N).
pub(crate) const ZONE9_LATITUDE_OF_ORIGIN: f64 = 36.0;

/// One `(easting, northing)` pair in JGD2011 / Japan Plane Rectangular CS IX
/// and the WGS 84 `(lon, lat)` it must come back as.
///
/// The projected pair is the one `oxigis-core`'s reprojection tests pin against
/// the ellipsoidal Transverse Mercator series; a point near Tokyo Station, six
/// kilometres west and thirty-five kilometres south of the zone origin.
pub(crate) const ZONE9_CONTROL_POINT: ((f64, f64), (f64, f64)) = (
    (-5_995.185_165_976_9, -35_367.230_136_018_2),
    (139.7671, 35.6812),
);

/// A three-point shapefile in JGD2011 / Japan Plane Rectangular CS IX —
/// **metres, not degrees** — with a `NAME` column. Returns `(shp, dbf)`.
///
/// This is the fixture the whole CRS feature exists for: before it, a Japanese
/// municipal shapefile could not be opened at all. The three vertices are the
/// zone origin (which must invert to exactly the origin's lon/lat), the control
/// point above, and a point 20 km north-east of the origin.
pub(crate) fn jgd2011_zone9_cities() -> (Vec<u8>, Vec<u8>) {
    let ((east, north), _) = ZONE9_CONTROL_POINT;
    let points = [(0.0, 0.0), (east, north), (20_000.0, 20_000.0)];
    let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let shp = shp_bytes(
        ShapeType::Point,
        (min_x, min_y, max_x, max_y),
        points
            .iter()
            .map(|(x, y)| Shape::Point(ShpPoint::new(*x, *y)))
            .collect(),
    );
    let dbf = dbf_bytes(
        vec![FieldDescriptor::new("NAME".into(), FieldType::Character, 12, 0).expect("field")],
        vec![
            vec![FieldValue::String("origin".into())],
            vec![FieldValue::String("tokyo".into())],
            vec![FieldValue::String("northeast".into())],
        ],
    );
    (shp, dbf)
}

/// The same three vertices as one clockwise ring, so the polygon-assembly path
/// is exercised through a reprojecting CRS too.
pub(crate) fn jgd2011_zone9_polygon() -> Shape {
    Shape::Polygon(
        MultiPartShape::new(
            vec![0],
            ring(&[
                (0.0, 0.0),
                (0.0, 20_000.0),
                (20_000.0, 20_000.0),
                (20_000.0, 0.0),
                (0.0, 0.0),
            ]),
        )
        .expect("zone IX ring"),
    )
}
