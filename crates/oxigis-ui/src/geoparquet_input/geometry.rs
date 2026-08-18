// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! `oxigeo_geoparquet::geometry::Geometry` (already decoded from ISO WKB or a
//! GeoArrow-native array by [`super::decode_geometries`]) → GeoJSON
//! [`GjGeometry`].
//!
//! **2-D only**, mirroring [`crate::shapefile_input`] and
//! [`crate::gpkg_input::geometry`]: `Z`/`M` are present on every
//! [`GpCoordinate`] the decoder hands back (it needs them to walk the
//! coordinate stream) and are simply never read here, because the map is a
//! Web Mercator plane and `crate::local_vector::project_position` only ever
//! looks at `[0]`/`[1]`.

use oxigeo::geojson::types::{
    Geometry as GjGeometry, GeometryCollection as GjGeometryCollection, LineString as GjLineString,
    MultiLineString as GjMultiLineString, MultiPoint as GjMultiPoint,
    MultiPolygon as GjMultiPolygon, Point as GjPoint, Polygon as GjPolygon, Position,
};
use oxigeo_geoparquet::geometry::{
    Coordinate as GpCoordinate, Geometry as GpGeometry, LineString as GpLineString,
    Point as GpPoint, Polygon as GpPolygon,
};

use oxigis_core::crs::Reprojector;

/// Converts one decoded GeoParquet geometry into GeoJSON, or [`None`] if it
/// degenerates to nothing once its coordinates are projected — a non-finite
/// no-data marker on a bare `Point`, or (recursively, for a
/// `GeometryCollection`) every member doing the same.
pub(super) fn to_geojson_geometry(geometry: &GpGeometry, crs: &Reprojector) -> Option<GjGeometry> {
    match geometry {
        GpGeometry::Point(point) => single_point(&point.coord, crs),
        GpGeometry::LineString(line) => line_string(line, crs),
        GpGeometry::Polygon(polygon) => polygon_geometry(polygon, crs),
        GpGeometry::MultiPoint(multi) => multi_point(&multi.points, crs),
        GpGeometry::MultiLineString(multi) => multi_line_string(&multi.linestrings, crs),
        GpGeometry::MultiPolygon(multi) => multi_polygon(&multi.polygons, crs),
        GpGeometry::GeometryCollection(collection) => {
            geometry_collection(&collection.geometries, crs)
        }
    }
}

/// A `[lon, lat]` GeoJSON position, or [`None`] if either ordinate is not
/// finite after projection (some producers write a sentinel like `-1e38` as a
/// no-data marker; mirrors [`crate::shapefile_input`]'s `position`).
fn position(coord: &GpCoordinate, crs: &Reprojector) -> Option<Position> {
    let (lon, lat) = crs.to_lon_lat(coord.x, coord.y)?;
    Some(vec![lon, lat])
}

/// Every finite position of a coordinate sequence, in order.
fn positions(coords: &[GpCoordinate], crs: &Reprojector) -> Vec<Position> {
    coords
        .iter()
        .filter_map(|coord| position(coord, crs))
        .collect()
}

fn single_point(coord: &GpCoordinate, crs: &Reprojector) -> Option<GjGeometry> {
    GjPoint::new(position(coord, crs)?)
        .ok()
        .map(GjGeometry::Point)
}

fn line_string(line: &GpLineString, crs: &Reprojector) -> Option<GjGeometry> {
    GjLineString::new(positions(&line.coords, crs))
        .ok()
        .map(GjGeometry::LineString)
}

/// One polygon ring (exterior or hole) as GeoJSON positions.
fn ring(line: &GpLineString, crs: &Reprojector) -> Vec<Position> {
    positions(&line.coords, crs)
}

/// One polygon's ring list — exterior first, then its holes — or [`None`] when
/// the exterior filtered away to nothing.
///
/// Ring roles are positional, so the exterior gets a guard the holes do not:
/// a polygon whose exterior lost every vertex to the non-finite filter has no
/// footprint left to draw, and handing on a ring list whose first entry is
/// empty leaves `crate::local_vector::convert_polygon` — which takes
/// `coordinates[0]` as the exterior positionally — with nothing to build from.
/// Holes are kept exactly as they came, degenerate or not, because filtering
/// *them* is what would shift a later ring into the exterior slot. (The
/// GeoPackage reader's [`crate::gpkg_input::geometry`] twin reaches the same
/// rule from the other side: it filters rings by length and therefore has to
/// refuse the polygon outright when ring 0 is the one that goes.)
fn to_gj_polygon(polygon: &GpPolygon, crs: &Reprojector) -> Option<GjPolygon> {
    let exterior = ring(&polygon.exterior, crs);
    if exterior.is_empty() {
        return None;
    }
    let mut rings = vec![exterior];
    rings.extend(polygon.interiors.iter().map(|hole| ring(hole, crs)));
    GjPolygon::new(rings).ok()
}

fn polygon_geometry(polygon: &GpPolygon, crs: &Reprojector) -> Option<GjGeometry> {
    to_gj_polygon(polygon, crs).map(GjGeometry::Polygon)
}

fn multi_point(points: &[GpPoint], crs: &Reprojector) -> Option<GjGeometry> {
    let coords: Vec<Position> = points
        .iter()
        .filter_map(|point| position(&point.coord, crs))
        .collect();
    if coords.is_empty() {
        return None;
    }
    GjMultiPoint::new(coords).ok().map(GjGeometry::MultiPoint)
}

fn multi_line_string(lines: &[GpLineString], crs: &Reprojector) -> Option<GjGeometry> {
    let parts: Vec<Vec<Position>> = lines
        .iter()
        .map(|line| positions(&line.coords, crs))
        .filter(|coords| coords.len() >= 2)
        .collect();
    if parts.is_empty() {
        return None;
    }
    GjMultiLineString::new(parts)
        .ok()
        .map(GjGeometry::MultiLineString)
}

fn multi_polygon(polygons: &[GpPolygon], crs: &Reprojector) -> Option<GjGeometry> {
    // Exactly [`to_gj_polygon`]'s rule, member by member: a member with no
    // usable exterior is dropped from the multi-polygon rather than allowed to
    // contribute an empty exterior ring.
    let parts: Vec<Vec<Vec<Position>>> = polygons
        .iter()
        .filter_map(|polygon| Some(to_gj_polygon(polygon, crs)?.coordinates))
        .collect();
    if parts.is_empty() {
        return None;
    }
    GjMultiPolygon::new(parts)
        .ok()
        .map(GjGeometry::MultiPolygon)
}

fn geometry_collection(members: &[GpGeometry], crs: &Reprojector) -> Option<GjGeometry> {
    let converted: Vec<GjGeometry> = members
        .iter()
        .filter_map(|member| to_geojson_geometry(member, crs))
        .collect();
    if converted.is_empty() {
        return None;
    }
    GjGeometryCollection::new(converted)
        .ok()
        .map(GjGeometry::GeometryCollection)
}
