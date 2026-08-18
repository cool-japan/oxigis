#!/usr/bin/env python3
"""Generate real-SQLite GeoPackage fixtures for OxiGIS's gpkg_input tests.

Python's sqlite3 IS real SQLite, so these files exercise the genuine on-disk
format: overflow-page chains, interior b-tree pages, INTEGER PRIMARY KEY
rowid aliasing (stored as NULL in the record), quoted identifiers, etc.
Ground truth is emitted alongside as JSON for the Rust tests to assert against.
"""
import json
import math
import sqlite3
import struct
import sys
from pathlib import Path

OUT = Path(__file__).parent


def wkb_point(x, y):
    return struct.pack("<BIdd", 1, 1, x, y)


def wkb_linestring(coords):
    b = struct.pack("<BII", 1, 2, len(coords))
    for x, y in coords:
        b += struct.pack("<dd", x, y)
    return b


def wkb_polygon(rings):
    b = struct.pack("<BII", 1, 3, len(rings))
    for ring in rings:
        b += struct.pack("<I", len(ring))
        for x, y in ring:
            b += struct.pack("<dd", x, y)
    return b


def wkb_multipolygon(polys):
    b = struct.pack("<BII", 1, 6, len(polys))
    for rings in polys:
        b += wkb_polygon(rings)
    return b


def gp_blob(wkb, srs_id, envelope=None):
    """GeoPackage binary: magic GP, version 0, flags, srs_id, [envelope], wkb."""
    flags = 0x01  # little-endian header ints, no envelope
    hdr = b"GP" + bytes([0, flags])
    if envelope is not None:
        flags = 0x01 | (1 << 1)  # envelope indicator 1: XY
        hdr = b"GP" + bytes([0, flags])
        hdr += struct.pack("<i", srs_id)
        hdr += struct.pack("<dddd", *envelope)
        return hdr + wkb
    hdr += struct.pack("<i", srs_id)
    return hdr + wkb


def std_tables(cur):
    cur.execute(
        "CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER NOT NULL PRIMARY KEY,"
        " organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,"
        " definition TEXT NOT NULL, description TEXT)"
    )
    cur.executemany(
        "INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)",
        [
            ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", None),
            ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None),
            ("WGS 84 geodetic", 4326, "EPSG", 4326, 'GEOGCS["WGS 84",DATUM["WGS_1984"]]', "longitude/latitude"),
            ("Web Mercator", 3857, "EPSG", 3857, 'PROJCS["WGS 84 / Pseudo-Mercator"]', None),
            ("RGF93 / Lambert-93", 2154, "EPSG", 2154, 'PROJCS["RGF93 / Lambert-93"]', None),
        ],
    )
    cur.execute(
        "CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,"
        " identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL DEFAULT"
        " (strftime('%Y-%m-%dT%H:%M:%fZ','now')), min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,"
        " srs_id INTEGER)"
    )
    cur.execute(
        "CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,"
        " geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,"
        " CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name))"
    )


def basic():
    """basic.gpkg: multi-table, mixed SRS, big-polygon overflow row, rowid alias,
    quoted identifiers, NULL geometry, attributes-only table, unsupported SRS table."""
    path = OUT / "basic.gpkg"
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("PRAGMA page_size=4096")
    std_tables(cur)
    truth = {"tables": {}}

    # cities: points, 4326, mixed attribute types, Japanese text, quoted column
    # name with a space, NULL geometry row, fid = INTEGER PRIMARY KEY (rowid
    # alias -> stored as NULL in the record, the reader must substitute rowid).
    cur.execute(
        'CREATE TABLE cities (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB, name TEXT, "name ja" TEXT,'
        " population INTEGER, elevation REAL, active BOOLEAN, notes TEXT)"
    )
    cities = [
        (139.6917, 35.6895, "Tokyo", "東京", 13960000, 40.0, 1, None),
        (135.5023, 34.6937, "Osaka", "大阪", 2691000, 24.5, 1, "kansai"),
        (141.3545, 43.0621, "Sapporo", "札幌", 1973000, 26.0, 0, None),
    ]
    for lon, lat, name, ja, pop, elev, act, notes in cities:
        cur.execute(
            "INSERT INTO cities (geom, name, \"name ja\", population, elevation, active, notes) VALUES (?,?,?,?,?,?,?)",
            (gp_blob(wkb_point(lon, lat), 4326), name, ja, pop, elev, act, notes),
        )
    # NULL geometry row (legal per spec)
    cur.execute(
        "INSERT INTO cities (geom, name, \"name ja\", population, elevation, active, notes) VALUES (NULL,'Nowhere','どこにもない',0,0.0,0,NULL)"
    )
    cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) VALUES ('cities','features','cities',4326)")
    cur.execute("INSERT INTO gpkg_geometry_columns VALUES ('cities','geom','POINT',4326,0,0)")
    truth["tables"]["cities"] = {
        "srs_id": 4326, "geometry_type": "POINT", "feature_count": 4,
        "first_point_lonlat": [139.6917, 35.6895],
        "columns": ["fid", "geom", "name", "name ja", "population", "elevation", "active", "notes"],
        "null_geometry_rows": 1,
        "fids": [1, 2, 3, 4],
    }

    # parks: polygons with a hole; one HUGE polygon whose record exceeds one
    # 4096-byte page -> real overflow-page chain written by real SQLite.
    cur.execute("CREATE TABLE parks (fid INTEGER PRIMARY KEY, geom BLOB, name TEXT, area_ha REAL)")
    donut = [
        [(139.0, 35.0), (139.2, 35.0), (139.2, 35.2), (139.0, 35.2), (139.0, 35.0)],
        [(139.05, 35.05), (139.15, 35.05), (139.15, 35.15), (139.05, 35.15), (139.05, 35.05)],
    ]
    cur.execute("INSERT INTO parks VALUES (1, ?, 'Donut Park', 12.5)", (gp_blob(wkb_polygon(donut), 4326),))
    # big ring: 600 vertices => WKB ~ 600*16 + overhead ≈ 9.6 KB > 4061 inline max
    n = 600
    big_ring = [
        (140.0 + 0.5 * math.cos(2 * math.pi * i / (n - 1)), 36.0 + 0.5 * math.sin(2 * math.pi * i / (n - 1)))
        for i in range(n - 1)
    ]
    big_ring.append(big_ring[0])
    big = gp_blob(wkb_polygon([big_ring]), 4326)
    assert len(big) > 4096, len(big)
    cur.execute("INSERT INTO parks VALUES (2, ?, 'Big Round Park', 7853.9)", (big,))
    cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) VALUES ('parks','features','parks',4326)")
    cur.execute("INSERT INTO gpkg_geometry_columns VALUES ('parks','geom','POLYGON',4326,0,0)")
    truth["tables"]["parks"] = {
        "srs_id": 4326, "geometry_type": "POLYGON", "feature_count": 2,
        "big_row_wkb_bytes": len(big), "big_ring_vertices": n,
        "columns": ["fid", "geom", "name", "area_ha"],
        "donut_holes": 1,
    }

    # roads: linestrings in EPSG:3857 (Web Mercator) -> loader must inverse-project.
    cur.execute("CREATE TABLE roads (fid INTEGER PRIMARY KEY, geom BLOB, name TEXT)")
    # Tokyo Station ~ (15550408.8, 4257159.1) in 3857
    tokyo_3857 = (15550408.8, 4257159.1)
    osaka_3857 = (15083528.5, 4118061.5)
    cur.execute(
        "INSERT INTO roads VALUES (1, ?, 'Tokaido')",
        (gp_blob(wkb_linestring([tokyo_3857, osaka_3857]), 3857, envelope=(
            min(tokyo_3857[0], osaka_3857[0]), min(tokyo_3857[1], osaka_3857[1]),
            max(tokyo_3857[0], osaka_3857[0]), max(tokyo_3857[1], osaka_3857[1]))),),
    )
    cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) VALUES ('roads','features','roads',3857)")
    cur.execute("INSERT INTO gpkg_geometry_columns VALUES ('roads','geom','LINESTRING',3857,0,0)")
    truth["tables"]["roads"] = {
        "srs_id": 3857, "geometry_type": "LINESTRING", "feature_count": 1,
        "first_vertex_3857": list(tokyo_3857),
        "first_vertex_lonlat_approx": [139.7016, 35.6813],
        "envelope_in_gp_header": True,
        "columns": ["fid", "geom", "name"],
    }

    # regions: unsupported SRS 2154 -> loader must refuse THIS table by name, load the rest.
    cur.execute("CREATE TABLE regions (fid INTEGER PRIMARY KEY, geom BLOB, nom TEXT)")
    cur.execute(
        "INSERT INTO regions VALUES (1, ?, 'Île-de-France')",
        (gp_blob(wkb_polygon([[(650000, 6860000), (660000, 6860000), (660000, 6870000), (650000, 6860000)]]), 2154),),
    )
    cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) VALUES ('regions','features','regions',2154)")
    cur.execute("INSERT INTO gpkg_geometry_columns VALUES ('regions','geom','POLYGON',2154,0,0)")
    truth["tables"]["regions"] = {"srs_id": 2154, "refused": True, "srs_name": "RGF93 / Lambert-93"}

    # notes_attr: attributes-only table (data_type='attributes') -> skipped, not an error.
    cur.execute("CREATE TABLE notes_attr (id INTEGER PRIMARY KEY, body TEXT)")
    cur.execute("INSERT INTO notes_attr VALUES (1, 'no geometry here')")
    cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) VALUES ('notes_attr','attributes','notes',0)")
    truth["tables"]["notes_attr"] = {"data_type": "attributes", "skipped": True}

    con.commit()
    cur.execute("PRAGMA page_count")
    truth["page_count"] = cur.fetchone()[0]
    cur.execute("PRAGMA page_size")
    truth["page_size"] = cur.fetchone()[0]
    con.close()
    (OUT / "basic_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False))
    print(f"basic.gpkg: {path.stat().st_size} bytes, {truth['page_count']} pages")


def paged():
    """paged.gpkg: page_size 512 -> interior b-tree pages + low overflow threshold
    (usable-35 = 477 B) so even a 600-byte text attribute overflows."""
    path = OUT / "paged.gpkg"
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("PRAGMA page_size=512")
    std_tables(cur)
    cur.execute("CREATE TABLE pts (fid INTEGER PRIMARY KEY, geom BLOB, tag TEXT)")
    n_rows = 300
    for i in range(n_rows):
        lon = -180.0 + 360.0 * i / n_rows
        lat = -80.0 + 160.0 * i / n_rows
        tag = f"row-{i:04d}-" + ("x" * (600 if i % 37 == 0 else 8))  # some rows overflow
        cur.execute("INSERT INTO pts (geom, tag) VALUES (?, ?)", (gp_blob(wkb_point(lon, lat), 4326), tag))
    cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) VALUES ('pts','features','pts',4326)")
    cur.execute("INSERT INTO gpkg_geometry_columns VALUES ('pts','geom','POINT',4326,0,0)")
    con.commit()
    cur.execute("PRAGMA page_count")
    pages = cur.fetchone()[0]
    con.close()
    truth = {
        "page_size": 512, "page_count": pages,
        "tables": {"pts": {"srs_id": 4326, "feature_count": n_rows,
                            "overflow_rows": len([i for i in range(n_rows) if i % 37 == 0]),
                            "first_point_lonlat": [-180.0, -80.0],
                            "last_point_lonlat": [-180.0 + 360.0 * (n_rows - 1) / n_rows,
                                                   -80.0 + 160.0 * (n_rows - 1) / n_rows]}},
    }
    (OUT / "paged_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"paged.gpkg: {path.stat().st_size} bytes, {pages} pages (interior pages expected: {pages > 60})")


def without_rowid():
    """without_rowid.gpkg: a WITHOUT ROWID feature table (malformed per GPKG spec)
    -> the reader must detect the index-btree root and refuse cleanly."""
    path = OUT / "without_rowid.gpkg"
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("PRAGMA page_size=4096")
    std_tables(cur)
    cur.execute("CREATE TABLE weird (pk TEXT PRIMARY KEY, geom BLOB, v INTEGER) WITHOUT ROWID")
    cur.execute("INSERT INTO weird VALUES ('a', ?, 1)", (gp_blob(wkb_point(0.0, 0.0), 4326),))
    cur.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) VALUES ('weird','features','weird',4326)")
    cur.execute("INSERT INTO gpkg_geometry_columns VALUES ('weird','geom','POINT',4326,0,0)")
    con.commit()
    con.close()
    print(f"without_rowid.gpkg: {path.stat().st_size} bytes")


def unicode_ids():
    """unicode.gpkg: a feature table whose name and three of whose column names
    are *unquoted* non-ASCII identifiers, plus a table-level PRIMARY KEY with a
    DESC suffix.

    Both are things real SQLite accepts and a hand-rolled reader is likely to
    get wrong. Verified against SQLite 3.37.2 while writing the file:

    * SQLite's tokenizer treats every byte >= 0x80 as an identifier character,
      so `CREATE TABLE 地点 (... 名前 TEXT ...)` is stored verbatim and unquoted
      in sqlite_master (asserted below) and PRAGMA table_info reports the
      columns in declaration order.
    * a *table-level* `PRIMARY KEY (fid DESC)` over an INTEGER column is still
      a rowid alias -- unlike the column-level `INTEGER PRIMARY KEY DESC`
      form, which is not. Probed with `UPDATE t SET rowid = 999` (the declared
      column moves with the rowid) and by dumping the record bytes (the fid
      slot holds serial type 0, i.e. NULL, so only the cell's rowid has the
      value).
    """
    path = OUT / "unicode.gpkg"
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("PRAGMA page_size=4096")
    std_tables(cur)
    cur.execute(
        "CREATE TABLE 地点 (fid INTEGER, geom BLOB, 名前 TEXT, 人口 INTEGER, 標高 REAL,"
        " PRIMARY KEY (fid DESC))"
    )
    rows = [
        (1, 139.6917, 35.6895, "東京", 13960000, 40.0),
        (2, 135.5023, 34.6937, "大阪", 2691000, 24.5),
        (3, 141.3545, 43.0621, "札幌", 1973000, 26.0),
    ]
    for fid, lon, lat, name, pop, elev in rows:
        cur.execute(
            "INSERT INTO 地点 VALUES (?,?,?,?,?)",
            (fid, gp_blob(wkb_point(lon, lat), 4326), name, pop, elev),
        )
    cur.execute(
        "INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id)"
        " VALUES ('地点','features','地点',4326)"
    )
    cur.execute("INSERT INTO gpkg_geometry_columns VALUES ('地点','geom','POINT',4326,0,0)")
    con.commit()
    stored_sql = cur.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='地点'"
    ).fetchone()[0]
    assert '"' not in stored_sql and "`" not in stored_sql and "[" not in stored_sql, stored_sql
    assert "名前" in stored_sql, stored_sql
    info = cur.execute("PRAGMA table_info(地点)").fetchall()
    cur.execute("PRAGMA page_count")
    pages = cur.fetchone()[0]
    cur.execute("PRAGMA page_size")
    size = cur.fetchone()[0]
    con.close()
    truth = {
        "page_size": size,
        "page_count": pages,
        "tables": {
            "地点": {
                "srs_id": 4326,
                "geometry_type": "POINT",
                "create_sql": stored_sql,
                "feature_count": len(rows),
                "columns": [column[1] for column in info],
                "fids": [row[0] for row in rows],
                "names": [row[3] for row in rows],
                "populations": [row[4] for row in rows],
                "elevations": [row[5] for row in rows],
                "first_point_lonlat": [rows[0][1], rows[0][2]],
                "rowid_alias_column": "fid",
            }
        },
    }
    (OUT / "unicode_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False))
    print(f"unicode.gpkg: {path.stat().st_size} bytes, {pages} pages")


if __name__ == "__main__":
    # Regenerating rewrites every committed fixture (gpkg_contents.last_change
    # defaults to `now`), so run only the one you mean to replace.
    wanted = sys.argv[1:] or ["basic", "paged", "without_rowid", "unicode"]
    builders = {
        "basic": basic,
        "paged": paged,
        "without_rowid": without_rowid,
        "unicode": unicode_ids,
    }
    for key in wanted:
        builders[key]()
    print("OK")
