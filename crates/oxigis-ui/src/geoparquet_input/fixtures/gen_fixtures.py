#!/usr/bin/env python3
"""Generate GeoParquet test fixtures via pyarrow 24 (real parquet writer).

Covers: WKB encoding, CRS84-default / PROJJSON-3857 / unsupported-CRS refusal,
codec matrix (uncompressed/snappy/brotli/lz4 readable; zstd/gzip must produce
the "disabled codec" error), covering.bbox column exclusion, null geometry,
mixed attribute types incl. Japanese text. Truth JSON alongside.
"""
import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).parent


def wkb_point(x, y):
    return struct.pack("<BIdd", 1, 1, x, y)


def wkb_polygon(rings):
    b = struct.pack("<BII", 1, 3, len(rings))
    for ring in rings:
        b += struct.pack("<I", len(ring))
        for x, y in ring:
            b += struct.pack("<dd", x, y)
    return b


def geo_meta(crs=None, covering=None):
    col = {"encoding": "WKB", "geometry_types": ["Point", "Polygon"]}
    if crs is not None:
        col["crs"] = crs
    if covering is not None:
        col["covering"] = covering
    return json.dumps({"version": "1.1.0", "primary_column": "geometry", "columns": {"geometry": col}})


PROJJSON_3857 = {
    "$schema": "https://proj.org/schemas/v0.7/projjson.schema.json",
    "type": "ProjectedCRS",
    "name": "WGS 84 / Pseudo-Mercator",
    "id": {"authority": "EPSG", "code": 3857},
}
PROJJSON_2154 = {
    "type": "ProjectedCRS",
    "name": "RGF93 v1 / Lambert-93",
    "id": {"authority": "EPSG", "code": 2154},
}


def base_table(coords):
    geoms = [wkb_point(x, y) for x, y in coords[:-1]] + [None]  # last row: null geometry
    return {
        "geometry": pa.array(geoms, type=pa.binary()),
        "name": pa.array(["Tokyo", "Osaka", "札幌", None], type=pa.string()),
        "population": pa.array([13960000, 2691000, 1973000, None], type=pa.int64()),
        "elevation": pa.array([40.0, 24.5, 26.0, None], type=pa.float32()),
        "active": pa.array([True, True, False, None], type=pa.bool_()),
    }


def write(path, cols, meta_json, compression):
    schema = pa.schema([pa.field(k, v.type) for k, v in cols.items()], metadata={b"geo": meta_json.encode()})
    table = pa.Table.from_arrays(list(cols.values()), schema=schema)
    pq.write_table(table, path, compression=compression)
    return path.stat().st_size


truth = {}
wgs_coords = [(139.6917, 35.6895), (135.5023, 34.6937), (141.3545, 43.0621), None]

# 1. codec matrix, all WGS84-default (no crs key): must decode (except zstd/gzip)
for codec, fname, readable in [
    ("NONE", "uncompressed.parquet", True),
    ("SNAPPY", "snappy.parquet", True),
    ("BROTLI", "brotli.parquet", True),
    ("LZ4", "lz4.parquet", True),          # pyarrow LZ4 = lz4_raw (Hadoop-style LZ4_RAW codec)
    ("ZSTD", "zstd.parquet", False),
    ("GZIP", "gzip.parquet", False),
]:
    size = write(OUT / fname, base_table(wgs_coords), geo_meta(), codec)
    truth[fname] = {
        "codec": codec, "readable": readable, "rows": 4, "null_geometry_rows": 1,
        "first_point_lonlat": [139.6917, 35.6895],
        "columns": ["name", "population", "elevation", "active"],
        "japanese_name_row": 2, "bytes": size,
    }

# 2. PROJJSON EPSG:3857 — mercator-metre coordinates, must inverse-project
m_coords = [(15550408.8, 4257159.1), (15083528.5, 4118061.5), (15736060.1, 5312683.6), None]
write(OUT / "projjson_3857.parquet", base_table(m_coords), geo_meta(crs=PROJJSON_3857), "SNAPPY")
truth["projjson_3857.parquet"] = {
    "crs": "EPSG:3857", "rows": 4, "first_point_3857": [15550408.8, 4257159.1],
}

# 3. unsupported CRS — must refuse naming it
write(OUT / "unsupported_crs.parquet", base_table(wgs_coords), geo_meta(crs=PROJJSON_2154), "SNAPPY")
truth["unsupported_crs.parquet"] = {"refused": True, "crs_name": "RGF93 v1 / Lambert-93"}

# 4. covering.bbox struct column named "bbox" — must NOT appear in properties
cols = base_table(wgs_coords)
xs = [c[0] if c else None for c in wgs_coords]
ys = [c[1] if c else None for c in wgs_coords]
cols["bbox"] = pa.StructArray.from_arrays(
    [pa.array(xs, type=pa.float64()), pa.array(ys, type=pa.float64()),
     pa.array(xs, type=pa.float64()), pa.array(ys, type=pa.float64())],
    names=["xmin", "ymin", "xmax", "ymax"],
)
covering = {"bbox": {"xmin": ["bbox", "xmin"], "ymin": ["bbox", "ymin"],
                      "xmax": ["bbox", "xmax"], "ymax": ["bbox", "ymax"]}}
write(OUT / "covering_bbox.parquet", cols, geo_meta(covering=covering), "SNAPPY")
truth["covering_bbox.parquet"] = {"rows": 4, "bbox_column_excluded": True,
                                    "columns": ["name", "population", "elevation", "active"]}

# 5. polygon + point mixed, with a donut hole (WKB polygon path)
donut = wkb_polygon([
    [(139.0, 35.0), (139.2, 35.0), (139.2, 35.2), (139.0, 35.2), (139.0, 35.0)],
    [(139.05, 35.05), (139.15, 35.05), (139.15, 35.15), (139.05, 35.15), (139.05, 35.05)],
])
cols = {
    "geometry": pa.array([wkb_point(139.7, 35.7), donut], type=pa.binary()),
    "kind": pa.array(["pt", "donut"], type=pa.string()),
}
write(OUT / "mixed_geom.parquet", cols, geo_meta(), "SNAPPY")
truth["mixed_geom.parquet"] = {"rows": 2, "donut_holes": 1}

(OUT / "truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False))

# verify: pyarrow reads back every readable file with identical row counts
for fname, t in truth.items():
    if not fname.endswith(".parquet"):
        continue
    pf = pq.ParquetFile(OUT / fname)
    codecs = {pf.metadata.row_group(0).column(i).compression for i in range(pf.metadata.num_columns)}
    print(f"{fname}: rows={pf.metadata.num_rows} codecs={codecs} geo={'geo' in (pf.schema_arrow.metadata or {}) or b'geo' in pf.schema_arrow.metadata}")
