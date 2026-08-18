# oxigis-desktop

OxiGIS native desktop shell: a single `oxigis` binary for Linux, macOS and Windows.

**Status:** Alpha

Hosts the shared `oxigis-ui` panels and `oxigis-render` map view inside a
native window, and supplies everything that view needs from the local
machine but cannot provide itself — network and file I/O, and font
discovery.

## Features

- **Native window** — `eframe`/egui in a winit window, with the map drawn
  through wgpu.
- **Blocking HTTP tile & range transport on worker pools** — `ureq` (Pure
  Rust TLS via `rustls-graviola`, with no `ring` and no system OpenSSL)
  drives XYZ/vector tile fetches and COG/PMTiles/MBTiles range reads, so a
  request never blocks the render thread. The workers pull from **one shared
  queue** rather than each owning a private one filled round-robin: a
  per-worker queue head-of-line-blocks every job that landed behind one slow
  or unreachable host while its siblings sit idle. A `429`/`503` carrying a
  `Retry-After` pauses that origin — and only that origin — for the delay it
  asks for, shared by every worker.
- **N-layer stack rendering with a health watch** — every tiled layer in the
  project gets its own source, and every source in this shell is built
  *before* its bytes are read (a COG's header, an archive's directory and a
  tile's first fetch all land on a worker pool), so a successful
  construction says nothing about whether the layer will ever draw. The
  shell polls each source's own refusal state (`provider_watch`) and reports
  it, instead of leaving the user with a layer in the panel, a basemap-only
  map, and no message.
- **Native file dialogs** — project Open / Save / Save As, tile-archive
  Open, and PDF export use `rfd` on Windows and macOS, where it resolves to
  pure-Rust bindings (raw-dylib `windows-sys`; the `objc2` family) with no
  `cc`, no `-sys` crate and no prebuilt import library. Linux is
  deliberately off that edge — rfd's GTK backend would pull `gtk-sys`/
  `glib-sys` into a Pure Rust graph and its portal backend an async runtime
  — so the shell asks there with its own in-app `PathPrompt` window, which
  shows the absolute path a name resolves to and refuses the obvious
  mistakes.
- **Project files that cannot be half-written** — `*.oxigis.json` is written
  to a temp file in the *same* directory, flushed, and only then renamed
  over the destination, so a full disk, a permission error or a crash
  mid-write leaves the previous save byte-for-byte intact. Reads are
  bounded, because a project may carry inline GeoJSON.
- **Session persistence** — window geometry, the recent-project list and the
  directory the file dialogs open in are remembered between launches, in a
  small line-oriented file this crate writes itself rather than by turning
  on `eframe`'s `persistence` feature (which would drag `ron` and `serde`
  into the shipped binary for four values).
- **Command line** — `oxigis [OPTIONS] [PATH]...`: `-h`/`--help`,
  `-V`/`--version` (which names this binary and the crates it is assembled
  from), `--log-file`, and any number of data or project paths to open at
  startup. Arguments are handled as `OsString` throughout, so a path that is
  not valid UTF-8 still opens.
- **Local file range reads** — a local `.pmtiles`/`.mbtiles` archive is
  read with `seek`+`read` on its own worker pool instead of loading whole,
  so a 137 GB archive still opens with a 16 KiB read.
- **Background CJK font discovery** — scans the OS font directories on a
  background thread for Japanese/Korean/Simplified/Traditional Chinese
  label fallback faces (plus a bold chain), streaming each face in as it's
  read so startup never blocks on it. No single platform face covers all
  four scripts — Meiryo has no Hangul, Malgun Gothic no kana — so the scan
  keeps one candidate per script rather than one overall, and classifies by
  what a face actually is: macOS's `STHeiti` is *Heiti TC*, Traditional
  rather than Simplified, and a variable face whose `wght` default is a
  hairline master (NotoSansJP-VF on stock Windows 11) ranks below the static
  candidates, because neither the label rasteriser nor the default print
  path can select a heavier instance.
- **GeoParquet input** — enables `oxigis-ui`'s optional `geoparquet`
  feature; this is the one crate in the workspace where it's turned on.
- **PDF map export** — composes the current view to a PDF on a background
  thread with embedded Latin + CJK fonts, building one provider per entry of
  the N-layer tile stack (`export_stack`) rather than reading only the three
  legacy single slots, so a project holding an orthophoto under a hillshade
  under a cadastral tileset prints the map it is showing. Windows and macOS
  show a native save dialog; Linux asks with the in-app path prompt.

## Build

```bash
cargo build -p oxigis-desktop --release  # binary: oxigis
```

## Tests

151 tests passing.

Part of [OxiGIS](https://github.com/cool-japan/oxigis) — Pure Rust full-stack GIS.
See the workspace README for the crate matrix and build instructions.

© 2026 COOLJAPAN OU (Team Kitasan) · Apache-2.0
