#!/usr/bin/env bash
# OxiGIS web shell — build the wasm bundle and serve it locally.
#
#   ./crates/oxigis-web/serve.sh            # dev profile, port 8080
#   ./crates/oxigis-web/serve.sh 9000       # dev profile, port 9000
#   ./crates/oxigis-web/serve.sh test       # run the shell's tests, no browser
#   OXIGIS_WASM_PROFILE=wasm-release ./crates/oxigis-web/serve.sh
#
# Serving on localhost is deliberate: WebGPU is only exposed in a secure
# context, and `http://localhost` counts as one while a LAN IP over plain HTTP
# does not (that would silently drop the app to the WebGL2 fallback).
set -euo pipefail

crate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${crate_dir}/../.." && pwd)"

# `serve.sh test` — the two commands that actually enforce this crate, neither
# of which needs a browser, a headless driver or any CI.
#
#   1. Host `cargo nextest run`. The verification RULES — `Content-Range`
#      parsing, the answer-matches-the-question check and the validator pins —
#      live in `src/range_rules.rs`, which is deliberately NOT `cfg`-gated to
#      wasm32 precisely so this works. Their tests are ordinary `#[test]` fns.
#   2. A wasm32 `cargo check`. Everything else in the crate is `fetch()`,
#      `web-sys` and `wgpu` glue that only exists on wasm32, so a host build
#      compiles almost none of it; this is what type-checks the real shell.
#
# What is deliberately NOT here: `wasm-pack test --node` / `--headless
# --chrome`. Those run the `wasm-bindgen-test` harness, and this crate declares
# no `wasm-bindgen-test` dependency and has no `#[wasm_bindgen_test]` function,
# so they would compile the crate and execute ZERO tests — a green run that
# proves nothing. Adding that harness is the right move only for something that
# genuinely needs a browser (the `fetch()` glue itself), not for the rules.
if [[ "${1:-}" == "test" ]]; then
    echo "==> cargo nextest run -p oxigis-web  (range_rules, on the host)"
    (cd -- "${repo_root}" && cargo nextest run -p oxigis-web --all-features)
    echo
    echo "==> cargo check -p oxigis-web --target wasm32-unknown-unknown"
    echo "    (rustup target add wasm32-unknown-unknown, once, if this fails)"
    (cd -- "${repo_root}" && cargo check -p oxigis-web --target wasm32-unknown-unknown)
    exit 0
fi

port="${1:-8080}"
# `dev` keeps the build fast; `wasm-release` is the size-optimised workspace
# profile (opt-level=s, fat LTO) used for anything shipped.
profile="${OXIGIS_WASM_PROFILE:-dev}"

for tool in wasm-pack python3; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "serve.sh: \`${tool}\` not found in PATH." >&2
        case "${tool}" in
            wasm-pack) echo "  install it with: cargo install wasm-pack" >&2 ;;
            python3) echo "  install python3, or serve ${crate_dir} with any static file server" >&2 ;;
        esac
        exit 1
    fi
done

echo "==> wasm-pack build (profile: ${profile})"
case "${profile}" in
    dev) profile_flag="--dev" ;;
    release) profile_flag="--release" ;;
    profiling) profile_flag="--profiling" ;;
    *) profile_flag="--profile ${profile}" ;;
esac

# Run from the repo root so .cargo/config.toml (getrandom wasm_js backend) and
# the workspace lockfile apply. Output lands in crates/oxigis-web/pkg/, which is
# exactly what index.html imports.
(
    cd -- "${repo_root}"
    # shellcheck disable=SC2086
    wasm-pack build crates/oxigis-web --target web ${profile_flag}
)

echo
echo "==> serving ${crate_dir} at http://localhost:${port}/"
echo "    (Ctrl-C to stop; reload the page after a rebuild)"
python3 -m http.server "${port}" --bind 127.0.0.1 --directory "${crate_dir}"
