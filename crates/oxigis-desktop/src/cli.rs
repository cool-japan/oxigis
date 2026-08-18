//! The command line and the log: everything this shell decides before the
//! window exists.
//!
//! Split out of `main.rs` for size (COOLJAPAN: files stay under 2000 lines);
//! the two belong together because `--log-file` is parsed here and consumed
//! by [`init_logging`] in the same breath.

use tracing_subscriber::filter::{LevelFilter, Targets};
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

/// The `--help` text, and the only place the accepted arguments are spelled.
pub const USAGE: &str = "\
OxiGIS — desktop GIS

Usage: oxigis [OPTIONS] [PATH]...

Arguments:
  [PATH]...  Files to open: .geojson/.json, .shp, .gpkg, .parquet/.geoparquet,
             a .geolibre.json project, or one .pmtiles/.mbtiles tile archive

Options:
  -h, --help             Print this help and exit
  -V, --version          Print version information and exit
      --log-file <PATH>  Also write the log to PATH (a GUI launch has no
                         console to read the log from)
      --                 Treat every later argument as a path

Environment:
  RUST_LOG  Log filter: `info` (default), `debug`, `oxigis_desktop=debug`,
            `oxigis_ui=warn,oxigis_desktop=trace`
";

/// What the command line asked this process to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Startup {
    /// Print [`USAGE`] and exit successfully.
    Help,
    /// Print [`version_line`] and exit successfully.
    Version,
    /// Open the window, with these paths queued and the log optionally teed.
    Run {
        /// Positional arguments, in the order they were given.
        paths: Vec<std::path::PathBuf>,
        /// Where `--log-file` asked for a second copy of the log.
        log_file: Option<std::path::PathBuf>,
    },
    /// An argument this build does not understand; the string is the
    /// complaint, printed with [`USAGE`] before a non-zero exit.
    Refused(String),
}

/// Parses the arguments *after* the program name.
///
/// [`std::ffi::OsString`] throughout, never [`String`]: `std::env::args`
/// panics on a non-UTF-8 argument, and a file name is exactly the argument a
/// user is most likely to hand over in some other encoding. An argument that
/// is not valid UTF-8 therefore cannot be a flag (no flag here is spelled
/// outside ASCII) and is taken as a path, unconverted, so it still names the
/// file the shell meant.
pub fn parse_args<I>(args: I) -> Startup
where
    I: IntoIterator<Item = std::ffi::OsString>,
{
    let mut paths = Vec::new();
    let mut log_file = None;
    let mut literal = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let text = match arg.to_str() {
            Some(text) if !literal => text,
            _ => {
                paths.push(std::path::PathBuf::from(arg));
                continue;
            }
        };
        match text {
            "--" => literal = true,
            "-h" | "--help" => return Startup::Help,
            "-V" | "--version" => return Startup::Version,
            "--log-file" => match args.next() {
                Some(value) => log_file = Some(std::path::PathBuf::from(value)),
                None => return Startup::Refused("--log-file needs a path".to_owned()),
            },
            _ if text.starts_with("--log-file=") => {
                log_file = Some(std::path::PathBuf::from(&text["--log-file=".len()..]));
            }
            // A single `-` is a path (some shells hand one over); anything
            // else starting with `-` is a flag, and an unknown flag must not
            // be silently opened as a file.
            _ if text.starts_with('-') && text != "-" => {
                return Startup::Refused(format!("unrecognised option \u{201c}{text}\u{201d}"));
            }
            // Lossless: this branch is reached only for valid UTF-8.
            _ => paths.push(std::path::PathBuf::from(text)),
        }
    }
    Startup::Run { paths, log_file }
}

/// The `--version` line: this binary and the three crates it is assembled
/// from, since a bug report needs all four.
pub fn version_line() -> String {
    format!(
        "oxigis {} (core {}, render {}, ui {})",
        env!("CARGO_PKG_VERSION"),
        oxigis_core::VERSION,
        oxigis_render::VERSION,
        oxigis_ui::VERSION,
    )
}

/// Builds the log filter from a `RUST_LOG` spec, defaulting to `info`.
///
/// [`Targets`] rather than `EnvFilter`: the workspace's `tracing-subscriber`
/// carries default features only, so `env-filter` (and the `regex-automata`
/// graph behind it) is not in the build and `fmt::init`'s documented
/// `RUST_LOG` handling does not apply. What this understands is `EnvFilter`'s
/// target/level subset — `debug`, `oxigis_desktop=debug`,
/// `oxigis_ui=warn,oxigis_desktop=trace` — without span-field directives.
///
/// Returns the filter plus, when the spec was unusable, the complaint to log
/// once a subscriber exists: there is nowhere to report it before that.
fn log_filter(spec: Option<&str>) -> (Targets, Option<String>) {
    let default = Targets::new().with_default(LevelFilter::INFO);
    let Some(spec) = spec.map(str::trim).filter(|spec| !spec.is_empty()) else {
        return (default, None);
    };
    match spec.parse::<Targets>() {
        Ok(targets) => (targets, None),
        Err(error) => (
            default,
            Some(format!(
                "RUST_LOG=\u{201c}{spec}\u{201d} is not a usable filter ({error}); \
                 logging at info",
            )),
        ),
    }
}

/// Installs the process-wide subscriber: stdout always, `log_file` as well
/// when one was asked for.
///
/// A GUI launch (Finder, a desktop launcher, a double-clicked file) has no
/// console, so stdout alone means a user reporting "my labels are boxes" has
/// no way to produce the `debug!` lines that would explain it. Nothing here
/// fails the process: a log that cannot be set up is reported and the window
/// still opens.
pub fn init_logging(log_file: Option<&std::path::Path>) {
    let (filter, complaint) = log_filter(std::env::var("RUST_LOG").ok().as_deref());
    let (file, file_error) = match log_file {
        Some(path) => match std::fs::File::create(path) {
            Ok(file) => (Some(std::sync::Arc::new(file)), None),
            Err(error) => (
                None,
                Some(format!(
                    "could not write the log to {}: {error}",
                    path.display()
                )),
            ),
        },
        None => (None, None),
    };
    // `Arc<File>` is a `MakeWriter` because `&File` writes; no rotation, so a
    // named log is one file per launch, truncated at start.
    let file_layer = file.map(|file| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(file)
    });
    // Layered, not `fmt::init()`: the `fmt` builder's own default max level is
    // INFO, which would filter DEBUG out underneath `filter` whatever the spec
    // said. A registry with the filter as a layer has no such floor.
    let installed = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(file_layer)
        .with(filter)
        .try_init();
    if let Err(error) = installed {
        eprintln!("oxigis: the log could not be initialised: {error}");
        return;
    }
    if let Some(complaint) = complaint {
        tracing::warn!("OxiGIS desktop: {complaint}");
    }
    if let Some(error) = file_error {
        tracing::warn!("OxiGIS desktop: {error}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an argument list the way an OS hands one over.
    fn args(list: &[&str]) -> Vec<std::ffi::OsString> {
        list.iter().map(std::ffi::OsString::from).collect()
    }

    /// The paths of a `Run`, or a failure naming what came back instead.
    fn run_paths(startup: &Startup) -> &[std::path::PathBuf] {
        match startup {
            Startup::Run { paths, .. } => paths,
            other => panic!("expected a Run, got {other:?}"),
        }
    }

    #[test]
    fn a_bare_launch_runs_with_nothing_queued() {
        assert_eq!(
            parse_args(args(&[])),
            Startup::Run {
                paths: Vec::new(),
                log_file: None,
            }
        );
    }

    #[test]
    fn help_and_version_are_recognised_in_both_spellings() {
        assert_eq!(parse_args(args(&["--help"])), Startup::Help);
        assert_eq!(parse_args(args(&["-h"])), Startup::Help);
        assert_eq!(parse_args(args(&["--version"])), Startup::Version);
        assert_eq!(parse_args(args(&["-V"])), Startup::Version);
        // A flag wins over the paths beside it: nothing is opened.
        assert_eq!(parse_args(args(&["city.gpkg", "--help"])), Startup::Help);
    }

    #[test]
    fn positional_paths_keep_their_order() {
        let startup = parse_args(args(&["a.geojson", "b.gpkg", "c.pmtiles"]));
        assert_eq!(
            run_paths(&startup),
            [
                std::path::PathBuf::from("a.geojson"),
                std::path::PathBuf::from("b.gpkg"),
                std::path::PathBuf::from("c.pmtiles"),
            ]
        );
    }

    #[test]
    fn an_unknown_option_is_refused_rather_than_opened_as_a_file() {
        match parse_args(args(&["--zoom=4"])) {
            Startup::Refused(complaint) => assert!(complaint.contains("--zoom=4"), "{complaint}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
        // A lone `-` is a path, not a flag.
        assert_eq!(
            run_paths(&parse_args(args(&["-"]))),
            [std::path::PathBuf::from("-")]
        );
    }

    #[test]
    fn a_double_dash_ends_the_flags() {
        let startup = parse_args(args(&["--", "--help", "-V"]));
        assert_eq!(
            run_paths(&startup),
            [
                std::path::PathBuf::from("--help"),
                std::path::PathBuf::from("-V"),
            ]
        );
    }

    #[test]
    fn the_log_file_is_accepted_in_both_forms_and_needs_a_value() {
        let expected = Some(std::path::PathBuf::from("run.log"));
        assert_eq!(
            parse_args(args(&["--log-file", "run.log"])),
            Startup::Run {
                paths: Vec::new(),
                log_file: expected.clone(),
            }
        );
        assert_eq!(
            parse_args(args(&["--log-file=run.log"])),
            Startup::Run {
                paths: Vec::new(),
                log_file: expected,
            }
        );
        match parse_args(args(&["--log-file"])) {
            Startup::Refused(complaint) => assert!(complaint.contains("--log-file"), "{complaint}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// A file name the OS accepts but UTF-8 does not must still open the file
    /// it names — `std::env::args` would have panicked on it.
    #[test]
    #[cfg(unix)]
    fn a_non_utf8_argument_survives_as_a_path() {
        use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};
        let raw = vec![b'c', 0xff, 0xfe, b'.', b'g', b'p', b'k', b'g'];
        let startup = parse_args(vec![std::ffi::OsString::from_vec(raw.clone())]);
        let paths = run_paths(&startup);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].as_os_str().as_bytes(), raw.as_slice());
    }

    #[test]
    fn no_rust_log_leaves_every_target_at_info() {
        let (filter, complaint) = log_filter(None);
        assert!(complaint.is_none());
        assert!(filter.would_enable("oxigis_desktop", &tracing::Level::INFO));
        assert!(!filter.would_enable("oxigis_desktop", &tracing::Level::DEBUG));
        // Whitespace is not a filter either.
        assert_eq!(log_filter(Some("  ")).0, filter);
    }

    /// The whole point of the finding: `RUST_LOG=oxigis_desktop=debug` must
    /// reach this crate's `debug!` calls, which `fmt::init()` never did.
    #[test]
    fn a_per_target_spec_opens_that_target_only() {
        let (filter, complaint) = log_filter(Some("oxigis_desktop=debug"));
        assert!(complaint.is_none());
        assert!(filter.would_enable("oxigis_desktop", &tracing::Level::DEBUG));
        assert!(!filter.would_enable("oxigis_ui", &tracing::Level::DEBUG));
        let (bare, _) = log_filter(Some("debug"));
        assert!(bare.would_enable("oxigis_ui", &tracing::Level::DEBUG));
    }

    #[test]
    fn an_unusable_spec_falls_back_to_info_with_a_complaint() {
        let (filter, complaint) = log_filter(Some("oxigis_desktop=verbose"));
        let complaint = complaint.expect("the spec must be reported");
        assert!(complaint.contains("oxigis_desktop=verbose"), "{complaint}");
        assert!(filter.would_enable("oxigis_desktop", &tracing::Level::INFO));
        assert!(!filter.would_enable("oxigis_desktop", &tracing::Level::DEBUG));
    }
}
