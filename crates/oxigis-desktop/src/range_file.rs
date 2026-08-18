//! Native local-file range transport: `seek` + `read` on a worker pool.
//!
//! The platform half of `oxigis-ui`'s [`oxigis_ui::RangeTransport`] seam for a
//! file on disk, and the twin of [`crate::range_http`] in every respect that
//! matters: [`FILE_RANGE_WORKER_THREADS`] detached workers, one queued job per
//! read, the outcome reported through the same [`oxigis_ui::RangeSink`]. Only
//! the byte source differs.
//!
//! # Why this exists at all
//!
//! A local `.pmtiles` could be read whole into memory and served with
//! `oxigis_ui::MemoryRangeTransport` — that is exactly what the browser has to
//! do. On a desktop it would be a mistake: a planet-scale PMTiles archive is
//! **137 GB**, and the whole point of the format is that opening one costs a
//! 16 KiB read and a tile costs a few kilobytes more. Streaming it means a local
//! archive and a remote one go through the *identical* provider, and the only
//! difference in the whole stack is which transport was handed in.
//!
//! # Correctness of a ranged read from a file
//!
//! * A read that runs past the end of the file is **fine** and comes back
//!   short: the reader deliberately over-asks for a speculative 16 KiB header
//!   block, and a 282-byte archive is a legitimate archive.
//! * A read whose *start* is past the end is not: it means the reader computed
//!   an offset the file cannot hold, and returning zero bytes would let a
//!   corrupt directory look like an empty tile. That is a permanent failure.
//! * [`std::io::ErrorKind::NotFound`] and
//!   [`std::io::ErrorKind::PermissionDenied`] are permanent — no number of
//!   retries conjures a file or a permission — and every other IO error is
//!   transient, matching the taxonomy [`crate::range_http`] applies to HTTP.
//! * A range wider than [`MAX_FILE_RANGE_BYTES`] is refused, not clamped: a
//!   clamp-and-read would come back short and be parsed as a legitimately
//!   small file, the one place a cap would silently change data instead of
//!   refusing it — see [`read_from`].
//! * **The file changed underneath the read.** The local twin of
//!   [`crate::range_http`]'s validator pinning: [`FileValidatorPins`] pins
//!   each path's length, modified time and platform file identity on first
//!   read and refuses a later disagreement, catching a `.pmtiles`/`.mbtiles`
//!   rewritten in place — the ordinary way a local tile build is refreshed. An
//!   *atomically renamed* replacement is not caught, and does not need to be:
//!   the [`File`] this crate keeps open across reads (below) resolves to the
//!   *old* inode for ever after a rename, so every read stays one consistent,
//!   if stale, revision rather than a mixture.

use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Sender, channel};

use oxigis_render::ByteRange;
use oxigis_ui::{RangeJob, RangeSink, RangeTransport, TileError};

/// Number of blocking file-reader threads.
///
/// Was two, "like [`crate::range_http::RANGE_WORKER_THREADS`]" — stale even
/// when it was written: that constant has been six since tiles v1.4, sized so
/// a cold paged-MBTiles viewport's ~56 small reads (the measurement is in
/// `range_http`'s module docs) do not serialise two at a time. Four, not six:
/// a local read has no per-request network latency to hide behind
/// concurrency, so more workers buys less here than it does for HTTP, but a
/// cold open still fans out across more than a single pair of threads.
pub const FILE_RANGE_WORKER_THREADS: usize = 4;

/// Hard limit on a single read from a local archive, in bytes.
///
/// The reader's own caps are smaller (16 MiB for a tile body, 4 MiB for a leaf
/// directory), so this only bounds what a *corrupt* length field can turn into
/// an allocation before the parser ever sees it.
pub const MAX_FILE_RANGE_BYTES: u64 = 32 * 1024 * 1024;

/// One queued file read.
struct Job {
    /// Path to read from.
    path: String,
    /// Range to read.
    range: ByteRange,
    /// What the provider will do with the bytes.
    job: RangeJob,
    /// Where the outcome is reported.
    sink: RangeSink,
}

/// Blocking local-file [`RangeTransport`] for native builds.
///
/// Owns [`FILE_RANGE_WORKER_THREADS`] detached worker threads; dropping it
/// closes their queues, which ends each worker's loop after the read in flight.
pub struct FileRangeTransport {
    /// One queue per worker.
    queues: Vec<Sender<Job>>,
    /// Round-robin cursor over `queues`.
    next: AtomicUsize,
}

impl FileRangeTransport {
    /// Starts the worker pool.
    ///
    /// No file is opened here: the path travels with each job, so one transport
    /// serves any number of archives and a file that disappears between reads
    /// is reported per read rather than making construction fail.
    ///
    /// # Errors
    ///
    /// Returns the [`std::io::Error`] from [`std::thread::Builder::spawn`] if
    /// the OS refuses to start a worker thread; the caller should leave the
    /// archive layer unattached rather than treating this as fatal.
    pub fn new() -> Result<Self, std::io::Error> {
        // Shared by every worker, so a path opened first by worker 2 is
        // checked against the length/mtime/inode worker 0 pinned. Scoped to
        // the transport, which is built per layer: "remove and re-add the
        // layer" is therefore literally what clears a drift refusal, exactly
        // as for the HTTP twin's `ValidatorPins`.
        let pins = Arc::new(FileValidatorPins::default());
        let mut queues = Vec::with_capacity(FILE_RANGE_WORKER_THREADS);
        for index in 0..FILE_RANGE_WORKER_THREADS {
            let (tx, rx) = channel::<Job>();
            let pins = Arc::clone(&pins);
            std::thread::Builder::new()
                .name(format!("oxigis-archive-{index}"))
                .spawn(move || {
                    // One open file kept per worker, reopened whenever the path
                    // changes: an interactive pan is hundreds of reads of the
                    // same archive, and reopening per read would be hundreds of
                    // directory lookups for nothing. Revalidated on every read
                    // regardless (see `FileValidatorPins::check`), so keeping
                    // the handle open this long never risks missing a rewrite.
                    let mut open: Option<(String, File)> = None;
                    for queued in rx {
                        let result = read_range(&mut open, &pins, &queued.path, queued.range);
                        queued.sink.deliver(queued.job, result);
                    }
                })?;
            queues.push(tx);
        }
        Ok(Self {
            queues,
            next: AtomicUsize::new(0),
        })
    }
}

impl core::fmt::Debug for FileRangeTransport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FileRangeTransport")
            .field("workers", &self.queues.len())
            .finish()
    }
}

impl RangeTransport for FileRangeTransport {
    fn request_range(&self, url: String, range: ByteRange, job: RangeJob, sink: RangeSink) {
        if self.queues.is_empty() {
            sink.deliver(job, Err(TileError::permanent("no archive worker threads")));
            return;
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.queues.len();
        let Some(queue) = self.queues.get(index) else {
            sink.deliver(
                job,
                Err(TileError::permanent("archive worker queue vanished")),
            );
            return;
        };
        let queued = Job {
            path: url,
            range,
            job,
            sink: sink.clone(),
        };
        if let Err(error) = queue.send(queued) {
            sink.deliver(
                job,
                Err(TileError::permanent(format!(
                    "archive worker is gone: {error}"
                ))),
            );
        }
    }
}

/// What the user is told when a local archive changed underneath an open
/// layer.
///
/// `crate::range_http::DRIFT_ADVICE` says the same thing for a URL; this is
/// its own copy because that one bakes in "on the server", which would be
/// wrong here.
const DRIFT_ADVICE: &str = "the archive changed on disk; remove and re-add the layer";

/// What a path's metadata pinned at first read: enough to notice a
/// same-inode rewrite-in-place, the ordinary way a local tile build is
/// refreshed. See the module docs for why an atomic rename is not, and does
/// not need to be, caught by this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct FileValidator {
    /// File length in bytes. Always available once [`std::fs::Metadata`] is
    /// in hand, unlike the two fields below.
    len: u64,
    /// Last-modified time, when the platform and filesystem report one.
    modified: Option<std::time::SystemTime>,
    /// Unix inode, when the platform exposes one — see [`platform_id_of`].
    /// Windows has no stable equivalent (`MetadataExt::file_index` exists but
    /// sits behind the still-unstable `windows_by_handle` feature — tracking
    /// issue #63010 — and reaching it without one more `-sys` dependency this
    /// workspace does not otherwise need is not worth the trade for a field
    /// that is a defense-in-depth extra on top of `len`/`modified`, not the
    /// only signal), so this is always [`None`] there.
    platform_id: Option<u64>,
}

impl FileValidator {
    /// Reads the fields this type pins out of already-fetched `metadata`.
    fn observe(metadata: &std::fs::Metadata) -> Self {
        Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            platform_id: platform_id_of(metadata),
        }
    }
}

/// The Unix inode [`FileValidator`] pins, so a delete-then-recreate-with-the-
/// -same-name at an unchanged length and mtime is still caught. [`None`] on
/// every other target — see the field doc on [`FileValidator::platform_id`]
/// for why Windows has no stable equivalent to fall back to here.
#[cfg(unix)]
fn platform_id_of(metadata: &std::fs::Metadata) -> Option<u64> {
    use std::os::unix::fs::MetadataExt as _;
    Some(metadata.ino())
}

/// Non-Unix fallback: `len`/`modified` alone still catch the common case (a
/// rewrite that changes the file's size, or whose filesystem tracks
/// modification time).
#[cfg(not(unix))]
fn platform_id_of(_metadata: &std::fs::Metadata) -> Option<u64> {
    None
}

/// Per-path validators, pinned at first read and checked at every later one —
/// the local twin of `crate::range_http::ValidatorPins`, bounded by the same
/// [`crate::range_http::MAX_PINNED_VALIDATORS`] for the same reason: a session
/// opens a handful of local archives, not thousands.
#[derive(Debug, Default)]
struct FileValidatorPins {
    /// `(path, validator)` in first-seen order.
    entries: std::sync::Mutex<Vec<(String, FileValidator)>>,
}

impl FileValidatorPins {
    /// Pins `observed` for `path` on the first call, and refuses a later call
    /// whose length, modified time or platform id disagrees. A field that is
    /// [`None`] on either side is not compared — a host filesystem that
    /// cannot report a modified time has not thereby proven a rewrite,
    /// mirroring how `crate::range_http::ValidatorPins` treats an
    /// unoffered `ETag`.
    ///
    /// Checked on **every** read, not only when a worker (re)opens the file:
    /// a worker that already holds the file open never calls [`File::open`]
    /// again for the rest of the session (see [`FileRangeTransport::new`]),
    /// so an open-only check would fire during a narrow window at startup and
    /// then never again — missing exactly the case this exists for, a
    /// rewrite happening to an archive that has already been open a while.
    /// [`std::fs::Metadata`] on an already-open [`File`] is one syscall with
    /// no path lookup, the same cost class as the `seek`/`read` a read
    /// already does.
    ///
    /// # Errors
    ///
    /// A permanent refusal naming [`DRIFT_ADVICE`] when a pinned field
    /// disagrees with `observed`.
    fn check(&self, path: &str, observed: FileValidator) -> Result<(), TileError> {
        let mut entries = self.lock();
        if let Some((_, pinned)) = entries.iter_mut().find(|(held, _)| held == path) {
            if pinned.len != observed.len {
                return Err(TileError::permanent(format!(
                    "{path}: {DRIFT_ADVICE} (its length went from {} to {} bytes)",
                    pinned.len, observed.len
                )));
            }
            if let (Some(was), Some(now)) = (pinned.modified, observed.modified)
                && was != now
            {
                return Err(TileError::permanent(format!(
                    "{path}: {DRIFT_ADVICE} (its modified time changed)"
                )));
            }
            if let (Some(was), Some(now)) = (pinned.platform_id, observed.platform_id)
                && was != now
            {
                return Err(TileError::permanent(format!(
                    "{path}: {DRIFT_ADVICE} (it is no longer the same file on disk)"
                )));
            }
            if pinned.modified.is_none() {
                pinned.modified = observed.modified;
            }
            if pinned.platform_id.is_none() {
                pinned.platform_id = observed.platform_id;
            }
            return Ok(());
        }
        while entries.len() >= crate::range_http::MAX_PINNED_VALIDATORS && !entries.is_empty() {
            entries.remove(0);
        }
        entries.push((path.to_owned(), observed));
        Ok(())
    }

    /// The entry list, recovering the contents if a worker panicked holding
    /// it — a poisoned mutex here means some *other* read panicked; the pins
    /// themselves are plain data and are still exactly as valid as they were.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(String, FileValidator)>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Reads one range out of `path`, reusing `open` when it already holds it,
/// and refusing the read if `pins` finds `path` has changed since it was
/// first observed (see [`FileValidatorPins::check`]).
fn read_range(
    open: &mut Option<(String, File)>,
    pins: &FileValidatorPins,
    path: &str,
    range: ByteRange,
) -> Result<Vec<u8>, TileError> {
    let matches = open.as_ref().is_some_and(|(held, _)| held.as_str() == path);
    if !matches {
        match File::open(path) {
            Ok(file) => *open = Some((path.to_owned(), file)),
            Err(error) => {
                *open = None;
                return Err(classify_io(&error, path));
            }
        }
    }
    let Some((_, file)) = open.as_mut() else {
        return Err(TileError::transient(format!("{path} could not be opened")));
    };
    let metadata = match file.metadata() {
        Ok(metadata) => metadata,
        Err(error) => {
            *open = None;
            return Err(classify_io(&error, path));
        }
    };
    if let Err(error) = pins.check(path, FileValidator::observe(&metadata)) {
        // A permanent, session-scoped refusal — the same file will keep
        // failing this check, so there is nothing to gain from holding the
        // handle open, and every reason to let the next request reopen (and
        // re-observe) it fresh.
        *open = None;
        return Err(error);
    }
    read_from(file, path, range).inspect_err(|error| {
        // A failed read may have left the handle in an unknown position, and a
        // vanished file must not be read from a stale descriptor for ever.
        if !error.retryable() {
            *open = None;
        }
    })
}

/// Performs one seek-and-read.
///
/// A range wider than [`MAX_FILE_RANGE_BYTES`] is refused rather than
/// clamped: the reader's own caps (16 MiB for an archive tile, 4 MiB for a
/// leaf directory) mean this is not reachable from a well-formed archive
/// today, so overrunning this much more generous bound at all means the
/// caller computed a range from corrupt data. Clamping it would come back
/// short and be silently parsed as a legitimately small file — the HTTP twin
/// refuses the same overrun by name via `.limit(MAX_RANGE_BYTES)` rather than
/// truncating.
fn read_from(file: &mut File, path: &str, range: ByteRange) -> Result<Vec<u8>, TileError> {
    let length = range.end.saturating_sub(range.start);
    if length > MAX_FILE_RANGE_BYTES {
        return Err(TileError::permanent(format!(
            "a {length}-byte read of {path} exceeds the {MAX_FILE_RANGE_BYTES}-byte cap"
        )));
    }
    let Ok(capacity) = usize::try_from(length) else {
        return Err(TileError::permanent(format!(
            "a {length}-byte read of {path} does not fit this machine's address space"
        )));
    };
    if let Err(error) = file.seek(SeekFrom::Start(range.start)) {
        return Err(classify_io(&error, path));
    }
    let mut buffer = vec![0u8; capacity];
    let mut filled = 0usize;
    while filled < capacity {
        let Some(rest) = buffer.get_mut(filled..) else {
            break;
        };
        match file.read(rest) {
            // A short read at the end of the file is normal, not a failure:
            // the speculative 16 KiB prefetch over-asks by design.
            Ok(0) => break,
            Ok(read) => filled = filled.saturating_add(read),
            Err(ref error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(classify_io(&error, path)),
        }
    }
    if filled == 0 {
        return Err(TileError::permanent(format!(
            "byte {} is past the end of {path}",
            range.start
        )));
    }
    buffer.truncate(filled);
    Ok(buffer)
}

/// Classifies a filesystem error for the shared retry policy.
///
/// A missing file and a refused permission are properties of the system, not of
/// this attempt; everything else (a busy device, a network share hiccup) may
/// well work next frame.
fn classify_io(error: &std::io::Error, path: &str) -> TileError {
    let message = format!("{path}: {error}");
    match error.kind() {
        std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied => {
            TileError::permanent(message)
        }
        _ => TileError::transient(message),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DRIFT_ADVICE, FILE_RANGE_WORKER_THREADS, FileRangeTransport, FileValidator,
        FileValidatorPins, MAX_FILE_RANGE_BYTES, read_range,
    };
    use oxigis_render::ByteRange;

    /// Writes `bytes` to a uniquely named file in the temp directory.
    fn temp_archive(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let path = std::env::temp_dir().join(format!("oxigis-{name}-{stamp}.bin"));
        std::fs::write(&path, bytes).expect("the fixture must be writable");
        path
    }

    fn range(start: u64, end: u64) -> ByteRange {
        ByteRange::new(start, end).expect("a non-empty range")
    }

    #[test]
    fn the_transport_starts_a_worker_pool() {
        let transport = FileRangeTransport::new().expect("worker threads must start");
        assert!(format!("{transport:?}").contains(&FILE_RANGE_WORKER_THREADS.to_string()));
    }

    #[test]
    fn a_range_inside_the_file_reads_exactly_those_bytes() {
        let path = temp_archive("inside", &[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut open = None;
        let pins = FileValidatorPins::default();
        let bytes = read_range(&mut open, &pins, &path.display().to_string(), range(2, 5))
            .expect("an in-bounds read");
        assert_eq!(bytes, vec![2, 3, 4]);
        // The handle is kept for the next read of the same path.
        assert!(open.is_some());
        let _removed = std::fs::remove_file(&path);
    }

    #[test]
    fn a_range_running_past_the_end_comes_back_short_rather_than_failing() {
        let path = temp_archive("short", &[9, 8, 7]);
        let mut open = None;
        let pins = FileValidatorPins::default();
        let bytes = read_range(
            &mut open,
            &pins,
            &path.display().to_string(),
            range(1, 16_384),
        )
        .expect("a short read at EOF is legitimate");
        assert_eq!(bytes, vec![8, 7]);
        let _removed = std::fs::remove_file(&path);
    }

    #[test]
    fn a_range_wider_than_the_cap_is_refused_rather_than_truncated() {
        // A regression pin: a naive `.min(MAX_FILE_RANGE_BYTES)` clamp would
        // read the shrunk length as a legitimate short file instead of
        // refusing. The file itself can stay tiny — the refusal must fire on
        // the requested range, before any read is attempted.
        let path = temp_archive("oversize", &[0u8; 8]);
        let mut open = None;
        let pins = FileValidatorPins::default();
        let error = read_range(
            &mut open,
            &pins,
            &path.display().to_string(),
            range(0, MAX_FILE_RANGE_BYTES + 1),
        )
        .expect_err("a range past the cap must be refused, not silently shrunk");
        assert!(!error.retryable(), "{error}");
        assert!(error.message().contains("exceeds"), "{error}");
        let _removed = std::fs::remove_file(&path);
    }

    #[test]
    fn a_range_starting_past_the_end_is_a_permanent_failure() {
        let path = temp_archive("past", &[1, 2, 3]);
        let mut open = None;
        let pins = FileValidatorPins::default();
        let error = read_range(
            &mut open,
            &pins,
            &path.display().to_string(),
            range(99, 128),
        )
        .expect_err("a start past the end is not a short read");
        assert!(!error.retryable());
        assert!(error.message().contains("past the end"), "{error}");
        let _removed = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_is_permanent_not_retried_for_ever() {
        let mut open = None;
        let pins = FileValidatorPins::default();
        let error = read_range(
            &mut open,
            &pins,
            "I:\\this\\path\\does\\not\\exist\\at\\all.pmtiles",
            range(0, 16),
        )
        .expect_err("a missing file must fail");
        assert!(!error.retryable(), "{error}");
        assert!(open.is_none());
    }

    // -----------------------------------------------------------------------
    // A local archive rewritten under an open layer is refused, not silently
    // read as a mixture of old and new bytes.
    // -----------------------------------------------------------------------

    #[test]
    fn a_file_rewritten_in_place_is_refused_on_the_next_read() {
        // The scenario this finding is about: a `.pmtiles`/`.mbtiles`
        // regenerated in place at the SAME path — no rename — the ordinary
        // way a local tile build is refreshed.
        let path = temp_archive("rewritten", &[0u8; 8]);
        let path_str = path.display().to_string();
        let mut open = None;
        let pins = FileValidatorPins::default();

        let _first = read_range(&mut open, &pins, &path_str, range(0, 4)).expect("first read");

        // Rewrite the same path with a different length, in place.
        std::fs::write(&path, [1u8; 16]).expect("the fixture must be rewritable");

        let error = read_range(&mut open, &pins, &path_str, range(0, 4))
            .expect_err("a length change on an already-pinned path must be refused");
        assert!(!error.retryable(), "{error}");
        assert!(error.message().contains(DRIFT_ADVICE), "{error}");

        // And it keeps refusing — a drifted pin does not clear itself.
        let again = read_range(&mut open, &pins, &path_str, range(0, 4))
            .expect_err("the refusal must persist for the life of this transport");
        assert!(again.message().contains(DRIFT_ADVICE), "{again}");

        let _removed = std::fs::remove_file(&path);
    }

    #[test]
    fn an_unchanged_file_reads_repeatedly_without_tripping_drift() {
        let path = temp_archive("stable", &[7u8; 32]);
        let path_str = path.display().to_string();
        let mut open = None;
        let pins = FileValidatorPins::default();
        for attempt in 0..5 {
            read_range(&mut open, &pins, &path_str, range(0, 4))
                .unwrap_or_else(|error| panic!("read {attempt} of an unchanged file: {error}"));
        }
        let _removed = std::fs::remove_file(&path);
    }

    #[test]
    fn a_changed_platform_id_at_the_same_length_is_drift() {
        // What a delete-then-create-with-the-same-name produces on Unix
        // (unlike a rename, which this transport never even observes): the
        // length can coincide while the inode does not.
        let pins = FileValidatorPins::default();
        let first = FileValidator {
            len: 100,
            modified: None,
            platform_id: Some(11),
        };
        assert_eq!(pins.check("p", first), Ok(()));
        let second = FileValidator {
            platform_id: Some(22),
            ..first
        };
        let error = pins
            .check("p", second)
            .expect_err("a changed platform id is drift");
        assert!(!error.retryable(), "{error}");
        assert!(error.message().contains(DRIFT_ADVICE), "{error}");
    }

    #[test]
    fn a_field_unknown_on_one_side_does_not_manufacture_drift() {
        // Mirrors `crate::range_http::ValidatorPins`' "unoffered ETag" rule: a
        // filesystem that cannot report a modified time (or a first read that
        // for whatever reason didn't) must not turn into a permanent refusal
        // the moment it does.
        let pins = FileValidatorPins::default();
        let unknown = FileValidator {
            len: 10,
            modified: None,
            platform_id: None,
        };
        assert_eq!(pins.check("p", unknown), Ok(()));
        let now_known = FileValidator {
            modified: Some(std::time::SystemTime::now()),
            ..unknown
        };
        assert_eq!(pins.check("p", now_known), Ok(()));

        // The retained value now guards: a THIRD call with a different
        // `modified` must be caught, proving the second call's observation
        // was actually pinned rather than discarded — the same "fill the gap
        // once" rule `crate::range_http::ValidatorPins` applies to `ETag`.
        let different_time = FileValidator {
            modified: Some(std::time::SystemTime::now() + std::time::Duration::from_secs(3600)),
            ..now_known
        };
        let error = pins
            .check("p", different_time)
            .expect_err("the modified time pinned by the second call must now guard");
        assert!(error.message().contains(DRIFT_ADVICE), "{error}");
    }

    #[test]
    fn the_pin_store_is_bounded() {
        let pins = FileValidatorPins::default();
        let cap = crate::range_http::MAX_PINNED_VALIDATORS;
        for index in 0..(cap * 2) {
            let validator = FileValidator {
                len: index as u64,
                modified: None,
                platform_id: None,
            };
            assert_eq!(pins.check(&format!("path-{index}"), validator), Ok(()));
        }
        assert_eq!(pins.lock().len(), cap);
    }

    #[test]
    fn a_real_archive_streams_through_the_provider_that_reads_a_remote_one() {
        use oxigis_ui::{ArchiveTileProvider, TileProvider as _};

        // The whole point of this transport, asserted end to end: the SAME
        // provider that reads an archive over HTTP reads one off the disk, with
        // nothing swapped but the transport.
        let path = temp_archive("pmtiles", &oxigis_render::pmtiles::sample_pmtiles_raster());
        let transport = FileRangeTransport::new().expect("worker threads must start");
        let provider = ArchiveTileProvider::pmtiles(
            path.display().to_string(),
            &egui::Context::default(),
            Box::new(transport),
        )
        .expect("the provider must build");

        let tile = oxigis_render::TileId::new(0, 0, 0).expect("0/0/0");
        let mut decoded = None;
        for _ in 0..200 {
            if let Some(pixels) = provider.tile(tile) {
                decoded = Some(pixels);
                break;
            }
            if let Some(failure) = provider.failure() {
                panic!("the local archive failed to open: {failure}");
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let pixels = decoded.expect("a tile must arrive within 5 s");
        assert_eq!(pixels.width(), 2);
        assert_eq!(&pixels.rgba()[..3], &[220, 40, 40]);
        let _removed = std::fs::remove_file(&path);
    }

    /// The MBTiles twin of the test above is
    /// [`a_local_mbtiles_path_pages_all_the_way_to_a_drawn_tile`], reachable
    /// since tiles v1.5 because `oxigis-ui`'s `fixtures` feature exports one
    /// hand-built indexed archive. What is still uncovered is
    /// `map_gpu::replace_provider`, which needs a wgpu render state and cannot
    /// run headless: the provider is the honest boundary.
    #[test]
    fn a_local_mbtiles_path_pages_all_the_way_to_a_drawn_tile() {
        use oxigis_core::ArchiveFormat;
        use oxigis_ui::{ArchiveContent, ArchiveProbe, ArchiveTileProvider, TileProvider as _};

        // The bytes are hand-assembled and deterministic; only the *file name*
        // carries a clock, as a uniqueness suffix. The `.bin` suffix is
        // irrelevant — every call below names `ArchiveFormat::MbTiles`
        // explicitly and nothing here sniffs the extension.
        let path = temp_archive("mbtiles", &oxigis_ui::sample_mbtiles_raster());
        let location = path.display().to_string();

        // 1. Open path → survey, exactly what `main.rs` does on a pick. A
        //    survey-time refusal lands HERE, before any layer exists — the
        //    thing tiles v1.4 moved, and which nothing CI-visible checked on a
        //    local path until now.
        let probe = ArchiveProbe::start(
            location.clone(),
            ArchiveFormat::MbTiles,
            &egui::Context::default(),
            Box::new(FileRangeTransport::new().expect("worker threads must start")),
        );
        let mut surveyed = None;
        for _ in 0..200 {
            if let Some(answer) = probe.take() {
                surveyed = Some(answer.expect("the local archive must survey cleanly"));
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let opened = surveyed.expect("the survey must finish within 5 s");
        assert_eq!(opened.info.content, ArchiveContent::Raster);
        assert_eq!(opened.info.min_zoom, 0);
        assert_eq!(opened.info.max_zoom, 2);
        assert_eq!(opened.location, location);

        // 2. Page → pixels. The exact call `main.rs` makes for a local
        //    `.mbtiles`, with `declared_total` computed the way
        //    `archive_length` computes it. Composed WITHOUT a base provider on
        //    purpose: `ArchiveTileProvider::tile` asks its base first and
        //    unconditionally, so a base here would fetch OSM tiles from CI.
        let declared_total = std::fs::metadata(&path).map(|metadata| metadata.len()).ok();
        assert!(declared_total.is_some(), "the fixture was just written");
        let provider = ArchiveTileProvider::paged_mbtiles(
            location,
            &egui::Context::default(),
            Box::new(FileRangeTransport::new().expect("worker threads must start")),
            declared_total,
        )
        .expect("the provider must build");

        let root = oxigis_render::TileId::new(0, 0, 0).expect("0/0/0");
        let mut decoded = None;
        for _ in 0..200 {
            if let Some(pixels) = provider.tile(root) {
                decoded = Some(pixels);
                break;
            }
            if let Some(failure) = provider.failure() {
                panic!("the local archive failed to page: {failure}");
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let pixels = decoded.unwrap_or_else(|| {
            panic!(
                "0/0/0 must arrive within 5 s; stats {:?}, failure {:?}",
                provider.stats(),
                provider.failure()
            )
        });
        assert_eq!(pixels.width(), 2);
        assert_eq!(
            &pixels.rgba()[..3],
            &[10, 120, 200],
            "the MBTiles fixture's colour — a wrong archive would say so"
        );

        // 3. An XYZ address whose MBTiles row differs: the TMS flip surviving
        //    the whole composition, which a single-tile test would miss.
        let flipped = oxigis_render::TileId::new(1, 0, 1).expect("1/0/1");
        let mut second = None;
        for _ in 0..200 {
            if let Some(pixels) = provider.tile(flipped) {
                second = Some(pixels);
                break;
            }
            if let Some(failure) = provider.failure() {
                panic!("1/0/1 failed to page: {failure}");
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let second = second.unwrap_or_else(|| {
            panic!(
                "1/0/1 must arrive within 5 s; stats {:?}, failure {:?}",
                provider.stats(),
                provider.failure()
            )
        });
        assert_eq!(second.width(), 2);

        assert!(provider.is_open());
        assert_eq!(provider.stats().failed, 0);
        let _removed = std::fs::remove_file(&path);
    }
}
