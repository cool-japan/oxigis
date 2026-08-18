//! Native range transport: blocking HTTP `Range` requests on a worker pool.
//!
//! The platform half of `oxigis-ui`'s [`oxigis_ui::RangeTransport`] seam, i.e.
//! the Cloud-Optimized GeoTIFF counterpart of [`crate::tile_http`]. The shared
//! provider ([`oxigis_ui::CogTileProvider`]) owns the COG parse, the tile
//! composition and the retry policy; this module only turns
//! *(URL, byte range)* into bytes without ever blocking the render thread.
//!
//! Everything in [`crate::tile_http`]'s module documentation about why threads
//! rather than an async runtime, and about the Pure-Rust TLS stack, applies here
//! unchanged — the same [`ureq`] TLS and user-agent configuration is used, but
//! with this module's own, longer request timeout ([`RANGE_REQUEST_TIMEOUT`]):
//! a COG source tile is a multi-megabyte range read, which on a slow link
//! cannot finish inside the 15-second budget sized for kilobyte XYZ tiles.
//!
//! # Correctness of a ranged read
//!
//! A ranged `GET` has failure modes a plain `GET` does not, and every one of
//! them is caught here rather than passed off as data:
//!
//! * **The server ignored `Range`** and answered `200 OK` with the whole file.
//!   Handing those bytes back as if they were the requested range would make the
//!   parser read a tile directory out of the file header. Only `206 Partial
//!   Content` is accepted.
//! * **The response is shorter than asked for.** That is legitimate — the reader
//!   deliberately over-asks for a speculative header block — so short reads are
//!   passed through, and the parser detects a genuinely truncated file itself.
//! * **The server answered a *different* range.** A proxy or CDN may answer
//!   `206` with bytes from somewhere else entirely, and a status check alone
//!   cannot see it: the archive readers validate the offset *they* asked for,
//!   not the one the server replied with. [`verify_content_range`] compares the
//!   `Content-Range` header against both the request and the body length, and
//!   refuses a disagreement by name. An **absent** `Content-Range` on a 206 is
//!   accepted — real hosts omit it and the read is otherwise well-formed — with
//!   the file length simply recorded as unknown.
//! * **The file changed underneath the read.** PMTiles v3 carries no in-file
//!   validator at all (all 127 header bytes are accounted for: no ETag, no
//!   checksum), so a continuously-republished planet build can serve a header
//!   from one revision and a leaf directory from the next, and the mixture
//!   decodes to plausible garbage. [`ValidatorPins`] pins the first answer's
//!   `(ETag, total length)` per URL and refuses a later disagreement.
//!
//! # Validator pinning, and why only the native shell sends `If-Match`
//!
//! Once a strong `ETag` is pinned, every later request for that URL carries
//! `If-Match`, so the *server* rejects a stale read with `412` before a single
//! wrong byte is transferred (verified against both public archive hosts used by
//! the live tests). The browser transport deliberately does **not**: `If-Match`
//! is not a CORS-safelisted request header, so adding it turns a working
//! `Access-Control-Allow-Headers: Range` host into an opaque `TypeError`. There
//! the comparison is passive, against whatever `ETag` the host chose to expose —
//! see `oxigis-web`'s `range_fetch`.

use std::sync::Arc;

use oxigis_render::ByteRange;
use oxigis_ui::{RangeJob, RangeSink, RangeTransport, TileError};

use crate::tile_http::JobQueue;

/// Number of blocking HTTP worker threads for range requests.
///
/// Raised from two to six in tiles v1.4, with the measurement that forced it: a
/// **cold paged-SQLite viewport issues 56 small reads** — a b-tree descent per
/// visible tile, one page per level, plus the survey — where a COG viewport
/// issues at most sixteen large ones. At two workers and the 250–1000 ms
/// per-request latency measured against the public archive hosts, those 56 reads
/// serialise into 7–28 seconds before the first tile draws; at six they fit in
/// the two-to-nine-second band a first paint can live with. Six is also still
/// polite to a single origin: browsers allow six connections per host for
/// exactly this trade-off.
pub const RANGE_WORKER_THREADS: usize = 6;

/// How many URLs' validators are remembered at once.
///
/// A session opens a handful of archives, not thousands; the bound exists so a
/// long-lived process that has been pointed at many URLs cannot grow this map
/// without limit. The oldest entry is dropped when a new URL arrives, which at
/// worst costs one un-verified first read of an archive nothing is looking at
/// any more.
pub const MAX_PINNED_VALIDATORS: usize = 64;

/// Hard limit on a single ranged response body, in bytes.
///
/// A 512×512 16-bit DEFLATE tile is well under a megabyte, and the reader's
/// largest speculative request is 64 KiB; 32 MiB leaves generous headroom while
/// keeping a misconfigured URL a bounded mistake.
pub const MAX_RANGE_BYTES: u64 = 32 * 1024 * 1024;

/// Wall-clock budget for one ranged request, connect through last byte.
///
/// Sized for real COG payloads rather than basemap tiles: a Sentinel-2 TCI
/// source tile is a 1024² three-band DEFLATE block of a few megabytes, and a
/// map tile needs several of them, fetched over long-haul links (the open-data
/// buckets live in one AWS region). Measured against `sentinel-cogs`
/// (us-west-2) from a domestic connection, individual reads exceeded the
/// 15-second XYZ budget ([`crate::tile_http::REQUEST_TIMEOUT`]) and were burnt
/// as transient retries; sixty seconds keeps a slow link working while still
/// bounding a stuck connection.
pub const RANGE_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// HTTP status that means "here is the range you asked for".
const STATUS_PARTIAL_CONTENT: u16 = 206;

/// HTTP status that means "the `If-Match` validator no longer holds".
const STATUS_PRECONDITION_FAILED: u16 = 412;

/// HTTP status that means "that range is not inside this resource".
const STATUS_RANGE_NOT_SATISFIABLE: u16 = 416;

/// HTTP status that means "you are going too fast" — a rate limit, not a
/// refusal of this read. Shares [`crate::tile_http::HostCooldowns`] with the
/// tile pool: a `Retry-After` from one COG worker pauses the origin for all
/// six.
const STATUS_TOO_MANY_REQUESTS: u16 = 429;

/// HTTP status at and above which a failure is treated as retryable.
const FIRST_SERVER_ERROR_STATUS: u16 = 500;

/// First HTTP status that is not a success.
const FIRST_NON_SUCCESS_STATUS: u16 = 300;

/// What the user is told when the file changed underneath an open archive.
///
/// Re-adding the layer is the honest remedy: every cached directory, leaf and
/// tile in memory belongs to the old revision, and there is no way to tell which
/// of them are still valid.
///
/// [`crate::range_file`] gives its local-file twin of this failure its own
/// copy of this constant, worded for disk rather than a server, rather than
/// sharing this one: every use of this string bakes in "on the server", which
/// would be wrong there.
const DRIFT_ADVICE: &str = "the archive changed on the server; remove and re-add the layer";

/// One queued range read.
struct Job {
    /// URL to read from.
    url: String,
    /// Range to ask for.
    range: ByteRange,
    /// What the provider will do with the bytes.
    job: RangeJob,
    /// Where the outcome is reported.
    sink: RangeSink,
}

/// Blocking-HTTP [`RangeTransport`] for native builds.
///
/// Owns [`RANGE_WORKER_THREADS`] detached worker threads pulling from one
/// shared [`JobQueue`] (see [`crate::tile_http`]'s module docs for why a
/// shared queue replaced one private queue per worker); dropping the
/// transport closes the queue, which wakes every worker and ends its loop
/// once the reads already queued — including one a worker is mid-flight
/// on — are done.
pub struct HttpRangeTransport {
    /// Reads not yet claimed by a worker.
    queue: Arc<JobQueue<Job>>,
    /// How many workers were started — for [`Debug`] and the empty-pool guard
    /// in [`RangeTransport::request_range`].
    worker_count: usize,
}

impl HttpRangeTransport {
    /// Builds the HTTP agent and starts the worker pool.
    ///
    /// # Errors
    ///
    /// Returns the [`std::io::Error`] from [`std::thread::Builder::spawn`] if
    /// the OS refuses to start a worker thread; the caller should fall back to
    /// leaving the COG layer unattached rather than treating this as fatal.
    pub fn new() -> Result<Self, std::io::Error> {
        let agent = Arc::new(crate::tile_http::build_agent_with_timeout(
            RANGE_REQUEST_TIMEOUT,
            RANGE_WORKER_THREADS,
        ));
        // Shared by every worker, so a leaf read on thread 5 is checked against
        // the validator the header read on thread 0 pinned. Scoped to the
        // transport, which is built per layer: "remove and re-add the layer" is
        // therefore literally what clears a drift refusal.
        let pins = Arc::new(ValidatorPins::default());
        // Shared the same way, so a 429 seen by one worker pauses the origin
        // for all six — see `crate::tile_http::HostCooldowns`.
        let cooldowns = Arc::new(crate::tile_http::HostCooldowns::default());
        let queue = Arc::new(JobQueue::<Job>::new());
        for index in 0..RANGE_WORKER_THREADS {
            let agent = Arc::clone(&agent);
            let pins = Arc::clone(&pins);
            let cooldowns = Arc::clone(&cooldowns);
            let queue = Arc::clone(&queue);
            std::thread::Builder::new()
                .name(format!("oxigis-cog-{index}"))
                .spawn(move || {
                    while let Some(queued) = queue.pop() {
                        let result =
                            fetch_range(&agent, &pins, &cooldowns, &queued.url, queued.range);
                        queued.sink.deliver(queued.job, result);
                    }
                })?;
        }
        Ok(Self {
            queue,
            worker_count: RANGE_WORKER_THREADS,
        })
    }
}

impl Drop for HttpRangeTransport {
    fn drop(&mut self) {
        // Non-blocking — see `HttpTileTransport`'s identical `Drop`.
        self.queue.close();
    }
}

impl core::fmt::Debug for HttpRangeTransport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HttpRangeTransport")
            .field("workers", &self.worker_count)
            .finish()
    }
}

impl RangeTransport for HttpRangeTransport {
    fn request_range(&self, url: String, range: ByteRange, job: RangeJob, sink: RangeSink) {
        if self.worker_count == 0 {
            sink.deliver(job, Err(TileError::permanent("no COG worker threads")));
            return;
        }
        self.queue.push(Job {
            url,
            range,
            job,
            sink,
        });
    }
}

/// One parsed `Content-Range` response header.
///
/// Both forms the specification defines are represented, because both carry
/// information this reader needs: the 206 form says *which* bytes came back, and
/// the 416 form is the only place a server volunteers the resource's **current**
/// length after refusing a read — which is exactly the drift signal a
/// republished archive produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContentRange {
    /// `bytes <first>-<last>/<total|*>` — the answer to a satisfied range.
    Satisfied {
        /// First byte of the returned run.
        first: u64,
        /// Last byte of the returned run, inclusive.
        last: u64,
        /// Total length of the resource, when the server declared it.
        total: Option<u64>,
    },
    /// `bytes */<total>` — the answer to an unsatisfiable range.
    Unsatisfied {
        /// Total length of the resource.
        total: u64,
    },
}

/// Parses a `Content-Range` header value, or [`None`] when it is not one.
///
/// Deliberately lenient about *shape* — leading/trailing space, an `=` where the
/// specification wants a space, any case of `bytes` — and strict about
/// *content*: a header that does not yield numbers is not half-believed. The
/// caller treats an unparsable header exactly like an absent one, so leniency
/// here only ever buys a check that would otherwise be skipped.
fn parse_content_range(value: &str) -> Option<ContentRange> {
    let trimmed = value.trim();
    if !trimmed.get(..5)?.eq_ignore_ascii_case("bytes") {
        return None;
    }
    let rest = trimmed.get(5..)?.trim_matches([' ', '\t', '=']);
    let (spec, total) = rest.split_once('/')?;
    let total = match total.trim() {
        "*" => None,
        digits => Some(digits.parse::<u64>().ok()?),
    };
    let spec = spec.trim();
    if spec == "*" {
        return total.map(|total| ContentRange::Unsatisfied { total });
    }
    let (first, last) = spec.split_once('-')?;
    Some(ContentRange::Satisfied {
        first: first.trim().parse().ok()?,
        last: last.trim().parse().ok()?,
        total,
    })
}

/// Checks a `206`'s `Content-Range` against what was asked for and what came
/// back, returning the resource length it declared.
///
/// The three rules, in the order a lie is most likely to matter:
///
/// * **absent (or unparsable) is accepted** — real hosts omit the header and the
///   read is otherwise well-formed; the total is simply unknown;
/// * **a different first byte is refused by name**, quoting both ranges, because
///   the readers above validate the offset they *asked* for and would decode
///   whatever arrived as if it were that offset;
/// * **a body length disagreeing with the header is refused**, which is what a
///   transcoding proxy (or a `Content-Encoding` the agent never negotiated)
///   looks like from here.
///
/// # Errors
///
/// A [`TileError::permanent`] naming the disagreement; nothing about a lying
/// intermediary gets better on a retry.
fn verify_content_range(
    header: Option<&str>,
    asked: ByteRange,
    body_len: usize,
) -> Result<Option<u64>, TileError> {
    let Some(parsed) = header.and_then(parse_content_range) else {
        return Ok(None);
    };
    let ContentRange::Satisfied { first, last, total } = parsed else {
        return Err(TileError::permanent(format!(
            "the server answered 206 to {} with an unsatisfied-range header, which is not an \
             answer at all",
            asked.header_value()
        )));
    };
    if first != asked.start {
        return Err(TileError::permanent(format!(
            "the server answered a different range than was asked: asked {}, answered \
             bytes {first}-{last}",
            asked.header_value()
        )));
    }
    let served = last
        .checked_sub(first)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| {
            TileError::permanent(format!(
                "the server answered {} with the inverted range bytes {first}-{last}",
                asked.header_value()
            ))
        })?;
    if served != body_len as u64 {
        return Err(TileError::permanent(format!(
            "the server's Content-Range names {served} bytes for {} but the body holds {body_len}",
            asked.header_value()
        )));
    }
    Ok(total)
}

/// What one URL's first accepted answer pinned about the file behind it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Validator {
    /// The `ETag` the answer carried, verbatim (quotes and any `W/` included).
    etag: Option<String>,
    /// The total length its `Content-Range` declared.
    total: Option<u64>,
}

/// Per-URL validators, pinned at the first answer and compared at every later
/// one.
///
/// Bounded by [`MAX_PINNED_VALIDATORS`] and stored as a small `Vec` rather than
/// a map: at that size the linear scan is free next to the network round trip it
/// guards, and insertion order is what makes "drop the oldest" mean anything.
#[derive(Debug, Default)]
struct ValidatorPins {
    /// `(url, validator)` in first-seen order.
    entries: std::sync::Mutex<Vec<(String, Validator)>>,
}

impl ValidatorPins {
    /// The validator pinned for `url`, if one is.
    fn pinned(&self, url: &str) -> Option<Validator> {
        let entries = self.lock();
        entries
            .iter()
            .find(|(pinned, _)| pinned == url)
            .map(|(_, validator)| validator.clone())
    }

    /// Records the first answer for `url`, or checks a later one against it.
    ///
    /// Fields are merged rather than replaced: a host that sends an `ETag` on
    /// the first answer and none on the second has not changed the file, and the
    /// pin must survive to catch the answer that has.
    ///
    /// # Errors
    ///
    /// A permanent drift refusal when an already-pinned field disagrees.
    fn observe(&self, url: &str, observed: &Validator) -> Result<(), TileError> {
        let mut entries = self.lock();
        if let Some((_, pinned)) = entries.iter_mut().find(|(pinned, _)| pinned == url) {
            if let (Some(was), Some(now)) = (pinned.etag.as_deref(), observed.etag.as_deref())
                && was != now
            {
                return Err(TileError::permanent(format!(
                    "{url}: {DRIFT_ADVICE} (its ETag went from {was} to {now})"
                )));
            }
            if let (Some(was), Some(now)) = (pinned.total, observed.total)
                && was != now
            {
                return Err(TileError::permanent(format!(
                    "{url}: {DRIFT_ADVICE} (its length went from {was} to {now} bytes)"
                )));
            }
            if pinned.etag.is_none() {
                pinned.etag = observed.etag.clone();
            }
            if pinned.total.is_none() {
                pinned.total = observed.total;
            }
            return Ok(());
        }
        if observed.etag.is_none() && observed.total.is_none() {
            // Nothing to pin: a host that volunteers neither validator leaves
            // drift undetectable, which is recorded rather than papered over.
            return Ok(());
        }
        while entries.len() >= MAX_PINNED_VALIDATORS && !entries.is_empty() {
            entries.remove(0);
        }
        entries.push((url.to_owned(), observed.clone()));
        Ok(())
    }

    /// The entry list, recovering the contents if a worker panicked holding it.
    ///
    /// A poisoned mutex here means some *other* read paniced; the pins
    /// themselves are plain data and are still exactly as valid as they were.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(String, Validator)>> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The `If-Match` value for a pinned validator, when one may legally be sent.
///
/// Only a **strong** validator qualifies: RFC 9110 forbids a weak one in
/// `If-Match`, and a host that received one would be within its rights to answer
/// `400`. A weak `ETag` is still pinned and still compared passively.
fn strong_validator(etag: Option<&str>) -> Option<&str> {
    let etag = etag?.trim();
    if etag.is_empty() || etag.starts_with("W/") || etag.starts_with("w/") {
        return None;
    }
    Some(etag)
}

/// One response header as a `String`, when it is present and is text.
fn header_value(response: &ureq::http::Response<ureq::Body>, name: &str) -> Option<String> {
    response
        .headers()
        .get(name)?
        .to_str()
        .ok()
        .map(str::to_owned)
}

/// Performs one blocking ranged `GET`, verifies the answer and classifies the
/// outcome.
///
/// Checks [`crate::tile_http::HostCooldowns`] before dialling at all: a `429`
/// pauses the whole pool's next request to that origin rather than each of
/// the six workers re-discovering the same rate limit on its own next
/// attempt — see that type's docs for why this fails fast instead of
/// sleeping out the pause.
///
/// Statuses are otherwise read off the response rather than raised as `ureq`
/// errors (`http_status_as_error(false)`), because `412` and `416` are
/// answers whose **headers** matter: the first is the drift the pinned
/// validator asked the server to check for us, the second is the only place a
/// server states the resource's current length after refusing a read, and a
/// `429`'s `Retry-After` needs the same access.
fn fetch_range(
    agent: &ureq::Agent,
    pins: &ValidatorPins,
    cooldowns: &crate::tile_http::HostCooldowns,
    url: &str,
    range: ByteRange,
) -> Result<Vec<u8>, TileError> {
    let origin = crate::tile_http::origin_of(url);
    if cooldowns.is_paused(origin) {
        return Err(TileError::transient(format!(
            "{origin} asked to be left alone after a 429/503 Retry-After; not dialled yet"
        )));
    }
    let pinned = pins.pinned(url);
    let mut request = agent
        .get(url)
        .config()
        .http_status_as_error(false)
        .build()
        .header("Range", range.header_value());
    if let Some(etag) = pinned
        .as_ref()
        .and_then(|validator| strong_validator(validator.etag.as_deref()))
    {
        request = request.header("If-Match", etag);
    }
    let response = match request.call() {
        Ok(response) => response,
        Err(error) => return Err(TileError::transient(error.to_string())),
    };
    let status = response.status().as_u16();
    let content_range = header_value(&response, "content-range");
    let etag = header_value(&response, "etag");
    if status != STATUS_PARTIAL_CONTENT {
        if (FIRST_SERVER_ERROR_STATUS..600).contains(&status) || status == STATUS_TOO_MANY_REQUESTS
        {
            cooldowns.note_retry_after(origin, header_value(&response, "retry-after").as_deref());
        }
        return Err(classify_status(
            status,
            content_range.as_deref(),
            pinned.as_ref(),
            url,
            range,
        ));
    }
    let mut body = response.into_body();
    let bytes = body
        .with_config()
        .limit(MAX_RANGE_BYTES)
        .read_to_vec()
        .map_err(|error| TileError::transient(format!("body read failed: {error}")))?;
    let total = verify_content_range(content_range.as_deref(), range, bytes.len())?;
    pins.observe(url, &Validator { etag, total })?;
    Ok(bytes)
}

/// Turns a non-206 status into the failure it means.
///
/// The `Retry-After` side effect lives in the caller ([`fetch_range`]), not
/// here: unlike the drift checks below, it needs the raw response to read a
/// header from, and keeping this function free of that keeps it exactly as
/// unit-testable as it already was.
fn classify_status(
    status: u16,
    content_range: Option<&str>,
    pinned: Option<&Validator>,
    url: &str,
    range: ByteRange,
) -> TileError {
    match status {
        STATUS_PRECONDITION_FAILED => TileError::permanent(format!(
            "{url}: {DRIFT_ADVICE} (the server rejected the pinned ETag)"
        )),
        STATUS_RANGE_NOT_SATISFIABLE => {
            // `bytes */total` states the file's CURRENT length. A different one
            // from the pinned total is the same drift a 412 reports, on a host
            // that ignores `If-Match`.
            if let Some(ContentRange::Unsatisfied { total }) =
                content_range.and_then(parse_content_range)
                && pinned
                    .and_then(|validator| validator.total)
                    .is_some_and(|was| was != total)
            {
                return TileError::permanent(format!(
                    "{url}: {DRIFT_ADVICE} (its length is now {total} bytes)"
                ));
            }
            TileError::permanent(format!(
                "the server answered HTTP 416 to {}: that range is not inside the resource",
                range.header_value()
            ))
        }
        STATUS_TOO_MANY_REQUESTS => TileError::transient(format!(
            "the server answered HTTP 429 to {}: too many requests, backing off",
            range.header_value()
        )),
        code if (FIRST_SERVER_ERROR_STATUS..600).contains(&code) => {
            TileError::transient(format!("HTTP {code}"))
        }
        code if code >= FIRST_NON_SUCCESS_STATUS => TileError::permanent(format!("HTTP {code}")),
        code => TileError::permanent(format!(
            "server answered HTTP {code} instead of 206 for {}: the resource does not support \
             Range requests, which a COG must",
            range.header_value()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ContentRange, DRIFT_ADVICE, HttpRangeTransport, MAX_PINNED_VALIDATORS,
        RANGE_WORKER_THREADS, Validator, ValidatorPins, classify_status, parse_content_range,
        strong_validator, verify_content_range,
    };
    use oxigis_render::ByteRange;

    /// A non-empty range, the way every caller in this crate builds one.
    fn range(start: u64, end: u64) -> ByteRange {
        ByteRange::new(start, end).expect("a non-empty range")
    }

    #[test]
    fn the_transport_starts_a_worker_pool() {
        let transport = HttpRangeTransport::new().expect("worker threads must start");
        assert!(format!("{transport:?}").contains(&RANGE_WORKER_THREADS.to_string()));
    }

    // -----------------------------------------------------------------------
    // Content-Range: the lying-transport battery
    // -----------------------------------------------------------------------

    #[test]
    fn content_range_parses_both_forms_and_refuses_nonsense() {
        assert_eq!(
            parse_content_range("bytes 0-15/1234"),
            Some(ContentRange::Satisfied {
                first: 0,
                last: 15,
                total: Some(1234),
            })
        );
        // A total the server declines to state is legal and common.
        assert_eq!(
            parse_content_range("bytes 100-199/*"),
            Some(ContentRange::Satisfied {
                first: 100,
                last: 199,
                total: None,
            })
        );
        // The 416 form, which is the only place a length arrives after refusal.
        assert_eq!(
            parse_content_range("bytes */900"),
            Some(ContentRange::Unsatisfied { total: 900 })
        );
        // Shape leniency: case, spacing and the `=` some servers write.
        assert_eq!(
            parse_content_range("  BYTES= 4 - 9 / 10 "),
            Some(ContentRange::Satisfied {
                first: 4,
                last: 9,
                total: Some(10),
            })
        );
        for nonsense in [
            "",
            "items 0-1/2",
            "bytes",
            "bytes 0-15",
            "bytes a-b/10",
            "bytes */*",
            "bytes 0-15/x",
        ] {
            assert_eq!(parse_content_range(nonsense), None, "{nonsense}");
        }
    }

    #[test]
    fn an_absent_or_unparsable_content_range_is_accepted_with_no_total() {
        // The acceptance rule the live battery depends on: real hosts omit it.
        assert_eq!(verify_content_range(None, range(0, 16), 16), Ok(None));
        assert_eq!(
            verify_content_range(Some("nonsense"), range(0, 16), 16),
            Ok(None)
        );
    }

    #[test]
    fn a_server_answering_a_different_range_is_refused_by_name() {
        // The failure v1.3 could not see: 206, right length, WRONG bytes.
        let error = verify_content_range(Some("bytes 4096-4111/99999"), range(0, 16), 16)
            .expect_err("a different range must be refused");
        assert!(!error.retryable(), "{error}");
        assert!(error.message().contains("bytes=0-15"), "{error}");
        assert!(error.message().contains("bytes 4096-4111"), "{error}");
    }

    #[test]
    fn a_content_range_disagreeing_with_the_body_length_is_refused() {
        let error = verify_content_range(Some("bytes 0-99/1000"), range(0, 100), 40)
            .expect_err("a transcoded body must be refused");
        assert!(!error.retryable(), "{error}");
        assert!(error.message().contains("100 bytes"), "{error}");
        assert!(error.message().contains("40"), "{error}");
    }

    #[test]
    fn a_short_but_honest_answer_is_accepted_and_reports_the_total() {
        // The speculative 16 KiB prefetch over a 282-byte archive: short, and
        // completely legitimate.
        assert_eq!(
            verify_content_range(Some("bytes 0-281/282"), range(0, 16_384), 282),
            Ok(Some(282))
        );
    }

    #[test]
    fn an_unsatisfied_header_on_a_206_is_refused() {
        let error = verify_content_range(Some("bytes */282"), range(0, 16), 16)
            .expect_err("`bytes */n` is not an answer to a satisfied range");
        assert!(!error.retryable(), "{error}");
    }

    #[test]
    fn an_inverted_content_range_is_refused_rather_than_underflowing() {
        let error = verify_content_range(Some("bytes 10-4/100"), range(10, 20), 6)
            .expect_err("an inverted range must be refused");
        assert!(error.message().contains("inverted"), "{error}");
    }

    // -----------------------------------------------------------------------
    // Validator pinning
    // -----------------------------------------------------------------------

    #[test]
    fn the_first_answer_pins_and_a_changed_etag_is_permanent_drift() {
        let pins = ValidatorPins::default();
        let first = Validator {
            etag: Some("\"abc\"".to_owned()),
            total: Some(1000),
        };
        assert_eq!(pins.observe("https://host/a.pmtiles", &first), Ok(()));
        assert_eq!(pins.pinned("https://host/a.pmtiles"), Some(first));

        let republished = Validator {
            etag: Some("\"def\"".to_owned()),
            total: Some(1000),
        };
        let error = pins
            .observe("https://host/a.pmtiles", &republished)
            .expect_err("a changed ETag is drift");
        assert!(!error.retryable(), "{error}");
        assert!(error.message().contains(DRIFT_ADVICE), "{error}");
    }

    #[test]
    fn a_changed_total_length_is_drift_even_when_no_etag_is_offered() {
        // The wasm-shaped case: a host exposing no ETag at all still cannot
        // change its length underneath an open archive unnoticed.
        let pins = ValidatorPins::default();
        assert_eq!(
            pins.observe(
                "https://host/planet.pmtiles",
                &Validator {
                    etag: None,
                    total: Some(137_000_000_000),
                },
            ),
            Ok(())
        );
        let error = pins
            .observe(
                "https://host/planet.pmtiles",
                &Validator {
                    etag: None,
                    total: Some(137_000_000_001),
                },
            )
            .expect_err("a changed length is drift");
        assert!(error.message().contains(DRIFT_ADVICE), "{error}");
    }

    #[test]
    fn a_later_answer_that_volunteers_nothing_neither_clears_nor_trips_the_pin() {
        let pins = ValidatorPins::default();
        let pinned = Validator {
            etag: Some("\"abc\"".to_owned()),
            total: Some(1000),
        };
        assert_eq!(pins.observe("u", &pinned), Ok(()));
        assert_eq!(pins.observe("u", &Validator::default()), Ok(()));
        assert_eq!(pins.pinned("u"), Some(pinned));
    }

    #[test]
    fn a_field_missing_from_the_first_answer_is_filled_in_by_a_later_one() {
        let pins = ValidatorPins::default();
        assert_eq!(
            pins.observe(
                "u",
                &Validator {
                    etag: Some("\"abc\"".to_owned()),
                    total: None,
                },
            ),
            Ok(())
        );
        assert_eq!(
            pins.observe(
                "u",
                &Validator {
                    etag: None,
                    total: Some(42),
                },
            ),
            Ok(())
        );
        assert_eq!(
            pins.pinned("u"),
            Some(Validator {
                etag: Some("\"abc\"".to_owned()),
                total: Some(42),
            })
        );
    }

    #[test]
    fn the_pin_store_is_bounded() {
        let pins = ValidatorPins::default();
        for index in 0..(MAX_PINNED_VALIDATORS * 2) {
            assert_eq!(
                pins.observe(
                    &format!("https://host/{index}.pmtiles"),
                    &Validator {
                        etag: Some(format!("\"{index}\"")),
                        total: Some(index as u64),
                    },
                ),
                Ok(())
            );
        }
        assert_eq!(pins.lock().len(), MAX_PINNED_VALIDATORS);
        // The oldest went; the newest is still pinned.
        assert!(pins.pinned("https://host/0.pmtiles").is_none());
        assert!(
            pins.pinned(&format!(
                "https://host/{}.pmtiles",
                MAX_PINNED_VALIDATORS * 2 - 1
            ))
            .is_some()
        );
    }

    #[test]
    fn only_a_strong_validator_is_ever_sent_as_if_match() {
        assert_eq!(strong_validator(Some("\"abc\"")), Some("\"abc\""));
        assert_eq!(strong_validator(Some("W/\"abc\"")), None);
        assert_eq!(strong_validator(Some("w/\"abc\"")), None);
        assert_eq!(strong_validator(Some("   ")), None);
        assert_eq!(strong_validator(None), None);
    }

    // -----------------------------------------------------------------------
    // Status classification
    // -----------------------------------------------------------------------

    #[test]
    fn a_412_is_the_drift_refusal_and_a_200_is_the_range_ignoring_one() {
        let drift = classify_status(412, None, None, "https://host/a.pmtiles", range(0, 16));
        assert!(!drift.retryable(), "{drift}");
        assert!(drift.message().contains(DRIFT_ADVICE), "{drift}");

        let ignored = classify_status(200, None, None, "https://host/a.pmtiles", range(0, 16));
        assert!(!ignored.retryable(), "{ignored}");
        assert!(ignored.message().contains("instead of 206"), "{ignored}");
    }

    #[test]
    fn a_416_naming_a_new_length_reports_drift_and_otherwise_reports_416() {
        let pinned = Validator {
            etag: None,
            total: Some(1000),
        };
        let drift = classify_status(
            416,
            Some("bytes */2000"),
            Some(&pinned),
            "https://host/a.pmtiles",
            range(1500, 1516),
        );
        assert!(drift.message().contains(DRIFT_ADVICE), "{drift}");

        let plain = classify_status(
            416,
            Some("bytes */1000"),
            Some(&pinned),
            "https://host/a.pmtiles",
            range(1500, 1516),
        );
        assert!(!plain.retryable(), "{plain}");
        assert!(plain.message().contains("416"), "{plain}");
        assert!(!plain.message().contains(DRIFT_ADVICE), "{plain}");
    }

    #[test]
    fn server_errors_stay_retryable_and_client_errors_do_not() {
        let transient = classify_status(503, None, None, "u", range(0, 16));
        assert!(transient.retryable(), "{transient}");
        let permanent = classify_status(404, None, None, "u", range(0, 16));
        assert!(!permanent.retryable(), "{permanent}");
    }

    // -----------------------------------------------------------------------
    // 429 is retryable, and `fetch_range` honours a pause
    // `crate::tile_http::HostCooldowns` already knows about.
    // -----------------------------------------------------------------------

    #[test]
    fn a_429_is_retryable_and_names_itself() {
        let rate_limited = classify_status(429, None, None, "u", range(0, 16));
        assert!(rate_limited.retryable(), "{rate_limited}");
        assert!(rate_limited.message().contains("429"), "{rate_limited}");
    }

    #[test]
    fn a_paused_origin_is_refused_by_fetch_range_without_dialling() {
        // `.invalid` never resolves (RFC 2606); a bug that let this call
        // through to the network would hang or fail on DNS instead of
        // returning immediately from the cooldown check.
        let agent = crate::tile_http::build_agent_with_timeout(
            super::RANGE_REQUEST_TIMEOUT,
            RANGE_WORKER_THREADS,
        );
        let pins = ValidatorPins::default();
        let cooldowns = crate::tile_http::HostCooldowns::default();
        let origin = "http://rate-limited.invalid";
        cooldowns.note_retry_after(origin, Some("60"));
        assert!(cooldowns.is_paused(origin));

        let error = super::fetch_range(
            &agent,
            &pins,
            &cooldowns,
            "http://rate-limited.invalid/tile",
            range(0, 16),
        )
        .expect_err("a paused origin must fail without a network round trip");
        assert!(error.retryable(), "{error}");
    }

    /// Live smoke test against a public EPSG:4326 Cloud-Optimized GeoTIFF.
    ///
    /// Ignored by default so the standard suite stays offline. Run it
    /// deliberately with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: reads a real COG from a public S3 bucket"]
    fn live_cog_round_trip() {
        // ESA WorldCover 2020, 10 m land cover: a public EPSG:4326 COG with
        // DEFLATE-compressed 8-bit palette tiles and a seven-level overview
        // pyramid — the axis-separable placement path, no reprojection.
        //
        // Override with `OXIGIS_LIVE_COG_URL` to point the test elsewhere.
        const URL: &str = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/\
                           v100/2020/map/ESA_WorldCover_10m_2020_v100_N51E000_Map.tif";
        live_cog_round_trip_at("OXIGIS_LIVE_COG_URL", URL, None);
    }

    /// Live smoke test against a public **UTM** Cloud-Optimized GeoTIFF, i.e.
    /// the reprojecting path.
    ///
    /// Ignored by default; run with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: reads a real Sentinel-2 COG from a public S3 bucket"]
    fn live_utm_cog_round_trip() {
        // Sentinel-2 L2A true-colour composite over Tokyo Bay, from the AWS
        // `sentinel-cogs` open-data archive: EPSG:32654 (WGS 84 / UTM zone 54N),
        // 10 980² at 10 m, three 8-bit bands, DEFLATE, five overview levels.
        // This is the product class the CRS work exists for — before it, this
        // file was reported as `Unsupported`.
        //
        // Override with `OXIGIS_LIVE_UTM_COG_URL` to point the test elsewhere.
        const URL: &str = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/\
                           sentinel-s2-l2a-cogs/54/S/UD/2021/11/\
                           S2B_54SUD_20211128_0_L2A/TCI.tif";
        live_cog_round_trip_at("OXIGIS_LIVE_UTM_COG_URL", URL, Some(32_654));
    }

    /// The body of the live smoke tests: exercises the whole production path —
    /// `CogTileProvider` → this transport → ureq/rustls-graviola over TLS →
    /// `oxigis_render`'s TIFF parser and codec → tile composition, i.e. the same
    /// code the app runs.
    ///
    /// The tiles it asks for are derived from the file's own georeference rather
    /// than hard-coded, so the test cannot silently start passing on transparent
    /// pixels if the URL changes. `expect_epsg`, when given, asserts the file is
    /// the CRS the caller thinks it is — it guards the *default* URL, so an
    /// `env_var` override (which may point at any CRS) skips it.
    fn live_cog_round_trip_at(env_var: &str, default_url: &str, expect_epsg: Option<u32>) {
        use oxigis_render::{TileId, WorldCoord};
        use oxigis_ui::{CogLayerConfig, CogTileProvider, TileProvider as _};

        // Surface the provider's `tracing::warn!` diagnostics (per-tile fetch
        // and decode failures) on the test's stdout; ignore the error if a
        // sibling test in the same process already installed a subscriber.
        let _ = tracing_subscriber::fmt().try_init();

        let transport = HttpRangeTransport::new().expect("worker threads must start");
        let overridden = std::env::var(env_var).is_ok();
        let url = std::env::var(env_var).unwrap_or_else(|_| default_url.to_owned());
        println!("live COG: {url}");
        let provider = CogTileProvider::new(
            &CogLayerConfig::new(url),
            &egui::Context::default(),
            Box::new(transport),
        )
        .expect("the provider must build");

        // Any tile request kicks the header read; the answer is irrelevant here.
        let probe = TileId::new(0, 0, 0).expect("0/0/0 is valid");
        let mut metadata = None;
        for _ in 0..150 {
            let _ = provider.tile(probe);
            if let Some(found) = provider.metadata() {
                metadata = Some(found);
                break;
            }
            if let Some(failure) = provider.failure() {
                panic!("the COG failed to open: {failure}");
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let metadata = metadata.expect("the COG header must arrive within 15 s");
        println!(
            "live COG: {} levels, EPSG {:?}, base {}x{}",
            metadata.level_count(),
            metadata.epsg,
            metadata.base_level().expect("a base level").width,
            metadata.base_level().expect("a base level").height,
        );
        if let Some(epsg) = expect_epsg
            && !overridden
        {
            assert_eq!(
                metadata.epsg,
                Some(epsg),
                "the fixture URL must still be the CRS this test is about"
            );
        }
        println!("live COG: classified as {:?}", metadata.crs());

        // A zoom at which one map tile is about one source pixel per screen pixel,
        // and the tile at the image's centre.
        let pixel = metadata
            .world_pixel_size(0)
            .expect("a georeferenced COG must report a pixel size");
        let zoom = (1.0 / (256.0 * pixel)).log2().floor().clamp(0.0, 20.0);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to 0..=20 immediately above"
        )]
        let zoom = zoom as u8;
        let (min_x, min_y, max_x, max_y) = metadata.world_bounds().expect("world bounds");
        let center = WorldCoord::new((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
        let tile = center.tile(zoom).expect("a valid tile");
        println!("live COG: sampling tile {}/{}/{}", tile.z, tile.x, tile.y);

        let mut decoded = None;
        for attempt in 0u32..1200 {
            if let Some(pixels) = provider.tile(tile) {
                decoded = Some(pixels);
                break;
            }
            if attempt % 100 == 99 {
                let stats = provider.stats();
                println!(
                    "live COG: waiting {} s: ready {} inflight {} failed {}",
                    (attempt + 1) / 10,
                    stats.ready,
                    stats.inflight,
                    stats.failed
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let pixels = decoded.expect("a COG tile must arrive within 120 s");
        assert_eq!(pixels.width(), 256);
        assert_eq!(pixels.height(), 256);
        assert!(
            pixels.rgba().chunks_exact(4).any(|px| px[3] > 0),
            "a tile at the centre of the image must have visible pixels"
        );
        let first = &pixels.rgba()[0..4];
        assert!(
            pixels.rgba().chunks_exact(4).any(|px| px != first),
            "real imagery is not a single flat colour"
        );

        // …and again at a zoom that forces an *overview* level, so the per-IFD
        // sample layout, codec, predictor and ColorMap are all exercised against
        // a real file rather than only the full-resolution directory.
        let coarse_zoom = zoom.saturating_sub(4);
        let coarse_level = metadata
            .select_level(coarse_zoom)
            .expect("a level for the coarse zoom");
        println!("live COG: coarse zoom {coarse_zoom} selects level {coarse_level}");
        assert!(
            coarse_level > 0,
            "four zoom levels out must reach an overview, not level 0"
        );
        let coarse_tile = center.tile(coarse_zoom).expect("a valid tile");
        let mut coarse = None;
        for _ in 0u32..1200 {
            if let Some(found) = provider.tile(coarse_tile) {
                coarse = Some(found);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let coarse = coarse.expect("an overview tile must arrive within 120 s");
        assert!(
            coarse.rgba().chunks_exact(4).any(|px| px[3] > 0),
            "the overview tile must have visible pixels"
        );
    }

    // -----------------------------------------------------------------------
    // Live PMTiles round trips (tiles v1.4, stage T1)
    //
    // Five archives on three hosts, exercising the whole production path:
    // `ArchiveProbe` / `ArchiveTileProvider` / `ArchiveTileTransport` → this
    // transport → ureq/rustls-graviola → `oxigis_render::pmtiles` → MVT or
    // image decode. `oxigis-ui`'s own `archive::tests` prove the same shapes
    // offline against hand-built fixtures; these prove they are also true of
    // files real writers actually produced.
    //
    // Two rules make them survivable:
    //
    // * **every tile address is derived from the archive's own header** —
    //   centre, bounds and zoom range — never hard-coded. A hand-picked
    //   coordinate that has drifted out of the data answers `Absent`, and a
    //   test asserting on `Absent` would go green while asserting nothing.
    // * **assertions are shapes, never byte offsets.** The planet build is
    //   republished continuously; its directory layout is different every week.
    //
    // Each takes an `OXIGIS_LIVE_*_URL` override, the `live_cog_round_trip`
    // idiom, so a host that disappears can be pointed elsewhere without a code
    // change.
    // -----------------------------------------------------------------------

    /// Every job a [`CountingRangeTransport`] was handed.
    type JobLog = std::sync::Arc<std::sync::Mutex<Vec<super::RangeJob>>>;

    /// A [`RangeTransport`] that records what the reader asked the network for
    /// before handing the job to the real one.
    ///
    /// What makes "the open took two reads", "the neighbour cost no leaf" and
    /// "a zoom past the archive's own maximum reads nothing at all" assertable
    /// against a live host rather than only against a fixture.
    struct CountingRangeTransport {
        /// The real transport.
        inner: HttpRangeTransport,
        /// Every job asked for, in order.
        log: JobLog,
    }

    impl CountingRangeTransport {
        fn new() -> (Self, JobLog) {
            let log: JobLog = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let transport = HttpRangeTransport::new().expect("worker threads must start");
            (
                Self {
                    inner: transport,
                    log: std::sync::Arc::clone(&log),
                },
                log,
            )
        }
    }

    impl super::RangeTransport for CountingRangeTransport {
        fn request_range(
            &self,
            url: String,
            range: oxigis_render::ByteRange,
            job: super::RangeJob,
            sink: super::RangeSink,
        ) {
            self.log
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(job);
            self.inner.request_range(url, range, job, sink);
        }
    }

    /// Every job recorded so far.
    fn recorded(log: &JobLog) -> Vec<super::RangeJob> {
        log.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// How many reads of each kind have been issued: `(header, leaf, tile)`.
    fn read_counts(log: &JobLog) -> (usize, usize, usize) {
        let jobs = recorded(log);
        let count =
            |want: fn(&super::RangeJob) -> bool| jobs.iter().filter(|job| want(job)).count();
        (
            count(|job| matches!(job, super::RangeJob::ArchiveHeader { .. })),
            count(|job| matches!(job, super::RangeJob::ArchiveLeaf { .. })),
            count(|job| matches!(job, super::RangeJob::ArchiveTile { .. })),
        )
    }

    /// The URL to read: the caller's default, or whatever `env_var` overrides
    /// it with.
    fn live_url(env_var: &str, default_url: &str) -> String {
        std::env::var(env_var).unwrap_or_else(|_| default_url.to_owned())
    }

    /// Reads an archive's own header through the production probe, returning
    /// what it says and how many range reads that took.
    fn probe_live_archive(url: &str) -> (oxigis_ui::ArchiveInfo, usize) {
        let (transport, log) = CountingRangeTransport::new();
        let ctx = egui::Context::default();
        let probe = oxigis_ui::ArchiveProbe::start(
            url,
            oxigis_core::ArchiveFormat::PmTiles,
            &ctx,
            Box::new(transport),
        );
        for _ in 0..300 {
            if let Some(answer) = probe.take() {
                let opened =
                    answer.unwrap_or_else(|error| panic!("{url} could not be opened: {error}"));
                return (opened.info, recorded(&log).len());
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        panic!("{url} did not answer its header within 30 s");
    }

    /// The tile covering the archive's **own** declared centre at `zoom`.
    fn centre_tile(info: &oxigis_ui::ArchiveInfo, zoom: u8) -> oxigis_render::TileId {
        oxigis_render::LonLat::new(info.center_deg.0, info.center_deg.1)
            .tile(zoom)
            .unwrap_or_else(|error| {
                panic!("the archive's centre is not a tile at z{zoom}: {error}")
            })
    }

    /// The tile immediately east of `tile`, or west of it at the eastern edge.
    fn neighbour_of(tile: oxigis_render::TileId) -> oxigis_render::TileId {
        let last = oxigis_render::TileId::tiles_per_axis(tile.z).saturating_sub(1);
        let x = if tile.x < last {
            tile.x.saturating_add(1)
        } else {
            tile.x.saturating_sub(1)
        };
        oxigis_render::TileId::new(tile.z, x, tile.y).unwrap_or(tile)
    }

    /// A vector provider reading `url` through the archive transport, plus the
    /// observation handle onto the same archive and the read log.
    fn live_vector_over(
        url: &str,
        info: &oxigis_ui::ArchiveInfo,
    ) -> (
        oxigis_ui::VectorTileProvider,
        oxigis_ui::ArchiveTileTransport,
        JobLog,
    ) {
        let (transport, log) = CountingRangeTransport::new();
        let archive = oxigis_ui::ArchiveTileTransport::pmtiles(url, Box::new(transport));
        let handle = archive.clone();
        // The production configuration, built the way `app/providers.rs` builds
        // it: an archive-backed config expands no URL template at all.
        let config = oxigis_ui::VectorTileConfig::from_archive(
            oxigis_ui::ArchiveLayerConfig::new(
                oxigis_core::ArchiveRef::Url {
                    url: url.to_owned(),
                },
                oxigis_core::ArchiveFormat::PmTiles,
            ),
            oxigis_ui::archive_paints(&info.layer_names),
        );
        let provider = oxigis_ui::VectorTileProvider::new(
            &config,
            &egui::Context::default(),
            Box::new(archive),
        )
        .expect("the vector provider must build");
        (provider, handle, log)
    }

    /// Drives `provider` until `tile` decodes, or gives up after `frames`
    /// 100 ms polls.
    fn pump_vector(
        provider: &oxigis_ui::VectorTileProvider,
        view: oxigis_render::MapView,
        tile: oxigis_render::TileId,
        frames: u32,
    ) -> Option<std::sync::Arc<oxigis_render::VectorTile>> {
        use oxigis_ui::VectorTileSource as _;
        for _ in 0..frames {
            let _invalidated = provider.begin_frame(view);
            let _mesh = provider.mesh(tile);
            if let Some(decoded) = provider.decoded(tile) {
                return Some(decoded);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        None
    }

    /// Drives a raster provider until `tile` answers, or gives up.
    fn pump_raster(
        provider: &oxigis_ui::ArchiveTileProvider,
        tile: oxigis_render::TileId,
        frames: u32,
    ) -> Option<oxigis_render::DecodedTile> {
        use oxigis_ui::TileProvider as _;
        for _ in 0..frames {
            if let Some(decoded) = provider.tile(tile) {
                return Some(decoded);
            }
            if let Some(failure) = provider.failure() {
                panic!("the archive failed: {failure}");
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        None
    }

    /// A view centred on the archive's own centre, at `zoom`.
    fn live_view(info: &oxigis_ui::ArchiveInfo, zoom: u8) -> oxigis_render::MapView {
        oxigis_render::MapView::new(
            oxigis_render::LonLat::new(info.center_deg.0, info.center_deg.1),
            f64::from(zoom),
            [512.0, 512.0],
        )
        .expect("a valid view")
    }

    /// A **planet-scale vector archive whose root directory is entirely leaf
    /// pointers**, so every single lookup is a genuine two-level walk: 2.6 GB of
    /// ODbL basemap, zoom 0–10, 327 129 tile entries behind 814 510 bytes of
    /// leaf directories reached through a 389-byte root.
    ///
    /// This is where the leaf cache earns its existence: the first tile pays a
    /// round trip for its leaf and its neighbour pays none.
    ///
    /// Ignored by default; run with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: reads a multi-gigabyte PMTiles planet archive over HTTP Range"]
    fn live_pmtiles_planet_leaf_hop() {
        // Override with `OXIGIS_LIVE_PMTILES_PLANET_URL`.
        const URL: &str = "https://r2-public.protomaps.com/protomaps-sample-datasets/\
                           protomaps_vector_planet_odbl_z10.pmtiles";
        let _ = tracing_subscriber::fmt().try_init();
        let url = live_url("OXIGIS_LIVE_PMTILES_PLANET_URL", URL);
        println!("live PMTiles planet: {url}");

        let (info, opening_reads) = probe_live_archive(&url);
        println!(
            "live PMTiles planet: {}, centre {:?} at z{}, {} reads to open",
            info.summary(),
            info.center_deg,
            info.center_zoom,
            opening_reads
        );
        assert_eq!(info.content, oxigis_ui::ArchiveContent::Vector);
        assert!(
            opening_reads <= 2,
            "PMTiles v3 promises header+root inside the first 16 KiB, so an open is at most a \
             prefetch plus one far-metadata read; this took {opening_reads}"
        );

        // A zoom the archive itself declares, deep enough that the centre tile
        // has a neighbour to share a leaf with.
        let zoom = info.max_zoom.min(info.center_zoom.max(4));
        assert!(zoom >= 1, "the planet build must hold more than zoom 0");
        let address = centre_tile(&info, zoom);
        let neighbour = neighbour_of(address);
        println!(
            "live PMTiles planet: sampling {}/{}/{} and {}/{}/{}",
            address.z, address.x, address.y, neighbour.z, neighbour.x, neighbour.y
        );

        let (provider, archive, log) = live_vector_over(&url, &info);
        let view = live_view(&info, zoom);
        let decoded = pump_vector(&provider, view, address, 1200)
            .expect("a planet tile must arrive within 120 s");
        assert!(
            !decoded.layers.is_empty(),
            "a planet basemap tile holds named MVT layers"
        );
        assert!(
            decoded
                .layers
                .iter()
                .any(|layer| !layer.features.is_empty()),
            "and at least one of them holds features"
        );

        let (leaves, leaf_bytes) = archive.leaf_stats();
        println!("live PMTiles planet: leaf cache holds {leaves} leaves, {leaf_bytes} bytes");
        assert!(
            leaves >= 1,
            "the planet's root is all leaf pointers, so the lookup must have hopped"
        );
        // Derived from the archive's own shape rather than chosen: 814 510
        // bytes of leaf directories behind a 389-byte root of leaf pointers is
        // kilobytes per leaf however they are counted — stored or decoded — and
        // one byte would mean the cache is holding a stub.
        assert!(
            leaf_bytes >= 1024,
            "a real leaf directory is kilobytes; {leaf_bytes} is not one"
        );
        let (_, leaf_reads, _) = read_counts(&log);
        assert!(leaf_reads >= 1, "and that hop was a real range read");

        // The neighbour rides the leaf already held: this is the difference
        // between a pan costing one read per tile and one *round trip* per tile.
        let held = archive.leaf_stats().0;
        let leaf_reads_before = read_counts(&log).1;
        let _ = pump_vector(&provider, view, neighbour, 1200)
            .expect("the neighbouring tile must arrive too");
        assert_eq!(
            archive.leaf_stats().0,
            held,
            "the neighbour must not have pulled a second leaf directory"
        );
        assert_eq!(
            read_counts(&log).1,
            leaf_reads_before,
            "and must not have re-read the one already held"
        );

        // The zoom gate: an address past the archive's own maximum is Absent
        // before any directory is consulted, so it costs ZERO range reads.
        let Some(past_max) = info.max_zoom.checked_add(1) else {
            return;
        };
        let too_deep = centre_tile(&info, past_max);
        let reads_before = recorded(&log).len();
        let empty = pump_vector(&provider, view, too_deep, 20)
            .expect("an absent tile becomes an EMPTY tile, not a failure");
        assert!(empty.layers.is_empty(), "an absent tile has no layers");
        assert_eq!(
            recorded(&log).len(),
            reads_before,
            "a zoom past the archive's own maximum must read nothing at all"
        );
        use oxigis_ui::VectorTileSource as _;
        let _ = provider.begin_frame(view);
        assert_eq!(provider.stats().failed, 0, "absent is not a failure");
    }

    /// A **root-only** vector archive on a second host: 514 root entries, no
    /// leaf level at all.
    ///
    /// Two things it pins that the planet cannot: the one-read open, and the
    /// path where `leaf_stats()` legitimately stays `(0, 0)` for ever. Being on
    /// a different host also means one host disappearing does not take the whole
    /// set with it.
    ///
    /// Ignored by default; run with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: reads a public PMTiles sample dataset over HTTP Range"]
    fn live_pmtiles_vector_root_only() {
        // Override with `OXIGIS_LIVE_PMTILES_VECTOR_URL`.
        const URL: &str = "https://r2-public.protomaps.com/protomaps-sample-datasets/\
                           cb_2018_us_zcta510_500k.pmtiles";
        let _ = tracing_subscriber::fmt().try_init();
        let url = live_url("OXIGIS_LIVE_PMTILES_VECTOR_URL", URL);
        println!("live PMTiles vector: {url}");

        let (info, opening_reads) = probe_live_archive(&url);
        println!(
            "live PMTiles vector: {}, {opening_reads} read(s)",
            info.summary()
        );
        assert_eq!(info.content, oxigis_ui::ArchiveContent::Vector);
        assert_eq!(
            opening_reads, 1,
            "a small archive's header, root AND metadata all fit the 16 KiB prefetch"
        );

        // The archive's own minimum zoom: whatever else is sparse, the tile
        // covering its own centre at its own lowest zoom must exist.
        let zoom = info.min_zoom;
        let address = centre_tile(&info, zoom);
        println!(
            "live PMTiles vector: sampling {}/{}/{}",
            address.z, address.x, address.y
        );

        let (provider, archive, log) = live_vector_over(&url, &info);
        let decoded = pump_vector(&provider, live_view(&info, zoom), address, 600)
            .expect("a tile must arrive within 60 s");
        assert!(
            !decoded.layers.is_empty(),
            "the archive's own metadata declares vector layers, so its tiles hold some"
        );

        let (header_reads, leaf_reads, tile_reads) = read_counts(&log);
        assert_eq!(header_reads, 1, "the provider re-opens with one read too");
        assert_eq!(leaf_reads, 0, "a root-only archive never reads a leaf");
        assert_eq!(tile_reads, 1, "so a lookup is exactly one range read");
        assert_eq!(
            archive.leaf_stats(),
            (0, 0),
            "and the leaf cache stays empty for ever"
        );
    }

    /// A **WebP** raster archive whose directories are gzipped and whose tile
    /// bodies are not — the independent-compression rule against a real writer,
    /// which no hand-built fixture can prove.
    ///
    /// Also settles a *verified-absent* neighbour: `Absent` is a final answer,
    /// cached, never retried and never counted as a failure, live.
    ///
    /// Ignored by default; run with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: reads a public WebP PMTiles sample over HTTP Range"]
    fn live_pmtiles_raster_webp() {
        // Override with `OXIGIS_LIVE_PMTILES_WEBP_URL`.
        const URL: &str = "https://pmtiles.io/usgs-mt-whitney-8-15-webp-512.pmtiles";
        let _ = tracing_subscriber::fmt().try_init();
        let url = live_url("OXIGIS_LIVE_PMTILES_WEBP_URL", URL);
        println!("live PMTiles webp: {url}");

        let (info, _reads) = probe_live_archive(&url);
        println!(
            "live PMTiles webp: {}, tile size {:?}",
            info.summary(),
            info.tile_size_px
        );
        assert_eq!(info.content, oxigis_ui::ArchiveContent::Raster);
        assert_eq!(
            info.codec,
            oxigis_render::pmtiles::TileType::Webp,
            "this is the raster codec real PMTiles archives are written in"
        );

        let (transport, log) = CountingRangeTransport::new();
        let provider = oxigis_ui::ArchiveTileProvider::pmtiles(
            url.clone(),
            &egui::Context::default(),
            Box::new(transport),
        )
        .expect("the raster provider must build");

        let zoom = info.min_zoom;
        let address = centre_tile(&info, zoom);
        println!(
            "live PMTiles webp: sampling {}/{}/{}",
            address.z, address.x, address.y
        );
        let decoded =
            pump_raster(&provider, address, 600).expect("a WebP tile must arrive within 60 s");
        // Reaching here at all is the assertion that matters: the directories
        // were gunzipped per `internal_compression` and the body was NOT, per
        // `tile_compression`. Either byte honoured the other way round fails.
        println!(
            "live PMTiles webp: decoded {}x{}",
            decoded.width(),
            decoded.height()
        );
        assert!(decoded.width() >= 256 && decoded.height() >= 256);
        assert_eq!(decoded.width(), decoded.height(), "tiles are square");
        let first = decoded.rgba().get(..4).unwrap_or_default();
        assert!(
            decoded.rgba().chunks_exact(4).any(|pixel| pixel != first),
            "real imagery is not a single flat colour"
        );

        // …and the neighbour, which this archive genuinely does not hold.
        let absent = neighbour_of(address);
        println!(
            "live PMTiles webp: settling the absent neighbour {}/{}/{}",
            absent.z, absent.x, absent.y
        );
        use oxigis_ui::TileProvider as _;
        let ready_before = provider.stats().ready;
        for _ in 0..200 {
            let _ = provider.tile(absent);
            if provider.stats().ready > ready_before {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(
            provider.stats().ready > ready_before,
            "an absent address must settle as a CACHED final answer"
        );
        assert_eq!(
            provider.stats().failed,
            0,
            "a sparse archive's miss is not a failure"
        );
        let settled = recorded(&log).len();
        for _ in 0..5 {
            let _ = provider.tile(absent);
        }
        assert_eq!(
            recorded(&log).len(),
            settled,
            "and is never retried: absent is final"
        );
    }

    /// A **PNG**, leaf-directoried, 30.5 GB terrain archive: the second raster
    /// codec and the second leafed archive, so neither the leaf walk nor the
    /// image path rests on one host or one file.
    ///
    /// Ignored by default; run with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: reads a 30 GB PNG PMTiles terrain archive over HTTP Range"]
    fn live_pmtiles_raster_png() {
        // Override with `OXIGIS_LIVE_PMTILES_PNG_URL`.
        const URL: &str =
            "https://r2-public.protomaps.com/protomaps-sample-datasets/terrarium_z9.pmtiles";
        let _ = tracing_subscriber::fmt().try_init();
        let url = live_url("OXIGIS_LIVE_PMTILES_PNG_URL", URL);
        println!("live PMTiles png: {url}");

        let (info, _reads) = probe_live_archive(&url);
        println!("live PMTiles png: {}", info.summary());
        assert_eq!(info.content, oxigis_ui::ArchiveContent::Raster);
        assert_eq!(info.codec, oxigis_render::pmtiles::TileType::Png);

        let (transport, log) = CountingRangeTransport::new();
        let provider = oxigis_ui::ArchiveTileProvider::pmtiles(
            url.clone(),
            &egui::Context::default(),
            Box::new(transport),
        )
        .expect("the raster provider must build");

        let zoom = info.max_zoom.min(info.center_zoom.max(info.min_zoom));
        let address = centre_tile(&info, zoom);
        println!(
            "live PMTiles png: sampling {}/{}/{}",
            address.z, address.x, address.y
        );
        let decoded =
            pump_raster(&provider, address, 900).expect("a PNG tile must arrive within 90 s");
        assert!(decoded.width() >= 256 && decoded.height() >= 256);
        let first = decoded.rgba().get(..4).unwrap_or_default();
        assert!(
            decoded.rgba().chunks_exact(4).any(|pixel| pixel != first),
            "a terrain tile is not a single flat colour"
        );

        let (leaves, leaf_bytes) = provider.leaf_stats();
        println!("live PMTiles png: leaf cache holds {leaves} leaves, {leaf_bytes} bytes");
        assert!(
            leaves >= 1,
            "this archive has a leaf level; it must be used"
        );
        assert!(leaf_bytes > 0);
        assert!(read_counts(&log).1 >= 1, "and the leaf was a real read");
        assert_eq!(provider.stats().failed, 0);
    }

    /// A live **MBTiles** archive read over HTTP `Range` requests, page by page.
    ///
    /// **No default URL, deliberately.** No public, Range-honouring `.mbtiles`
    /// could be verified this session: the format is distributed as a download,
    /// not as a service, and every candidate either 404s or is served by a host
    /// that ignores `Range`. Inventing a default that rots on the first run would
    /// be worse than none — so this is skipped unless
    /// `OXIGIS_LIVE_MBTILES_URL` is set, and `mbtiles::paged::tests` is the real
    /// coverage: the paged reader is proven byte-for-byte against the resident
    /// one over every fixture shape, offline, on every target.
    ///
    /// Point it at anything: `OXIGIS_LIVE_MBTILES_URL=https://host/tokyo.mbtiles
    /// cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: needs OXIGIS_LIVE_MBTILES_URL; there is no public default"]
    fn live_mbtiles_over_range_requests() {
        let Ok(url) = std::env::var("OXIGIS_LIVE_MBTILES_URL") else {
            println!("live MBTiles: OXIGIS_LIVE_MBTILES_URL is not set; skipping");
            return;
        };
        let _ = tracing_subscriber::fmt().try_init();
        println!("live MBTiles: {url}");

        // The production probe, with the format it was told: a `.mbtiles` URL is
        // surveyed in one 16 KiB read exactly as a `.pmtiles` is, so a refusal —
        // no index, a NOCASE collation, WITHOUT ROWID — lands before a layer
        // could exist.
        let (transport, log) = CountingRangeTransport::new();
        let ctx = egui::Context::default();
        let probe = oxigis_ui::ArchiveProbe::start(
            url.clone(),
            oxigis_core::ArchiveFormat::MbTiles,
            &ctx,
            Box::new(transport),
        );
        let mut answer = None;
        for _ in 0..300 {
            if let Some(found) = probe.take() {
                answer = Some(found);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let opened = answer
            .expect("the survey must answer within 30 s")
            .unwrap_or_else(|error| panic!("{url} could not be surveyed: {error}"));
        let info = opened.info;
        println!(
            "live MBTiles: {}, {} survey read(s)",
            info.summary(),
            recorded(&log).len()
        );
        assert!(
            recorded(&log).len() <= 4,
            "a cold open should be one 16 KiB read, not {}",
            recorded(&log).len()
        );

        let zoom = info.min_zoom;
        let address = centre_tile(&info, zoom);
        println!(
            "live MBTiles: sampling {}/{}/{}",
            address.z, address.x, address.y
        );
        match info.content {
            oxigis_ui::ArchiveContent::Raster => {
                let (transport, log) = CountingRangeTransport::new();
                let provider = oxigis_ui::ArchiveTileProvider::paged_mbtiles(
                    url.clone(),
                    &egui::Context::default(),
                    Box::new(transport),
                    None,
                )
                .expect("the raster provider must build");
                let decoded =
                    pump_raster(&provider, address, 900).expect("a tile must arrive within 90 s");
                assert!(decoded.width() >= 64 && decoded.height() >= 64);
                println!("live MBTiles: {} page reads", recorded(&log).len());
                assert_eq!(provider.stats().failed, 0);
            }
            oxigis_ui::ArchiveContent::Vector => {
                let (transport, log) = CountingRangeTransport::new();
                let archive = oxigis_ui::ArchiveTileTransport::paged_mbtiles(
                    url.clone(),
                    Box::new(transport),
                    None,
                );
                let config = oxigis_ui::VectorTileConfig::from_archive(
                    oxigis_ui::ArchiveLayerConfig::new(
                        oxigis_core::ArchiveRef::Url { url: url.clone() },
                        oxigis_core::ArchiveFormat::MbTiles,
                    ),
                    oxigis_ui::archive_paints(&info.layer_names),
                );
                let provider = oxigis_ui::VectorTileProvider::new(
                    &config,
                    &egui::Context::default(),
                    Box::new(archive),
                )
                .expect("the vector provider must build");
                let decoded = pump_vector(&provider, live_view(&info, zoom), address, 900)
                    .expect("a tile must arrive within 90 s");
                assert!(!decoded.layers.is_empty());
                println!("live MBTiles: {} page reads", recorded(&log).len());
            }
        }
    }

    /// The most dangerous network behaviour in the whole archive path: a server
    /// that **ignores `Range`** and answers `200 OK` with the whole file.
    ///
    /// Those bytes are the file's *header*, so handing them back as if they were
    /// the requested range makes the reader decode a tile directory out of the
    /// magic. `httpbingo.org/bytes/N` does exactly this — verified: `200 OK`
    /// with all 2048 bytes in answer to `Range: bytes=0-15` — which makes it a
    /// public endpoint that can pin the refusal live.
    ///
    /// Ignored by default; run with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: probes a deliberately Range-ignoring endpoint"]
    fn live_range_ignoring_server_is_refused_by_name() {
        // Override with `OXIGIS_LIVE_NO_RANGE_URL`.
        const URL: &str = "https://httpbingo.org/bytes/2048";
        let url = live_url("OXIGIS_LIVE_NO_RANGE_URL", URL);
        println!("live no-Range probe: {url}");

        let agent = crate::tile_http::build_agent_with_timeout(
            super::RANGE_REQUEST_TIMEOUT,
            super::RANGE_WORKER_THREADS,
        );
        let pins = ValidatorPins::default();
        let cooldowns = crate::tile_http::HostCooldowns::default();
        let error = super::fetch_range(&agent, &pins, &cooldowns, &url, range(0, 16))
            .expect_err("a server that ignores Range must be refused, not decoded");
        println!("live no-Range probe: {error}");
        assert!(
            !error.retryable(),
            "a resource that does not support Range never will on a retry: {error}"
        );
        assert!(
            error.message().contains("instead of 206"),
            "the refusal must name what happened: {error}"
        );
    }
}
