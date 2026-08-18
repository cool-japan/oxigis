//! Native tile transport: blocking HTTP on a small worker-thread pool.
//!
//! This is the platform half of `oxigis-ui`'s
//! [`oxigis_ui::TileTransport`] seam for desktop builds. The shared provider
//! ([`oxigis_ui::XyzTileProvider`]) owns the cache, the in-flight set and the
//! retry policy; all this module does is turn a URL into bytes without ever
//! blocking the render thread.
//!
//! # Why threads and not an async runtime
//!
//! `eframe` already owns the event loop, and a tile fetch is a short,
//! independent, blocking request. A handful of `std::thread`s fed by
//! `std::sync::mpsc` channels expresses that with no executor, no `tokio`, and
//! no additional dependency (COOLJAPAN: keep the graph small and Pure Rust).
//! [`HttpTileTransport::request`] only pushes a job into a channel, so it
//! returns in microseconds as the seam requires.
//!
//! [`WORKER_THREADS`] workers pull from one shared [`JobQueue`] rather than
//! each owning a private one round-robin fills: a per-worker queue head-of-
//! line-blocks every job that landed on a worker stalled behind one slow or
//! unreachable host while its siblings sit idle, whereas a free worker always
//! takes the next job off a shared queue regardless of who it was originally
//! meant for. The queue's lock is held only for a push or a pop — nothing next
//! to the network round trip it guards — so a handful of threads contending
//! for it is not a cost worth avoiding a mutex over. Overall concurrency is
//! capped by the provider's in-flight limit
//! (`oxigis_ui::tile_provider::MAX_INFLIGHT_TILES`), so the queue stays short
//! by construction.
//!
//! # HTTP / TLS stack (Pure Rust audit)
//!
//! * [`ureq`] 3.x with `default-features = false`, which drops both the `gzip`
//!   (`flate2`) and `brotli` decoders — `deny.toml` bans those outside the
//!   `png`/`tiff`/`parquet` wrappers, and a PNG tile gains nothing from
//!   transport compression anyway.
//! * `rustls-no-provider` + `rustls-webpki-roots`: rustls with **no** crypto
//!   provider compiled in (ureq's default `rustls` feature would pull in
//!   `ring`, which is partly C) and the bundled Mozilla root store, so there is
//!   no dependency on a system OpenSSL either.
//! * The provider is supplied explicitly as
//!   [`rustls_graviola`](https://crates.io/crates/rustls-graviola) — Rust plus
//!   hand-written assembly, no C compiler, no `-sys` crate, no build script
//!   invoking `cc`. It is passed through `TlsConfig`'s
//!   `unversioned_rustls_crypto_provider` rather than installed as a
//!   process-wide default, so nothing depends on initialisation order and no
//!   other library can win the race.
//!
//! Net effect: HTTPS works out of the box on the default feature set with a
//! C/C++/Fortran-free dependency graph (`cargo tree -i ring`, `-i aws-lc-sys`,
//! `-i openssl-sys` all report "not found").

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use oxigis_render::TileId;
use oxigis_ui::{TileError, TileSink, TileTransport};

/// Number of blocking HTTP worker threads.
///
/// Four is enough to keep a 256-tile-per-screen basemap filling smoothly while
/// staying well inside the OpenStreetMap tile policy's expectation of a modest
/// number of parallel connections.
pub const WORKER_THREADS: usize = 4;

/// Hard limit on a single tile response body, in bytes.
///
/// A 256×256 PNG tile is a few kilobytes; 8 MiB leaves room for high-DPI
/// (512 px) and 16-bit terrain tiles while making a misconfigured URL that
/// points at a large file a bounded mistake rather than an out-of-memory one.
pub const MAX_TILE_BYTES: u64 = 8 * 1024 * 1024;

/// Wall-clock budget for one tile request, connect through last byte.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

/// Wall-clock budget for the TCP+TLS handshake alone.
///
/// Left unset, ureq has no connect timeout of its own — only
/// [`REQUEST_TIMEOUT`] bounds the whole call — so an unreachable host (a
/// black-holed address, a firewall silently dropping `SYN`) ties up a worker
/// for the full request budget before the pool notices. Five seconds is
/// generous for a real handshake and short enough that a dead host stops
/// costing a worker's turn quickly.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// How long ureq's pool keeps an idle connection before closing it.
///
/// ureq's own default is 15 s (`ureq::config::Config::max_idle_age`), so any
/// interactive pause longer than that — a coffee break, a laptop sleeping —
/// discards every pooled connection and the next pan re-handshakes all of
/// them. A minute rides out an ordinary pause without holding sockets open
/// indefinitely.
const IDLE_CONNECTION_AGE: Duration = Duration::from_secs(60);

/// One queued fetch: which tile, from where, and where to report back.
struct Job {
    /// Tile being fetched, echoed back to the sink.
    tile: TileId,
    /// Fully expanded URL to `GET`.
    url: String,
    /// Where the bytes (or the failure) are reported.
    sink: TileSink,
}

/// Guarded state behind a [`JobQueue`]: the jobs themselves, plus whether the
/// owning transport has been dropped.
struct QueueState<T> {
    /// Jobs not yet claimed by a worker, oldest first.
    jobs: VecDeque<T>,
    /// Set once by [`JobQueue::close`]; every worker's [`JobQueue::pop`] loop
    /// checks it after draining `jobs` so a `Drop` neither blocks on nor
    /// out-races an in-flight request.
    closed: bool,
}

/// A job queue shared by every worker in a pool (see the module docs for why
/// this replaced one private queue per worker in tiles v1.6).
///
/// `T` travels from [`JobQueue::push`] (the render thread, via
/// [`HttpTileTransport::request`]) to whichever worker's [`JobQueue::pop`]
/// wakes first, so `T` must be `Send` — every `T` this module instantiates it
/// with already crossed a channel boundary before this type existed.
pub(crate) struct JobQueue<T> {
    /// Jobs and the shutdown flag, behind one lock.
    state: Mutex<QueueState<T>>,
    /// Wakes a worker blocked in `pop` when `push` or `close` changes `state`.
    ready: Condvar,
}

impl<T> JobQueue<T> {
    /// An empty, open queue.
    ///
    /// `pub(crate)`: every member is, so that [`crate::range_http`] — which
    /// has its own `Job` type and its own worker pool — can share this queue
    /// implementation rather than reimplementing it.
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(QueueState {
                jobs: VecDeque::new(),
                closed: false,
            }),
            ready: Condvar::new(),
        }
    }

    /// Queues `job` and wakes one worker blocked in [`Self::pop`].
    pub(crate) fn push(&self, job: T) {
        let mut state = self.lock();
        state.jobs.push_back(job);
        drop(state);
        self.ready.notify_one();
    }

    /// Blocks until a job is available, returning [`None`] once the queue is
    /// closed and drained — a worker's signal to end its loop.
    ///
    /// Jobs already queued are always handed out before `None`, even after
    /// [`Self::close`]: a job accepted by [`JobQueue::push`] is a promise the
    /// caller's sink gets an answer, and dropping it silently on shutdown
    /// would break that promise for no reason — the worker is right here,
    /// about to exit anyway.
    pub(crate) fn pop(&self) -> Option<T> {
        let mut state = self.lock();
        loop {
            if let Some(job) = state.jobs.pop_front() {
                return Some(job);
            }
            if state.closed {
                return None;
            }
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Ends every worker's [`Self::pop`] loop once the jobs already queued are
    /// drained. Called once, from `Drop`.
    pub(crate) fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        drop(state);
        self.ready.notify_all();
    }

    /// The guarded state, recovering it if a worker panicked mid-request while
    /// holding the lock — a poisoned lock here means some *other* request
    /// panicked; the queue itself is still exactly as valid as it was.
    fn lock(&self) -> MutexGuard<'_, QueueState<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Blocking-HTTP [`TileTransport`] for native builds.
///
/// Owns [`WORKER_THREADS`] detached worker threads pulling from one shared
/// [`JobQueue`]; dropping the transport closes the queue, which wakes every
/// worker and ends its loop once the jobs already queued (including any
/// request a worker is mid-flight on) are done. Cheap to construct once per
/// map.
pub struct HttpTileTransport {
    /// Jobs not yet claimed by a worker.
    queue: Arc<JobQueue<Job>>,
    /// How many workers were started — for [`Debug`] and the empty-pool guard
    /// in [`TileTransport::request`]; [`WORKER_THREADS`] is never zero today,
    /// but the guard stays in case that ever becomes configurable.
    worker_count: usize,
}

impl HttpTileTransport {
    /// Builds the HTTP agent and starts the worker pool.
    ///
    /// The `User-Agent` identifies the application as the OpenStreetMap tile
    /// usage policy requires, from `CARGO_PKG_*` metadata so it tracks the
    /// crate version automatically.
    ///
    /// # Errors
    ///
    /// Returns the [`std::io::Error`] from [`std::thread::Builder::spawn`] if
    /// the OS refuses to start a worker thread; the caller should fall back to
    /// the synthetic tile source rather than treating this as fatal.
    pub fn new() -> Result<Self, std::io::Error> {
        let agent = Arc::new(build_agent());
        let cooldowns = Arc::new(HostCooldowns::default());
        let queue = Arc::new(JobQueue::<Job>::new());
        for index in 0..WORKER_THREADS {
            let agent = Arc::clone(&agent);
            let cooldowns = Arc::clone(&cooldowns);
            let queue = Arc::clone(&queue);
            std::thread::Builder::new()
                .name(format!("oxigis-tile-{index}"))
                .spawn(move || {
                    while let Some(job) = queue.pop() {
                        let result = fetch(&agent, &cooldowns, &job.url);
                        job.sink.deliver(job.tile, result);
                    }
                })?;
        }
        Ok(Self {
            queue,
            worker_count: WORKER_THREADS,
        })
    }

    /// The `User-Agent` sent with every tile request.
    #[must_use]
    pub fn user_agent() -> String {
        format!(
            "OxiGIS/{} ({})",
            env!("CARGO_PKG_VERSION"),
            env!("CARGO_PKG_REPOSITORY")
        )
    }
}

impl Drop for HttpTileTransport {
    fn drop(&mut self) {
        // Non-blocking: this only flips the flag and wakes the workers, it
        // does not wait for them to exit — they are detached and finish on
        // their own time, exactly as when they each owned a channel.
        self.queue.close();
    }
}

impl core::fmt::Debug for HttpTileTransport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HttpTileTransport")
            .field("workers", &self.worker_count)
            .finish()
    }
}

impl TileTransport for HttpTileTransport {
    fn request(&self, tile: TileId, url: String, sink: TileSink) {
        if self.worker_count == 0 {
            sink.deliver(tile, Err(TileError::permanent("no tile worker threads")));
            return;
        }
        self.queue.push(Job { tile, url, sink });
    }
}

/// Builds the connection-pooling [`ureq::Agent`] for XYZ tile fetches.
///
/// See the module docs for why the crypto provider is passed explicitly instead
/// of relying on rustls' process-wide default.
pub(crate) fn build_agent() -> ureq::Agent {
    build_agent_with_timeout(REQUEST_TIMEOUT, WORKER_THREADS)
}

/// Builds an agent with the shared TLS/user-agent configuration but a caller-
/// chosen wall-clock budget and pool size.
///
/// [`crate::range_http`] uses this with a longer timeout and a larger `workers`
/// (its own [`crate::range_http::RANGE_WORKER_THREADS`]): a COG source tile is
/// a multi-megabyte range read, an order of magnitude more data than a basemap
/// tile, and must not inherit either the short XYZ budget or a pool sized for
/// the smaller tile pool.
///
/// `workers` sizes the idle-connection pool to match: ureq's defaults
/// (`max_idle_connections_per_host = 3`, below either pool's worker count)
/// would otherwise close a returned connection back down to three, so up to a
/// third of six range workers (or a quarter of four tile workers) would pay a
/// fresh TCP+TLS handshake on their very next request even though every
/// worker talks to the same one or two origins — exactly the long-haul
/// open-data buckets whose per-request latency is the reason this many
/// workers exist. One idle slot per worker keeps a connection warm for each.
pub(crate) fn build_agent_with_timeout(timeout: Duration, workers: usize) -> ureq::Agent {
    let tls = ureq::tls::TlsConfig::builder()
        .provider(ureq::tls::TlsProvider::Rustls)
        .unversioned_rustls_crypto_provider(Arc::new(rustls_graviola::default_provider()))
        .build();
    ureq::Agent::config_builder()
        .user_agent(HttpTileTransport::user_agent())
        .tls_config(tls)
        .timeout_global(Some(timeout))
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .max_idle_connections(workers.saturating_mul(2))
        .max_idle_connections_per_host(workers)
        .max_idle_age(IDLE_CONNECTION_AGE)
        .build()
        .new_agent()
}

/// HTTP status meaning "you are going too fast" — a rate limit, not a refusal.
const STATUS_TOO_MANY_REQUESTS: u16 = 429;

/// HTTP status at and above which a failure is treated as transient (matches
/// `crate::range_http::FIRST_SERVER_ERROR_STATUS`).
const FIRST_SERVER_ERROR_STATUS: u16 = 500;

/// One response header's value as an owned [`String`], when present and text.
///
/// Shared with [`crate::range_http`], which reads `content-range` and `etag`
/// off the same response type this reads `retry-after` off.
pub(crate) fn header_value(
    response: &ureq::http::Response<ureq::Body>,
    name: &str,
) -> Option<String> {
    response
        .headers()
        .get(name)?
        .to_str()
        .ok()
        .map(str::to_owned)
}

/// The `scheme://host[:port]` portion of `url` — everything up to the first
/// `/` after the scheme — used to key a rate limit by origin rather than by
/// exact resource.
///
/// A byte-slice split, not a full URL parse: this crate carries no URL crate,
/// and a key that drops the path and query is exactly what's wanted here —
/// every tile or range request to the same server shares one cooldown.
pub(crate) fn origin_of(url: &str) -> &str {
    let Some(scheme_end) = url.find("://") else {
        return url;
    };
    let after = scheme_end + 3;
    match url.get(after..).and_then(|rest| rest.find('/')) {
        Some(offset) => &url[..after + offset],
        None => url,
    }
}

/// Parses a `Retry-After` header's delay-seconds form (RFC 9110 §10.2.3).
///
/// Only the integer-seconds form is parsed. The specification's other form,
/// an HTTP-date, would need calendar arithmetic this crate has no other use
/// for, and every tile/COG host exercised by this crate's live tests sends
/// delay-seconds. An unparsed value simply leaves no cooldown recorded — the
/// same as an absent header today — so nothing regresses; the caller's usual
/// retry backoff still applies.
fn parse_retry_after_secs(value: &str) -> Option<Duration> {
    let trimmed = value.trim();
    if trimmed.is_empty() || !trimmed.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(Duration::from_secs(trimmed.parse().ok()?))
}

/// How many origins' cooldowns are remembered at once.
///
/// Bounded like [`crate::range_http::MAX_PINNED_VALIDATORS`] and for the same
/// reason: a session talks to a handful of hosts, not thousands, and the bound
/// exists so a long-lived process pointed at many of them cannot grow this map
/// without limit.
const MAX_COOLDOWN_ORIGINS: usize = 64;

/// Per-origin pause requested by a `429`/`503` `Retry-After`, shared by every
/// worker in a pool so the whole pool honours one server's pause instead of
/// each worker re-discovering it for itself on its own next attempt.
///
/// Deliberately does **not** make a worker sleep out the pause: every worker
/// in a pool pulls from one shared queue ([`JobQueue`]), so a fast pan that
/// puts a dozen tiles in flight against one host that 429s all of them would
/// otherwise put every worker to sleep at once and stall the whole pool —
/// reintroducing the head-of-line blocking the shared queue exists to remove,
/// through a different door. Instead a request that lands inside the window
/// fails fast as transient without dialling, and the provider's own
/// exponential backoff (`oxigis_ui::tile_provider::FailureState::after_failure`,
/// 0.5 s doubling to an 8 s cap) paces when the next attempt is even made;
/// this only stops that attempt from *reaching* a host that has already said
/// not yet.
#[derive(Debug, Default)]
pub(crate) struct HostCooldowns {
    /// `(origin, resume no earlier than)`.
    entries: Mutex<Vec<(String, Instant)>>,
}

impl HostCooldowns {
    /// Whether a request to `origin` right now would land inside a pause it
    /// asked for.
    ///
    /// `pub(crate)`: [`crate::range_http`] shares one `HostCooldowns` type
    /// with this module (a 429 on a COG bucket and a 429 on a tile host are
    /// the same signal), so both this and [`Self::note_retry_after`] must be
    /// visible outside this module, not just outside this crate.
    pub(crate) fn is_paused(&self, origin: &str) -> bool {
        let now = Instant::now();
        self.lock()
            .iter()
            .any(|(held, until)| held == origin && *until > now)
    }

    /// Parses `retry_after` (delay-seconds form) and, if it parses, pauses
    /// `origin` until then. A no-op for an absent or unparsable header — see
    /// [`parse_retry_after_secs`] for why only that one form is understood.
    pub(crate) fn note_retry_after(&self, origin: &str, retry_after: Option<&str>) {
        if let Some(wait) = retry_after.and_then(parse_retry_after_secs) {
            self.note(origin, wait);
        }
    }

    /// Records that `origin` must not be dialled again before `wait` from now.
    ///
    /// Extends an existing pause rather than shortening one already running —
    /// a second `429` while the first pause is still in effect means the
    /// server is still not ready. An absurd `Retry-After` that would overflow
    /// [`Instant`] arithmetic is dropped rather than panicking or saturating
    /// to a pause that would never lift.
    fn note(&self, origin: &str, wait: Duration) {
        let Some(until) = Instant::now().checked_add(wait) else {
            return;
        };
        let mut entries = self.lock();
        if let Some((_, slot)) = entries.iter_mut().find(|(held, _)| held == origin) {
            if until > *slot {
                *slot = until;
            }
            return;
        }
        while entries.len() >= MAX_COOLDOWN_ORIGINS {
            entries.remove(0);
        }
        entries.push((origin.to_owned(), until));
    }

    /// The entry list, recovering the contents if a worker panicked holding
    /// it — a poisoned mutex here means some *other* request panicked; these
    /// entries are plain data and are still exactly as valid as they were.
    fn lock(&self) -> MutexGuard<'_, Vec<(String, Instant)>> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Turns a non-2xx status into the failure it means, noting `retry_after` in
/// `cooldowns` under `origin` when the status is retryable and the header
/// parses.
///
/// Takes the status and header as plain values rather than the raw `ureq`
/// response type, so this — the part with a policy worth testing — is
/// unit-testable with no live response in hand, the same reason
/// `crate::range_http::classify_status` is shaped this way.
fn classify_status(
    cooldowns: &HostCooldowns,
    origin: &str,
    status: u16,
    retry_after: Option<&str>,
) -> TileError {
    let transient =
        (FIRST_SERVER_ERROR_STATUS..600).contains(&status) || status == STATUS_TOO_MANY_REQUESTS;
    if transient {
        cooldowns.note_retry_after(origin, retry_after);
    }
    let message = format!("HTTP {status}");
    if transient {
        TileError::transient(message)
    } else {
        TileError::permanent(message)
    }
}

/// Performs one blocking `GET` and classifies the outcome.
///
/// Retryability follows the shared policy in `oxigis-ui`: HTTP 5xx, transport
/// and IO errors, and 429 (a rate limit, not a refusal — see
/// [`HostCooldowns`]) may be retried; every other 4xx (403, 404, ...) is
/// permanent for the session, unchanged from before.
///
/// Statuses are read off the response rather than raised as a `ureq` error
/// (`http_status_as_error(false)`), the same reason [`crate::range_http`]
/// disables it: a 429's `Retry-After` is a **header**, and it must be read
/// before the response is discarded.
fn fetch(agent: &ureq::Agent, cooldowns: &HostCooldowns, url: &str) -> Result<Vec<u8>, TileError> {
    let origin = origin_of(url);
    if cooldowns.is_paused(origin) {
        return Err(TileError::transient(format!(
            "{origin} asked to be left alone after a 429/503 Retry-After; not dialled yet"
        )));
    }
    let response = match agent
        .get(url)
        .config()
        .http_status_as_error(false)
        .build()
        .call()
    {
        Ok(response) => response,
        Err(error) => return Err(TileError::transient(error.to_string())),
    };
    let status = response.status().as_u16();
    if (200..300).contains(&status) {
        let mut body = response.into_body();
        return body
            .with_config()
            .limit(MAX_TILE_BYTES)
            .read_to_vec()
            .map_err(|error| TileError::transient(format!("body read failed: {error}")));
    }
    let retry_after = header_value(&response, "retry-after");
    Err(classify_status(
        cooldowns,
        origin,
        status,
        retry_after.as_deref(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        CONNECT_TIMEOUT, HostCooldowns, HttpTileTransport, IDLE_CONNECTION_AGE, REQUEST_TIMEOUT,
        STATUS_TOO_MANY_REQUESTS, WORKER_THREADS, build_agent, build_agent_with_timeout,
        classify_status, fetch, origin_of, parse_retry_after_secs,
    };
    use std::time::Duration;

    #[test]
    fn user_agent_identifies_oxigis_and_its_version() {
        let agent = HttpTileTransport::user_agent();
        assert!(agent.starts_with("OxiGIS/"));
        assert!(agent.contains(env!("CARGO_PKG_VERSION")));
        assert!(agent.contains("github.com"));
    }

    /// A regression pin: the idle-connection pool must be sized to at least
    /// the worker count, or a pool below it closes a connection right back
    /// down before the next worker's turn — the fresh-handshake-per-request
    /// bug this test pins.
    #[test]
    fn the_agent_pool_is_sized_to_the_worker_count() {
        let agent = build_agent_with_timeout(REQUEST_TIMEOUT, WORKER_THREADS);
        let config = agent.config();
        assert_eq!(config.max_idle_connections_per_host(), WORKER_THREADS);
        assert!(
            config.max_idle_connections() >= WORKER_THREADS,
            "the whole-pool cap must not undercut the per-host one"
        );
        assert_eq!(config.max_idle_age(), IDLE_CONNECTION_AGE);
        assert_eq!(config.timeouts().connect, Some(CONNECT_TIMEOUT));
        assert_eq!(config.timeouts().global, Some(REQUEST_TIMEOUT));
    }

    #[test]
    fn the_transport_starts_a_worker_pool() {
        let transport = HttpTileTransport::new().expect("worker threads must start");
        assert!(format!("{transport:?}").contains(&WORKER_THREADS.to_string()));
    }

    // -------------------------------------------------------------------
    // 429 is retryable, and `Retry-After` gates re-dialling.
    // -------------------------------------------------------------------

    #[test]
    fn origin_of_keeps_the_scheme_and_authority_and_drops_the_rest() {
        assert_eq!(
            origin_of("https://tile.openstreetmap.org/3/4/5.png?x=1"),
            "https://tile.openstreetmap.org"
        );
        assert_eq!(origin_of("https://host"), "https://host");
        assert_eq!(origin_of("https://host/"), "https://host");
        // No scheme separator at all: the whole string is returned rather
        // than panicking on the `+ 3` offset.
        assert_eq!(origin_of("not-a-url"), "not-a-url");
        assert_eq!(origin_of(""), "");
    }

    #[test]
    fn parse_retry_after_accepts_delay_seconds_and_rejects_everything_else() {
        assert_eq!(
            parse_retry_after_secs("120"),
            Some(Duration::from_secs(120))
        );
        assert_eq!(
            parse_retry_after_secs("  5  "),
            Some(Duration::from_secs(5))
        );
        assert_eq!(parse_retry_after_secs("0"), Some(Duration::from_secs(0)));
        // The HTTP-date form is a documented non-goal, not a crash.
        assert_eq!(
            parse_retry_after_secs("Wed, 21 Oct 2015 07:28:00 GMT"),
            None
        );
        assert_eq!(parse_retry_after_secs(""), None);
        assert_eq!(parse_retry_after_secs("-1"), None);
        assert_eq!(parse_retry_after_secs("1.5"), None);
        // A value that parses as a number but cannot fit `u64` must not panic.
        assert_eq!(parse_retry_after_secs("99999999999999999999999999"), None);
    }

    #[test]
    fn a_429_becomes_transient_and_pauses_its_origin() {
        let cooldowns = HostCooldowns::default();
        let origin = "https://limited.example";
        assert!(!cooldowns.is_paused(origin));

        let error = classify_status(&cooldowns, origin, STATUS_TOO_MANY_REQUESTS, Some("30"));
        assert!(error.retryable(), "{error}");
        assert!(cooldowns.is_paused(origin));
    }

    #[test]
    fn a_403_stays_permanent_and_does_not_pause_anything() {
        let cooldowns = HostCooldowns::default();
        let origin = "https://strict.example";
        let error = classify_status(&cooldowns, origin, 403, None);
        assert!(!error.retryable(), "{error}");
        assert!(!cooldowns.is_paused(origin));
    }

    #[test]
    fn a_5xx_is_transient_and_a_retry_after_on_one_pauses_its_origin_too() {
        let cooldowns = HostCooldowns::default();
        let origin = "https://flaky.example";
        let error = classify_status(&cooldowns, origin, 503, Some("2"));
        assert!(error.retryable(), "{error}");
        assert!(cooldowns.is_paused(origin));
    }

    #[test]
    fn a_429_with_no_usable_retry_after_is_still_transient_and_pauses_nothing() {
        // No header at all, and a header rustdoc's HTTP-date form would use —
        // neither may invent a wait; the provider's own backoff still paces
        // the next attempt.
        let cooldowns = HostCooldowns::default();
        for retry_after in [None, Some("Wed, 21 Oct 2015 07:28:00 GMT")] {
            let origin = "https://quiet.example";
            let error = classify_status(&cooldowns, origin, STATUS_TOO_MANY_REQUESTS, retry_after);
            assert!(error.retryable(), "{error}");
            assert!(!cooldowns.is_paused(origin), "{retry_after:?}");
        }
    }

    #[test]
    fn a_paused_origin_fails_fast_without_dialling() {
        // `fetch` itself checks the cooldown before it ever builds a request,
        // so this is provable without a live (or even reachable) host:
        // `.invalid` is the RFC 2606 TLD reserved to never resolve, so a bug
        // that let this call through to `agent.get(...).call()` would hang or
        // fail on DNS instead of returning immediately.
        let cooldowns = HostCooldowns::default();
        let origin = "http://rate-limited.invalid";
        let _ = classify_status(&cooldowns, origin, STATUS_TOO_MANY_REQUESTS, Some("60"));
        assert!(cooldowns.is_paused(origin));

        let error = fetch(
            &build_agent(),
            &cooldowns,
            "http://rate-limited.invalid/tile.png",
        )
        .expect_err("a paused origin must fail without a network round trip");
        assert!(error.retryable(), "{error}");
    }

    #[test]
    fn an_absurd_retry_after_does_not_panic_the_instant_arithmetic() {
        // `Instant::now() + Duration::MAX` would panic; the header parses
        // fine (it is all digits) but the resulting `Duration` must be
        // rejected by `checked_add`, not allowed to reach the `+`.
        let cooldowns = HostCooldowns::default();
        let origin = "https://hostile.example";
        let error = classify_status(
            &cooldowns,
            origin,
            STATUS_TOO_MANY_REQUESTS,
            Some("18446744073709551615"),
        );
        assert!(error.retryable(), "{error}");
        // The overflowing pause must not have been recorded.
        assert!(!cooldowns.is_paused(origin));
    }

    /// The recorded deadline for `origin`, reaching past `is_paused`'s bool
    /// straight at [`HostCooldowns`]'s private state — the only way to prove
    /// "extends, never shortens" without sleeping a test through real time.
    fn deadline(cooldowns: &HostCooldowns, origin: &str) -> std::time::Instant {
        *cooldowns
            .lock()
            .iter()
            .find(|(held, _)| held == origin)
            .map(|(_, until)| until)
            .expect("the pause must be recorded")
    }

    #[test]
    fn a_second_429_extends_a_running_pause_but_never_shortens_it() {
        let cooldowns = HostCooldowns::default();
        let origin = "https://limited.example";
        classify_status(&cooldowns, origin, STATUS_TOO_MANY_REQUESTS, Some("60"));
        let after_first = deadline(&cooldowns, origin);

        // A shorter follow-up `Retry-After` must not cut the first pause
        // short — the server is still saying "not yet".
        classify_status(&cooldowns, origin, STATUS_TOO_MANY_REQUESTS, Some("1"));
        assert_eq!(
            deadline(&cooldowns, origin),
            after_first,
            "a shorter pause must not shorten the deadline"
        );

        // A genuinely longer one does extend it.
        classify_status(&cooldowns, origin, STATUS_TOO_MANY_REQUESTS, Some("120"));
        assert!(
            deadline(&cooldowns, origin) > after_first,
            "a longer pause must extend the deadline"
        );
    }

    // -------------------------------------------------------------------
    // One shared queue, not one private queue per worker round-robined onto.
    // -------------------------------------------------------------------

    #[test]
    fn a_stalled_job_does_not_block_the_rest_of_the_queue() {
        use super::JobQueue;
        use std::sync::{Arc, Mutex};
        use std::time::{Duration, Instant};

        // With the old round-robin-onto-a-private-queue design, whichever
        // worker job 0 landed on would still be holding jobs 1 and 2 behind
        // it if the dispatch cursor happened to send them there too — the
        // exact bug this finding replaces. With one shared queue, the other
        // worker must pick them up regardless.
        let queue = Arc::new(JobQueue::<u32>::new());
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let release_rx = Arc::new(Mutex::new(release_rx));
        let done: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));

        let workers: Vec<_> = (0..2)
            .map(|_| {
                let queue = Arc::clone(&queue);
                let release_rx = Arc::clone(&release_rx);
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    while let Some(job) = queue.pop() {
                        if job == 0 {
                            // Blocks this one worker until the test releases
                            // it — standing in for a slow or hung host.
                            let _ = release_rx
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner)
                                .recv();
                        }
                        done.lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .push(job);
                    }
                })
            })
            .collect();

        queue.push(0);
        // Give a worker time to claim job 0 and block on it before the
        // others arrive, so this actually exercises the stalled-worker case
        // rather than racing the push.
        std::thread::sleep(Duration::from_millis(100));
        queue.push(1);
        queue.push(2);

        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let count = done
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len();
            if count >= 2 {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "jobs 1 and 2 must complete without waiting on stalled job 0"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        {
            let finished = done
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(finished.contains(&1));
            assert!(finished.contains(&2));
            assert!(!finished.contains(&0), "job 0 is still stalled");
        }

        let _ = release_tx.send(());
        queue.close();
        for worker in workers {
            let _ = worker.join();
        }
        assert!(
            done.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains(&0),
            "job 0 completes once released, rather than being lost"
        );
    }

    #[test]
    fn close_ends_pop_only_after_the_queue_drains() {
        let queue = super::JobQueue::<u32>::new();
        queue.push(1);
        queue.push(2);
        queue.close();
        // Already-queued jobs are still handed out after close...
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        // ...and only THEN does a worker's loop see the shutdown signal.
        assert_eq!(queue.pop(), None);
    }

    /// Live smoke test against the real OpenStreetMap tile service.
    ///
    /// Ignored by default so the standard suite stays offline and so we never
    /// hammer OSM from CI; run it deliberately with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    ///
    /// Exercises the entire production path: `XyzTileProvider` → this
    /// transport → ureq/rustls-graviola over TLS → `decode_tile` → the
    /// provider's ready cache, i.e. the same code the app runs.
    #[test]
    #[ignore = "network: fetches a real tile from tile.openstreetmap.org"]
    fn live_osm_tile_round_trip() {
        use oxigis_render::TileId;
        use oxigis_ui::BoxedTileProvider;
        use oxigis_ui::tile_provider::{BasemapConfig, XyzTileProvider};

        let transport = HttpTileTransport::new().expect("worker threads must start");
        let provider: BoxedTileProvider = Box::new(
            XyzTileProvider::new(
                &BasemapConfig::openstreetmap(),
                &egui::Context::default(),
                Box::new(transport),
            )
            .expect("the OSM basemap must build a provider"),
        );
        let tile = TileId::new(0, 0, 0).expect("0/0/0 is a valid tile");

        let mut decoded = None;
        for _ in 0..150 {
            if let Some(pixels) = provider.tile(tile) {
                decoded = Some(pixels);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        let pixels = decoded.expect("z0 OSM tile must arrive within 15 s");
        assert_eq!(pixels.width(), 256, "OSM serves 256 px tiles");
        assert_eq!(pixels.height(), 256);
        assert_eq!(pixels.rgba().len(), 256 * 256 * 4);
        // A real map tile is not a single flat colour.
        let first = &pixels.rgba()[0..4];
        assert!(
            pixels.rgba().chunks_exact(4).any(|px| px != first),
            "the decoded tile must contain more than one colour"
        );
    }

    /// Live smoke test against a *custom* XYZ endpoint — by default EOX's
    /// Sentinel-2 cloudless mosaic, a WMTS-REST service whose
    /// `GoogleMapsCompatible` tile matrix makes it a plain XYZ source with two
    /// twists the OSM test cannot cover: the template is `{z}/{y}/{x}` (row
    /// before column, the WMTS REST convention — absorbed by `XyzTemplate`'s
    /// named placeholders) and the tiles are JPEG rather than PNG.
    ///
    /// Override with `OXIGIS_LIVE_XYZ_URL` to point the test at any other
    /// template. Ignored by default; run with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: fetches a real tile from tiles.maps.eox.at"]
    fn live_custom_xyz_round_trip() {
        use oxigis_render::TileId;
        use oxigis_ui::BoxedTileProvider;
        use oxigis_ui::tile_provider::{BasemapConfig, XyzTileProvider};

        const URL: &str = "https://e.tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/\
                           default/GoogleMapsCompatible/{z}/{y}/{x}.jpg";
        let url = std::env::var("OXIGIS_LIVE_XYZ_URL").unwrap_or_else(|_| URL.to_owned());
        println!("live XYZ: {url}");
        let config = BasemapConfig {
            url_template: url,
            subdomains: Vec::new(),
            attribution: String::new(),
        };
        let transport = HttpTileTransport::new().expect("worker threads must start");
        let provider: BoxedTileProvider = Box::new(
            XyzTileProvider::new(&config, &egui::Context::default(), Box::new(transport))
                .expect("the template must build a provider"),
        );

        // x=2, y=1 at z2 covers Europe and the Mediterranean: any worldwide
        // imagery basemap has land/sea contrast there, so the flat-colour
        // assertion below cannot pass on an accidentally blank tile. (Which
        // path segment receives which coordinate is `XyzTemplate`'s
        // unit-tested job, not this test's.)
        let tile = TileId::new(2, 2, 1).expect("2/2/1 is a valid tile");
        let mut decoded = None;
        for _ in 0..300 {
            if let Some(pixels) = provider.tile(tile) {
                decoded = Some(pixels);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let pixels = decoded.expect("the z2 tile must arrive within 30 s");
        assert_eq!(pixels.width(), 256, "EOX serves 256 px tiles");
        assert_eq!(pixels.height(), 256);
        let first = &pixels.rgba()[0..4];
        assert!(
            pixels.rgba().chunks_exact(4).any(|px| px != first),
            "real imagery is not a single flat colour"
        );
    }

    /// Live: every built-in basemap preset serves a decodable, non-blank tile.
    ///
    /// The presets ship in the layer panel with their credit lines baked in,
    /// so each one must actually work — a dead sample is worse than no sample.
    /// Uses the same Europe/Mediterranean z2 tile as
    /// `live_custom_xyz_round_trip`: land/sea contrast there defeats the
    /// flat-colour check in every map style, satellite or cartographic.
    /// Ignored by default; run with
    /// `cargo nextest run -p oxigis-desktop --run-ignored=all`.
    #[test]
    #[ignore = "network: fetches one tile from every preset basemap service"]
    fn live_basemap_presets_round_trip() {
        use oxigis_render::TileId;
        use oxigis_ui::BoxedTileProvider;
        use oxigis_ui::tile_provider::{BASEMAP_PRESETS, XyzTileProvider};

        let tile = TileId::new(2, 2, 1).expect("2/2/1 is a valid tile");
        for preset in BASEMAP_PRESETS {
            println!("live preset '{}': {}", preset.name, preset.url_template);
            let transport = HttpTileTransport::new().expect("worker threads must start");
            let provider: BoxedTileProvider = Box::new(
                XyzTileProvider::new(
                    &preset.config(),
                    &egui::Context::default(),
                    Box::new(transport),
                )
                .unwrap_or_else(|error| {
                    panic!("preset '{}' must build a provider: {error}", preset.name)
                }),
            );
            let mut decoded = None;
            for _ in 0..300 {
                if let Some(pixels) = provider.tile(tile) {
                    decoded = Some(pixels);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            let pixels = decoded.unwrap_or_else(|| {
                panic!(
                    "preset '{}': the z2 tile must arrive within 30 s",
                    preset.name
                )
            });
            assert_eq!(pixels.width(), 256, "preset '{}' tile width", preset.name);
            assert_eq!(pixels.height(), 256, "preset '{}' tile height", preset.name);
            let first = &pixels.rgba()[0..4];
            assert!(
                pixels.rgba().chunks_exact(4).any(|px| px != first),
                "preset '{}': a real map tile is not a single flat colour",
                preset.name
            );
        }
    }

    /// Live smoke test against the keyless MapLibre demo vector tiles.
    ///
    /// Ignored by default, like the raster round trip above; run it with
    /// `cargo nextest run -p oxigis-desktop --run-ignored all \
    ///  -E 'test(live_maplibre_vector_tile_round_trip)'`.
    ///
    /// Exercises the whole production vector path: `VectorTileProvider` → this
    /// transport → ureq/rustls-graviola over TLS → gzip sniff → `decode_mvt` →
    /// the default paint table → `lyon` tessellation. The tile's layer names and
    /// feature counts are printed, since those names are what the default paint
    /// table keys on.
    #[test]
    #[ignore = "network: fetches a real vector tile from demotiles.maplibre.org"]
    fn live_maplibre_vector_tile_round_trip() {
        use oxigis_render::{LonLat, MapView, TileId};
        use oxigis_ui::{VectorTileConfig, VectorTileProvider, VectorTileSource as _};

        let config = VectorTileConfig::maplibre_demo();
        let transport = HttpTileTransport::new().expect("worker threads must start");
        let provider =
            VectorTileProvider::new(&config, &egui::Context::default(), Box::new(transport))
                .expect("the demo config must build a provider");

        let view = MapView::new(LonLat::new(0.0, 0.0), 0.0, [256.0, 256.0]).expect("a valid view");
        let _ = provider.begin_frame(view);
        let tile = TileId::new(0, 0, 0).expect("0/0/0 is a valid tile");

        let mut mesh = None;
        for _ in 0..150 {
            let _ = provider.begin_frame(view);
            if let Some(built) = provider.mesh(tile) {
                mesh = Some(built);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let mesh = mesh.expect("the z0 demo vector tile must arrive within 15 s");

        // Decode the same bytes again so the layer names can be reported: the
        // provider only hands out meshes, by design.
        let raw = fetch(
            &build_agent(),
            &HostCooldowns::default(),
            "https://demotiles.maplibre.org/tiles/0/0/0.pbf",
        )
        .expect("the demo tile must be fetchable");
        let bytes = if raw.starts_with(&oxigis_ui::vector_provider::GZIP_MAGIC) {
            oxiarc_deflate::gzip_decompress(&raw).expect("a gzip body must inflate")
        } else {
            raw
        };
        let decoded = oxigis_render::decode_mvt(&bytes).expect("the demo tile must decode");
        for layer in &decoded.layers {
            println!(
                "layer {:?}: {} features, extent {}",
                layer.name,
                layer.features.len(),
                layer.extent
            );
        }
        println!(
            "mesh: {} vertices, {} triangles",
            mesh.vertices.len(),
            mesh.triangle_count()
        );

        assert!(
            !decoded.layers.is_empty(),
            "the demo tile must carry at least one named layer"
        );
        assert!(
            decoded.layers.iter().all(|layer| !layer.name.is_empty()),
            "every layer must be named"
        );
        assert!(
            decoded
                .layers
                .iter()
                .any(|layer| !layer.features.is_empty()),
            "at least one layer must carry features"
        );
        assert!(
            !mesh.is_empty(),
            "the default paint table must tessellate the demo tile to something",
        );
    }

    /// Live smoke test of the **label** half of the vector path.
    ///
    /// Ignored by default, like its two neighbours; run it with
    /// `cargo nextest run -p oxigis-desktop --run-ignored all \
    ///  -E 'test(live_maplibre_label_placement)'`.
    ///
    /// Exercises everything §5.3 part C wires together, minus the GPU: the
    /// provider's decoded-tile seam ([`oxigis_ui::VectorTileSource::decoded`]),
    /// the demo style's symbol rules turned into a
    /// [`oxigis_render::LabelTable`], real `oxitext` shaping against the bundled
    /// Noto Sans, and the greedy collision pass. Only the atlas upload and the
    /// draw call need a device, and those the shells cover.
    ///
    /// The viewport is a synthetic 1024×1024 with the z0 tile covering all of
    /// it: at the app's real 256 px tile size every country name would overlap
    /// its neighbours and the collision pass would reject nearly all of them,
    /// which would make the test measure the wrong thing.
    #[test]
    #[ignore = "network: fetches a real vector tile from demotiles.maplibre.org"]
    fn live_maplibre_label_placement() {
        use oxigis_render::{LabelEngine, LabelPlacer, LonLat, MapView, TileId, TilePlacement};
        use oxigis_ui::{VectorTileConfig, VectorTileProvider, VectorTileSource as _};

        /// Edge of the synthetic viewport (and of the single tile in it).
        const VIEWPORT_PX: f32 = 1024.0;

        let config = VectorTileConfig::maplibre_demo();
        let labels = config.label_table();
        assert_eq!(
            labels.len(),
            2,
            "the demo style labels centroids and geolines"
        );

        let transport = HttpTileTransport::new().expect("worker threads must start");
        let provider =
            VectorTileProvider::new(&config, &egui::Context::default(), Box::new(transport))
                .expect("the demo config must build a provider");

        let view = MapView::new(LonLat::new(0.0, 0.0), 0.0, [256.0, 256.0]).expect("a valid view");
        let tile = TileId::new(0, 0, 0).expect("0/0/0 is a valid tile");
        let mut decoded = None;
        for _ in 0..150 {
            let _ = provider.begin_frame(view);
            // `mesh` is what drives the fetch loop; `decoded` only observes it.
            let _ = provider.mesh(tile);
            if let Some(vector_tile) = provider.decoded(tile) {
                decoded = Some(vector_tile);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let vector_tile = decoded.expect("the z0 demo vector tile must arrive within 15 s");

        // `ShapedLabel` keeps glyphs, not text, so the strings are reported from
        // the tile itself — the same property the label table reads.
        let names: Vec<String> = vector_tile
            .layers
            .iter()
            .filter(|layer| layer.name == "centroids")
            .flat_map(|layer| &layer.features)
            .filter_map(|feature| {
                feature
                    .properties
                    .iter()
                    .find(|(key, _)| key == oxigis_ui::DEMO_CENTROID_TEXT_FIELD)
                    .and_then(|(_, value)| oxigis_render::label_text(value))
            })
            .collect();
        println!(
            "first country names in the tile: {:?}",
            &names[..names.len().min(8)]
        );
        assert!(
            !names.is_empty(),
            "the demo tile's centroids layer must carry {} values",
            oxigis_ui::DEMO_CENTROID_TEXT_FIELD,
        );

        let mut engine = LabelEngine::new(oxifont_bundled::NOTO_SANS_REGULAR.to_vec())
            .expect("the bundled Noto Sans must parse");
        let placement = TilePlacement {
            tile,
            x: 0.0,
            y: 0.0,
            size: VIEWPORT_PX,
        };
        let mut placer = LabelPlacer::new([VIEWPORT_PX, VIEWPORT_PX]);
        placer
            .place_tile(&mut engine, &vector_tile, &placement, &labels)
            .expect("shaping the demo tile's names must succeed");
        assert!(
            !placer.is_stale(),
            "one z0 tile of country names must fit the atlas without a repack",
        );
        let considered = placer.considered();
        let placed = placer.finish();

        println!(
            "considered {considered} labels, placed {} of them in a {VIEWPORT_PX}x{VIEWPORT_PX} \
             viewport",
            placed.len(),
        );
        for label in placed.iter().take(8) {
            println!(
                "  origin ({:.0}, {:.0})  {:.0}x{:.0} px  colour {:?}",
                label.origin_px[0],
                label.origin_px[1],
                label.shaped.width_px(),
                label.shaped.height_px(),
                label.color,
            );
        }

        assert!(
            !placed.is_empty(),
            "the demo tile's country names must place at least one label",
        );
        assert!(
            considered >= placed.len(),
            "every placed label must have been considered",
        );
        for (index, left) in placed.iter().enumerate() {
            assert!(
                left.collision_box.is_inside([VIEWPORT_PX, VIEWPORT_PX]),
                "label {index} must be fully inside the viewport",
            );
            for right in &placed[index + 1..] {
                assert!(
                    !left.collision_box.intersects(&right.collision_box),
                    "accepted labels must not overlap: {:?} and {:?}",
                    left.collision_box,
                    right.collision_box,
                );
            }
        }
    }
}
