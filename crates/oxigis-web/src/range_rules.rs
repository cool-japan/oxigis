//! The verification rules a browser range read is held to: `Content-Range`
//! parsing, the answer-matches-the-question check, and the per-transport
//! validator pins that catch an archive republished underneath an open layer.
//!
//! Split out of `crate::range_fetch` on purpose. That module is
//! `#[cfg(target_arch = "wasm32")]` because it is `fetch()` glue, and gating the
//! *rules* along with the glue meant their tests were compiled by nothing: not
//! by `cargo test` on a host (the module does not exist there) and not by any
//! wasm runner either (the crate has no `wasm-bindgen-test` harness). These
//! functions are pure — bytes, headers and a thread-local — so they live here,
//! ungated, and `cargo nextest run -p oxigis-web` enforces them with no extra
//! tooling. `range_fetch` re-exports what its callers used to reach here.
//!
//! # Why the pins are per transport
//!
//! PMTiles v3 carries no in-file validator: a republished archive can serve a
//! header from one revision and a leaf from the next, and the reader would
//! parse the mix as one file. Both shells therefore pin the first answer's
//! `ETag`/length per URL and refuse a later answer that disagrees, telling the
//! user to remove and re-add the layer.
//!
//! On the desktop that advice works because `ValidatorPins` is owned by the
//! transport, and a transport is built per layer. A browser transport is a
//! stateless unit struct, so an earlier version of this pinned by URL alone in
//! a module-level `thread_local!` — shared by every transport for the lifetime
//! of the tab. Removing and re-adding the layer hit the same pin and got the
//! same permanent refusal, leaving the URL unreadable until a page reload,
//! which the message never suggested. [`TransportId`] restores the desktop
//! contract: pins are keyed by *(transport, URL)*, a rebuilt transport starts
//! clean, and the advice is literally what clears a refusal.

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use oxigis_render::ByteRange;
use oxigis_ui::TileError;

/// How many *(transport, URL)* validators are remembered at once.
///
/// The twin of `oxigis-desktop`'s `MAX_PINNED_VALIDATORS`, and the same
/// reasoning: a session opens a handful of archives, and the bound stops a
/// long-lived tab that has been pointed at many URLs — or has re-added the same
/// layer many times, each rebuild being a new [`TransportId`] — growing this
/// list without limit. Entries of a transport that is gone are simply evicted
/// oldest-first with everything else; nothing tracks liveness, because a
/// bounded list needs no bookkeeping to stay bounded.
pub const MAX_PINNED_VALIDATORS: usize = 64;

/// What the user is told when the file changed underneath an open archive.
pub const DRIFT_ADVICE: &str = "the archive changed on the server; remove and re-add the layer";

/// Source of [`TransportId`] values. Starts at 1 so `TransportId(0)` is not
/// silently produced by a `Default` somewhere.
static NEXT_TRANSPORT_ID: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    /// `(transport, url, validator)` in first-seen order; see
    /// [`observe_validator`].
    static PINNED: RefCell<Vec<(TransportId, String, Validator)>> =
        const { RefCell::new(Vec::new()) };
}

/// Identity of one range transport, so its validator pins are its own.
///
/// Cheap to copy and never reused within a session; see the [module docs][self]
/// for why a shared pin store was the wrong scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransportId(usize);

impl TransportId {
    /// Mints an identity no live transport holds.
    ///
    /// Saturating rather than wrapping: a tab that somehow minted `usize::MAX`
    /// identities settles on one shared id, which degrades to the old
    /// tab-wide pin scope instead of handing a fresh transport an *older*
    /// transport's pins.
    #[must_use]
    pub fn next() -> Self {
        let previous = NEXT_TRANSPORT_ID.fetch_add(1, Ordering::Relaxed);
        Self(previous.min(usize::MAX - 1))
    }

    /// The raw counter value, for diagnostics.
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }
}

/// One parsed `Content-Range` response header.
///
/// A line-for-line twin of `oxigis-desktop`'s: the two shells share no code (one
/// is `ureq`, the other is `fetch()`, and neither crate depends on the other),
/// so the *rule* is duplicated deliberately and both copies are pinned by their
/// own tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentRange {
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
#[must_use]
pub fn parse_content_range(value: &str) -> Option<ContentRange> {
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
/// # Errors
///
/// A [`TileError::permanent`] naming the disagreement; nothing about a lying
/// intermediary — or a body the browser decoded on the way in — gets better on a
/// retry.
pub fn verify_content_range(
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
            "the server's Content-Range names {served} bytes for {} but the body holds \
             {body_len} (a Content-Encoding the browser decoded looks like this)",
            asked.header_value()
        )));
    }
    Ok(total)
}

/// What one URL's first accepted answer exposed about the file behind it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Validator {
    /// The `ETag` the answer exposed, verbatim; [`None`] when the host does not
    /// expose one to script — the documented browser-drift gap.
    pub etag: Option<String>,
    /// The total length its `Content-Range` declared.
    pub total: Option<u64>,
}

/// Records `transport`'s first answer for `url`, or checks a later one against
/// it.
///
/// Fields are merged rather than replaced, so a host that exposes an `ETag` once
/// and not again keeps the pin that would catch the answer that really changed.
/// A different `transport` — which in this shell means a rebuilt layer — starts
/// from nothing, which is what makes [`DRIFT_ADVICE`] true.
///
/// # Errors
///
/// A permanent drift refusal when an already-pinned field disagrees.
pub fn observe_validator(
    transport: TransportId,
    url: &str,
    observed: &Validator,
) -> Result<(), TileError> {
    PINNED.with_borrow_mut(|entries| {
        if let Some((_, _, pinned)) = entries
            .iter_mut()
            .find(|(owner, pinned, _)| *owner == transport && pinned == url)
        {
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
            // A host exposing neither validator leaves drift undetectable here.
            return Ok(());
        }
        while entries.len() >= MAX_PINNED_VALIDATORS && !entries.is_empty() {
            entries.remove(0);
        }
        entries.push((transport, url.to_owned(), observed.clone()));
        Ok(())
    })
}

/// How many validator pins this thread is holding.
///
/// Exists for the bound test and for a shell that wants to report the store's
/// size; the entries themselves are private.
#[must_use]
pub fn pinned_len() -> usize {
    PINNED.with_borrow(Vec::len)
}

#[cfg(test)]
mod tests {
    use super::{
        ContentRange, DRIFT_ADVICE, MAX_PINNED_VALIDATORS, TransportId, Validator,
        observe_validator, parse_content_range, pinned_len, verify_content_range,
    };
    use oxigis_render::ByteRange;

    /// A non-empty range, the way every caller in this crate builds one.
    fn range(start: u64, end: u64) -> ByteRange {
        ByteRange::new(start, end).expect("a non-empty range")
    }

    /// A validator a host that exposes both fields would produce.
    fn validator(etag: &str, total: u64) -> Validator {
        Validator {
            etag: Some(etag.to_owned()),
            total: Some(total),
        }
    }

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
        assert_eq!(
            parse_content_range("bytes */900"),
            Some(ContentRange::Unsatisfied { total: 900 })
        );
        for nonsense in ["", "items 0-1/2", "bytes", "bytes 0-15", "bytes */*"] {
            assert_eq!(parse_content_range(nonsense), None, "{nonsense}");
        }
    }

    #[test]
    fn an_absent_content_range_is_accepted_and_a_wrong_one_is_refused() {
        assert_eq!(verify_content_range(None, range(0, 16), 16), Ok(None));
        let error = verify_content_range(Some("bytes 4096-4111/99999"), range(0, 16), 16)
            .expect_err("a different range must be refused");
        assert!(!error.retryable(), "{error}");
        assert!(error.message().contains("bytes=0-15"), "{error}");
    }

    #[test]
    fn a_decoded_body_is_refused_by_the_length_rule() {
        // The browser-only failure: the entity was gzipped, `Content-Range`
        // counts coded bytes and `fetch()` handed back the decoded ones.
        let error = verify_content_range(Some("bytes 0-99/1000"), range(0, 100), 512)
            .expect_err("a decoded body is not the requested range");
        assert!(error.message().contains("Content-Encoding"), "{error}");
    }

    #[test]
    fn an_exposed_etag_change_is_permanent_drift() {
        let transport = TransportId::next();
        let url = "https://host/a.pmtiles";
        assert_eq!(
            observe_validator(transport, url, &validator("\"abc\"", 1000)),
            Ok(())
        );
        let error = observe_validator(transport, url, &validator("\"def\"", 1000))
            .expect_err("a changed ETag is drift");
        assert!(!error.retryable(), "{error}");
        assert!(error.message().contains(DRIFT_ADVICE), "{error}");
    }

    #[test]
    fn an_exposed_length_change_is_permanent_drift() {
        let transport = TransportId::next();
        let url = "https://host/b.pmtiles";
        assert_eq!(
            observe_validator(
                transport,
                url,
                &Validator {
                    etag: None,
                    total: Some(1000),
                },
            ),
            Ok(())
        );
        let error = observe_validator(
            transport,
            url,
            &Validator {
                etag: None,
                total: Some(2000),
            },
        )
        .expect_err("a changed length is drift");
        assert!(error.message().contains("1000"), "{error}");
        assert!(error.message().contains("2000"), "{error}");
    }

    #[test]
    fn a_rebuilt_transport_starts_with_clean_pins() {
        // The regression the per-transport key exists for: "remove and re-add
        // the layer" builds a new transport, and that must clear the refusal
        // the drift message tells the user it clears.
        let url = "https://host/republished.pmtiles";
        let first = TransportId::next();
        assert_eq!(
            observe_validator(first, url, &validator("\"v1\"", 10)),
            Ok(())
        );
        assert!(
            observe_validator(first, url, &validator("\"v2\"", 20)).is_err(),
            "the same transport must still refuse the drifted answer"
        );
        let rebuilt = TransportId::next();
        assert_eq!(
            observe_validator(rebuilt, url, &validator("\"v2\"", 20)),
            Ok(()),
            "a rebuilt transport must accept the file as it is now"
        );
        assert!(
            observe_validator(rebuilt, url, &validator("\"v3\"", 30)).is_err(),
            "and must then pin what it accepted"
        );
    }

    #[test]
    fn two_live_transports_do_not_share_pins() {
        let url = "https://host/shared.pmtiles";
        let left = TransportId::next();
        let right = TransportId::next();
        assert_ne!(left, right);
        assert_eq!(observe_validator(left, url, &validator("\"l\"", 1)), Ok(()));
        assert_eq!(
            observe_validator(right, url, &validator("\"r\"", 2)),
            Ok(()),
            "one layer's pin must not decide another layer's first answer"
        );
    }

    #[test]
    fn a_host_exposing_nothing_pins_nothing_and_refuses_nothing() {
        let transport = TransportId::next();
        let before = pinned_len();
        assert_eq!(
            observe_validator(transport, "u", &Validator::default()),
            Ok(())
        );
        assert_eq!(
            observe_validator(transport, "u", &Validator::default()),
            Ok(())
        );
        assert_eq!(pinned_len(), before, "nothing exposed means nothing pinned");
    }

    #[test]
    fn the_pin_store_is_bounded() {
        let transport = TransportId::next();
        for index in 0..(MAX_PINNED_VALIDATORS * 2) {
            assert_eq!(
                observe_validator(
                    transport,
                    &format!("https://host/{index}.pmtiles"),
                    &Validator {
                        etag: Some(format!("\"{index}\"")),
                        total: None,
                    },
                ),
                Ok(())
            );
        }
        assert_eq!(pinned_len(), MAX_PINNED_VALIDATORS);
    }
}
