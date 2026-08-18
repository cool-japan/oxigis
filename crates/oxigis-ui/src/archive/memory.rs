// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`MemoryRangeTransport`]: a [`RangeTransport`] over bytes already in hand.
//!
//! Three jobs, one implementation:
//!
//! * it is the **browser's local-archive path** — a dropped `.pmtiles` arrives
//!   as bytes, never as a path, so there is no file to seek;
//! * it is the **offline test transport** for every archive test in this
//!   crate, on every target, with no network and no filesystem;
//! * it is the shape a native shell falls back to when a whole archive has
//!   already been read into memory.
//!
//! It answers **synchronously**, inside `request_range`, which is exactly what
//! the [`RangeTransport`] contract permits ("hand the work to a thread pool or
//! to the microtask queue and return immediately" describes the *asynchronous*
//! case; answering before returning is trivially not blocking). Callers must
//! therefore keep releasing their lock before calling a transport, which the
//! providers in this module already do for the COG reader's sake.

use std::sync::Arc;

use oxigis_render::ByteRange;

use crate::cog_provider::{RangeJob, RangeSink, RangeTransport};
use crate::tile_provider::TileError;

/// A [`RangeTransport`] served out of one in-memory buffer.
///
/// Cheap to construct from an [`Arc<[u8]>`] the caller already holds, so a
/// dropped archive's bytes are stored once and shared by the probe and by
/// whatever provider the probe's answer selects.
#[derive(Debug, Clone)]
pub struct MemoryRangeTransport {
    /// The whole archive.
    bytes: Arc<[u8]>,
}

impl MemoryRangeTransport {
    /// Serves ranges out of `bytes`.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Arc::from(bytes.into_boxed_slice()),
        }
    }

    /// Serves ranges out of an already-shared buffer.
    #[must_use]
    pub fn from_shared(bytes: Arc<[u8]>) -> Self {
        Self { bytes }
    }

    /// How many bytes the archive holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the buffer is empty, i.e. every read is past the end.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The slice `range` names, clamped at the end of the buffer.
    ///
    /// [`None`] when the range starts past the end — the one case an HTTP
    /// server answers with 416 rather than a short 206, and therefore the one
    /// case that is an error here too.
    fn slice(&self, range: ByteRange) -> Option<&[u8]> {
        let start = usize::try_from(range.start).ok()?;
        if start >= self.bytes.len() {
            return None;
        }
        let end = usize::try_from(range.end)
            .unwrap_or(usize::MAX)
            .min(self.bytes.len());
        self.bytes.get(start..end)
    }
}

impl RangeTransport for MemoryRangeTransport {
    fn request_range(&self, _url: String, range: ByteRange, job: RangeJob, sink: RangeSink) {
        // Clamped at the end, exactly as the documented HTTP contract allows:
        // the speculative 16 KiB prefetch runs past the end of a 282-byte
        // archive, and a short answer to it is normal rather than a failure.
        match self.slice(range) {
            Some(bytes) => sink.deliver(job, Ok(bytes.to_vec())),
            None => sink.deliver(
                job,
                Err(TileError::permanent(format!(
                    "byte {} is past the end of the {}-byte archive",
                    range.start,
                    self.bytes.len()
                ))),
            ),
        }
    }
}
