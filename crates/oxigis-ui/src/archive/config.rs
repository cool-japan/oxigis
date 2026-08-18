// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! [`ArchiveLayerConfig`]: what a tile-archive layer needs to be drawn.
//!
//! The archive twin of [`crate::cog_provider::CogLayerConfig`], and derived the
//! same way: from the **project**, freshly, on every frame that reconciles (see
//! `app/providers.rs`). It is deliberately small — a reference, a format and a
//! credit line — because everything else about an archive is in the archive's
//! own header and is re-read when the provider opens it.

use oxigis_core::{ArchiveFormat, ArchiveRef, archive_refusal};

/// A tile-archive layer: where the archive is and what container it is in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveLayerConfig {
    /// Where the archive is.
    pub archive: ArchiveRef,
    /// Which container format it is.
    pub format: ArchiveFormat,
    /// Credit line the archive's own metadata asked for; empty when it asked
    /// for none.
    pub attribution: String,
}

impl ArchiveLayerConfig {
    /// A layer reading `archive` as `format`, with no credit line.
    #[must_use]
    pub const fn new(archive: ArchiveRef, format: ArchiveFormat) -> Self {
        Self {
            archive,
            format,
            attribution: String::new(),
        }
    }

    /// Sets the credit line.
    #[must_use]
    pub fn with_attribution(mut self, attribution: impl Into<String>) -> Self {
        self.attribution = attribution.into();
        self
    }

    /// The string a range transport addresses — a URL or a path.
    #[must_use]
    pub fn location(&self) -> &str {
        self.archive.location()
    }

    /// Whether the archive lives on the local filesystem.
    ///
    /// The one question a shell asks to pick a transport: a path needs a file
    /// reader (native only), a URL needs the HTTP one.
    #[must_use]
    pub const fn is_local(&self) -> bool {
        matches!(self.archive, ArchiveRef::Path { .. })
    }

    /// Why this layer cannot be read, when it cannot.
    ///
    /// The same [`oxigis_core::archive_refusal`] rule the add seam consults, so
    /// a combination refused at add time cannot slip in through a hand-edited
    /// project file and then fail per tile instead.
    #[must_use]
    pub fn refusal(&self) -> Option<String> {
        archive_refusal(&self.archive, self.format)
    }

    /// A human-readable layer name: the archive's file name.
    #[must_use]
    pub fn display_name(&self) -> String {
        self.archive.file_name().to_owned()
    }
}
