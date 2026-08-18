//! Provider-reconciliation tests (editing v1.3): the basemap/COG/MVT seam
//! family, rewritten from the old take-once `pending_*` tests to the
//! offer-until-settled contract of `app/providers.rs`. Every rewrite keeps
//! the old test's intent — a swap is offered exactly once per change /
//! re-picking offers nothing / a load offers the saved config / a damaged
//! one offers nothing — and becomes strictly stronger: the old one-shot
//! could not express "the work is still outstanding", which was exactly the
//! defect (a frame without a render state consumed and dropped it).

use crate::layer_panel::LayerAction;
use crate::tile_provider::BasemapConfig;
use crate::vector_provider::VectorTileConfig;
use oxigis_core::Project;

use super::providers::{
    MAX_DRAWN_TILE_LAYERS, TileLayerPlan, TileLayerSource, TileStackWork, VectorWork,
};
use super::*;

/// The attribution the MapLibre demo source declares.
fn oxigis_ui_maplibre_attribution() -> &'static str {
    crate::vector_provider::MAPLIBRE_ATTRIBUTION
}

/// Confirms whatever raster/vector work is outstanding, as a shell with a
/// live render state would — the baseline every "offers no work" assertion
/// needs, because a fresh app has confirmed nothing yet.
fn settle_everything(app: &mut OxigisApp) {
    if let Some(work) = app.pending_raster_work() {
        app.settle_raster_work(work, Ok(()));
    }
    if let Some(work) = app.pending_vector_work() {
        app.settle_vector_work(work, Ok(()));
    }
    assert!(app.pending_raster_work().is_none());
    assert!(app.pending_vector_work().is_none());
}

#[test]
fn set_xyz_basemap_offers_a_raster_plan_and_credits_the_host() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::SetXyzBasemap(
        "https://tiles.example.org/wmts/{z}/{y}/{x}.jpg".to_string(),
    ));
    assert_eq!(
        app.basemap().url_template,
        "https://tiles.example.org/wmts/{z}/{y}/{x}.jpg"
    );
    assert_eq!(app.basemap().attribution, "© tiles.example.org");
    let work = app.pending_raster_work().expect("a swap must be offered");
    assert_eq!(&work.basemap, app.basemap());
    assert!(work.cog.is_none());
    // Strictly stronger than the old take-once seam: the work STAYS offered
    // until a shell settles it, so a frame without a render state defers the
    // install instead of losing it.
    assert_eq!(app.pending_raster_work(), Some(work.clone()));
    app.settle_raster_work(work, Ok(()));
    assert!(
        app.pending_raster_work().is_none(),
        "a settled plan is not offered again"
    );
}

#[test]
fn a_hostless_xyz_basemap_hides_the_credit_instead_of_crediting_no_one() {
    // Template validation checks placeholders, not URL shape, so a relative
    // `/tiles/{z}/{x}/{y}.png` is accepted — and its first `/`-split segment
    // is empty. The credit line must disappear (empty attribution hides the
    // overlay), not render as a dangling "© ".
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::SetXyzBasemap(
        "/tiles/{z}/{x}/{y}.png".to_string(),
    ));
    assert_eq!(app.basemap().url_template, "/tiles/{z}/{x}/{y}.png");
    assert_eq!(app.basemap().attribution, "");
}

#[test]
fn a_bad_or_subdomain_xyz_basemap_is_rejected_with_the_basemap_untouched() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    let before = app.basemap().clone();
    app.apply_layer_action(LayerAction::SetXyzBasemap(
        "https://example.org/no/placeholders.png".to_string(),
    ));
    assert_eq!(app.basemap(), &before);
    // `{s}` with no configured subdomains would pass a URL-shape check but
    // fail the expansion of every tile: it must be refused up front, not
    // installed behind a success status.
    app.apply_layer_action(LayerAction::SetXyzBasemap(
        "https://{s}.example.org/{z}/{x}/{y}.png".to_string(),
    ));
    assert_eq!(app.basemap(), &before);
    assert!(
        app.pending_raster_work().is_none(),
        "a refused basemap never reaches the plan"
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.starts_with("Basemap not changed")),
        "the user must be told the basemap was refused"
    );
}

#[test]
fn a_basemap_preset_installs_its_required_credit() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    let preset = crate::tile_provider::BASEMAP_PRESETS
        .iter()
        .find(|preset| preset.name.contains("2024"))
        .expect("the 2024 mosaic preset must exist");
    app.apply_layer_action(LayerAction::SetBasemapPreset(preset.config()));
    assert_eq!(app.basemap(), &preset.config());
    assert!(preset.matches(app.basemap()));
    let work = app.pending_raster_work().expect("a swap must be offered");
    assert_eq!(work.basemap, preset.config());
}

#[test]
fn re_applying_the_active_basemap_offers_no_work() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    // The first preset IS the default basemap, so this is exactly the
    // "click the already-highlighted entry" gesture: a fresh provider
    // would blank the map and re-fetch every visible tile for nothing.
    let preset = crate::tile_provider::BASEMAP_PRESETS[0].config();
    app.apply_layer_action(LayerAction::SetBasemapPreset(preset));
    assert!(app.pending_raster_work().is_none());
}

#[test]
fn a_custom_basemap_survives_the_project_save_load_round_trip() {
    let mut saver = OxigisApp::new();
    saver.apply_layer_action(LayerAction::SetXyzBasemap(
        "https://tiles.example.org/wmts/{z}/{y}/{x}.jpg".to_string(),
    ));
    let custom = saver.basemap().clone();
    // `apply_basemap` stamps the project immediately, so even a shell that
    // serializes `project()` without calling `sync_project_view` first
    // captures the active basemap.
    assert!(saver.project().basemap.is_some());
    saver.sync_project_view();
    let json = saver.project().to_json_string().expect("serialize");

    let mut loader = OxigisApp::new();
    settle_everything(&mut loader);
    let project = Project::from_json_string(&json).expect("deserialize");
    loader.load_project(project);
    assert_eq!(loader.basemap(), &custom);
    let work = loader
        .pending_raster_work()
        .expect("loading a project with a different basemap must offer the swap");
    assert_eq!(work.basemap, custom);
    loader.settle_raster_work(work, Ok(()));
    assert!(loader.pending_raster_work().is_none());
}

#[test]
fn an_old_project_without_a_basemap_leaves_the_active_one_alone() {
    // Files written before `Project::basemap` existed say nothing about the
    // basemap; loading one must behave exactly like those builds did —
    // leave whatever the user has active untouched, offer nothing.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::SetXyzBasemap(
        "https://tiles.example.org/{z}/{x}/{y}.png".to_string(),
    ));
    let custom = app.basemap().clone();
    settle_everything(&mut app);

    let json = Project::new("pre-basemap file")
        .to_json_string()
        .expect("serialize");
    assert!(!json.contains("\"basemap\""));
    app.load_project(Project::from_json_string(&json).expect("deserialize"));
    assert_eq!(app.basemap(), &custom);
    assert!(app.pending_raster_work().is_none());
}

#[test]
fn a_saved_basemap_matching_the_active_one_offers_no_work() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    let mut project = Project::new("osm project");
    project.basemap = Some((&BasemapConfig::default()).into());
    app.load_project(project);
    assert!(
        app.pending_raster_work().is_none(),
        "re-installing the already-active basemap would blank the map and \
         re-fetch every visible tile for nothing — the reconciliation mirror \
         survives the load precisely so equal plans mean no work"
    );
}

#[test]
fn an_unusable_saved_basemap_is_reported_and_the_active_one_kept() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    let before = app.basemap().clone();
    let mut project = Project::new("damaged basemap");
    project.basemap = Some(oxigis_core::ProjectBasemap {
        url_template: "https://{s}.example.org/{z}/{x}/{y}.png".to_string(),
        subdomains: Vec::new(),
        attribution: String::new(),
    });
    app.load_project(project);
    assert_eq!(app.basemap(), &before);
    assert!(app.pending_raster_work().is_none());
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("unusable")),
        "a damaged saved basemap must surface as a load notice, \
         got {:?}",
        app.status
    );
}

#[test]
fn file_new_resets_a_custom_basemap_to_the_default() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::SetXyzBasemap(
        "https://tiles.example.org/{z}/{x}/{y}.png".to_string(),
    ));
    settle_everything(&mut app);
    app.new_project();
    assert_eq!(app.basemap(), &BasemapConfig::default());
    let work = app
        .pending_raster_work()
        .expect("resetting to the default basemap must offer the swap");
    assert_eq!(work.basemap, BasemapConfig::default());
    app.settle_raster_work(work, Ok(()));
    // Already on the default: File ▸ New must not force a re-fetch.
    app.new_project();
    assert!(app.pending_raster_work().is_none());
}

#[test]
fn the_top_most_cog_layer_wins_the_raster_plan() {
    let mut app = OxigisApp::new();
    assert!(app.desired_raster().cog.is_none());
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/older.tif".to_string(),
    ));
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/newest.tif".to_string(),
    ));
    // The newest COG is what the composite provider draws, so that is what
    // the plan must carry — and the offered work must agree.
    let cog = app
        .desired_raster()
        .cog
        .expect("an added COG layer must reach the raster plan");
    assert_eq!(cog.url, "https://example.org/newest.tif");
    let work = app.pending_raster_work().expect("the plan is outstanding");
    assert_eq!(
        work.cog.as_ref().map(|cog| cog.url.as_str()),
        Some("https://example.org/newest.tif")
    );
}

#[test]
fn adding_a_vector_layer_offers_an_install_and_credits_maplibre() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    let url = app.vector_url_input().to_string();
    assert_eq!(url, VectorTileConfig::maplibre_demo().url_template);
    assert!(app.vector_attribution().is_empty());
    app.apply_layer_action(LayerAction::AddVectorTileLayer(url.clone()));
    assert_eq!(app.project().layers.len(), 1);
    assert!(app.selection().is_some());
    assert_eq!(app.vector_attribution(), oxigis_ui_maplibre_attribution());
    let Some(VectorWork::Install(config)) = app.pending_vector_work() else {
        panic!("the shell must be offered an install");
    };
    assert_eq!(config.url_template, url);
    assert_eq!(config.paints.len(), 5);
    assert_eq!(config.label_table().len(), 2);
    // Offered until settled — the strictly stronger contract.
    assert_eq!(
        app.pending_vector_work(),
        Some(VectorWork::Install(config.clone()))
    );
    app.settle_vector_work(VectorWork::Install(config), Ok(()));
    assert!(
        app.pending_vector_work().is_none(),
        "a settled install is not offered again"
    );
}

#[test]
fn a_custom_vector_url_carries_no_attribution() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        "https://example.test/{z}/{x}/{y}.pbf".to_string(),
    ));
    assert!(app.vector_attribution().is_empty());
    let Some(VectorWork::Install(config)) = app.pending_vector_work() else {
        panic!("the shell must be offered an install");
    };
    assert_eq!(config.url_template, "https://example.test/{z}/{x}/{y}.pbf");
    assert!(config.attribution.is_empty());
}

// --- New in v1.3: the capabilities the one-shot seams could not express ---

#[test]
fn a_loaded_project_installs_its_cog_and_vector_sources() {
    // THE load bug the reconciliation closes: the old seams were written
    // only by the add arms, so a saved project's COG/MVT layers drew
    // nothing until the user re-added them.
    let mut saver = OxigisApp::new();
    saver.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/scene.tif".to_string(),
    ));
    saver.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    saver.sync_project_view();
    let json = saver.project().to_json_string().expect("serialize");

    let mut loader = OxigisApp::new();
    settle_everything(&mut loader);
    loader.load_project(Project::from_json_string(&json).expect("deserialize"));
    let raster = loader
        .pending_raster_work()
        .expect("the saved COG must be offered");
    assert_eq!(
        raster.cog.map(|cog| cog.url),
        Some("https://example.org/scene.tif".to_string())
    );
    let Some(VectorWork::Install(config)) = loader.pending_vector_work() else {
        panic!("the saved vector layer must be offered");
    };
    assert_eq!(
        config.url_template,
        VectorTileConfig::maplibre_demo().url_template
    );
    assert_eq!(
        config.attribution,
        oxigis_ui_maplibre_attribution(),
        "a LOADED demo layer credits MapLibre exactly as a fresh add does"
    );
}

#[test]
fn removing_the_vector_layer_asks_for_a_detach_and_file_new_does_too() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    settle_everything(&mut app);
    let id = app.selection().expect("the added layer is selected");
    app.apply_layer_action(LayerAction::Remove(id));
    assert_eq!(
        app.pending_vector_work(),
        Some(VectorWork::Detach),
        "removing the layer detaches its source — the map matches the list"
    );
    app.settle_vector_work(VectorWork::Detach, Ok(()));
    assert!(app.pending_vector_work().is_none());

    // File ▸ New with a source still installed detaches it the same way.
    let mut fresh = OxigisApp::new();
    fresh.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    settle_everything(&mut fresh);
    fresh.new_project();
    assert_eq!(fresh.pending_vector_work(), Some(VectorWork::Detach));
}

#[test]
fn removing_the_drawn_cog_asks_for_the_bare_basemap_again() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/scene.tif".to_string(),
    ));
    settle_everything(&mut app);
    let id = app.selection().expect("the added layer is selected");
    app.apply_layer_action(LayerAction::Remove(id));
    let work = app
        .pending_raster_work()
        .expect("the removal must offer the bare-basemap rebuild");
    assert!(work.cog.is_none(), "the COG is gone from the plan");
    assert_eq!(&work.basemap, app.basemap());
}

#[test]
fn a_refused_install_is_not_offered_again_until_the_project_changes() {
    let mut app = OxigisApp::new();
    let work = app.pending_raster_work().expect("the first plan");
    app.settle_raster_work(work, Err("no worker pool".to_string()));
    assert!(
        app.pending_raster_work().is_none(),
        "a refusal is memoized — no per-frame rebuild spin"
    );
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("was not installed")),
        "the refusal is loud: {:?}",
        app.status
    );
    // Any project change that implies a NEW plan re-offers.
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/scene.tif".to_string(),
    ));
    assert!(
        app.pending_raster_work().is_some(),
        "a different plan is offered despite the memoized refusal"
    );
}

#[test]
fn a_provider_layers_attribution_dies_with_the_layer() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    assert_eq!(app.vector_attribution(), oxigis_ui_maplibre_attribution());
    let id = app.selection().expect("selected");
    app.apply_layer_action(LayerAction::Remove(id));
    assert!(
        app.vector_attribution().is_empty(),
        "the credit derives from the project, so it cannot outlive its layer"
    );
}

#[test]
fn the_print_snapshot_matches_the_derivations() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/scene.tif".to_string(),
    ));
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    let cog_id = app
        .project()
        .layers
        .layers()
        .iter()
        .find(|layer| matches!(&layer.kind, oxigis_core::LayerKind::Raster(_)))
        .map(|layer| layer.id)
        .expect("the COG layer exists");
    app.apply_layer_action(LayerAction::Remove(cog_id));
    app.request_print();
    let request = app.take_pending_print().expect("an export must be queued");
    assert!(request.cog.is_none(), "the removed COG does not print");
    assert!(
        request.vector.is_some(),
        "the surviving vector layer prints"
    );
    assert!(
        request
            .attribution
            .contains(oxigis_ui_maplibre_attribution()),
        "page and screen read the same credit builder"
    );
}

// --- Editing v1.3 E4: provider adds are recorded, symmetrically ---

#[test]
fn a_cog_add_and_its_undo_are_symmetric() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/scene.tif".to_string(),
    ));
    assert_eq!(app.undo.depth(), (1, 0), "the add is ONE undo step");
    assert_eq!(
        app.undo
            .peek_undo_entry()
            .map(crate::edit::stack::UndoEntry::label),
        Some("Add layer")
    );
    if let Some(work) = app.pending_raster_work() {
        app.settle_raster_work(work, Ok(()));
    }

    assert!(app.undo_once());
    assert!(
        app.project().layers.is_empty(),
        "the undo removes the layer entry"
    );
    let work = app
        .pending_raster_work()
        .expect("the undo un-draws the COG — the bare basemap is offered");
    assert!(work.cog.is_none());
    app.settle_raster_work(work, Ok(()));

    assert!(app.redo_once());
    assert_eq!(app.project().layers.len(), 1, "the redo puts it back");
    let work = app
        .pending_raster_work()
        .expect("and offers the COG composite again");
    assert_eq!(
        work.cog.map(|cog| cog.url),
        Some("https://example.org/scene.tif".to_string())
    );
}

#[test]
fn an_mvt_add_undo_produces_a_detach() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    assert_eq!(app.undo.depth(), (1, 0));
    if let Some(work) = app.pending_vector_work() {
        app.settle_vector_work(work, Ok(()));
    }

    assert!(app.undo_once());
    assert_eq!(
        app.pending_vector_work(),
        Some(VectorWork::Detach),
        "undoing the add detaches the drawn source"
    );
    app.settle_vector_work(VectorWork::Detach, Ok(()));

    assert!(app.redo_once());
    assert!(
        matches!(app.pending_vector_work(), Some(VectorWork::Install(_))),
        "the redo re-installs it"
    );
}

#[test]
fn an_xyz_demo_layer_is_recorded_and_needs_no_provider() {
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    app.apply_layer_action(LayerAction::AddDemoXyzLayer);
    assert_eq!(app.undo.depth(), (1, 0), "the add is undoable");
    assert!(
        app.status.as_deref().is_some_and(|status| {
            status.contains("draws it as the basemap") && status.contains("Ctrl+Z removes it.")
        }),
        "the v1.3 \u{201c}a list entry only\u{201d} sentence is retired: an \
         XYZ layer now has a promotion toggle, and the status names it: {:?}",
        app.status,
    );
    assert!(
        app.pending_raster_work().is_none() && app.pending_vector_work().is_none(),
        "an UNPROMOTED XYZ stack entry still derives no provider of its own"
    );
    assert!(app.undo_once());
    assert!(app.project().layers.is_empty());
    assert!(
        app.pending_raster_work().is_none() && app.pending_vector_work().is_none(),
        "and the undo needs none either — symmetric by construction"
    );
}

#[test]
fn removing_a_provider_layer_says_what_actually_happens() {
    // A drawn MVT layer: the map stops drawing it.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    let id = app.selection().expect("selected");
    app.apply_layer_action(LayerAction::Remove(id));
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("the map stops drawing it")),
        "a drawn provider layer's removal says so: {:?}",
        app.status,
    );

    // A layer that genuinely draws nothing promises only what a Ctrl+Z
    // actually restores. An XYZ template needing a `{s}` host list is the
    // case: a layer records no subdomains, so it resolves for neither the
    // basemap nor the stack. (A *plain* XYZ layer no longer qualifies — since
    // compositing v1.6 an unpromoted one is an ordinary stack entry and really
    // does stop drawing when it is removed.)
    let mut app = OxigisApp::new();
    let id = app.project.layers.add(oxigis_core::Layer::new(
        "Needs a host list",
        oxigis_core::LayerKind::Raster(oxigis_core::RasterSource::Xyz {
            url_template: "https://{s}.example.test/{z}/{x}/{y}.png".to_string(),
            attribution: String::new(),
        }),
    ));
    assert!(
        !app.desired_tile_stack().draws(id),
        "the fixture must genuinely draw nothing, or this asserts the wrong branch"
    );
    app.apply_layer_action(LayerAction::Remove(id));
    assert!(
        app.status
            .as_deref()
            .is_some_and(|status| status.contains("with its style and its position.")),
        "an undrawn layer's removal promises no features and no un-drawing: {:?}",
        app.status,
    );
}

// --- Editing v1.3 E5: visibility is consulted by the derivations ---

#[test]
fn the_top_most_visible_cog_wins_and_the_checkbox_finally_means_it() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/older.tif".to_string(),
    ));
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/newest.tif".to_string(),
    ));
    let newest = app.selection().expect("the newest COG is selected");
    settle_everything(&mut app);

    // Hiding the drawn COG falls back to the one below it.
    app.apply_layer_action(LayerAction::ToggleVisibility(newest));
    let work = app
        .pending_raster_work()
        .expect("hiding the drawn COG changes the plan");
    assert_eq!(
        work.cog.as_ref().map(|cog| cog.url.as_str()),
        Some("https://example.org/older.tif"),
        "the top-most VISIBLE COG wins"
    );
    app.settle_raster_work(work, Ok(()));

    // Showing it again puts it back on top.
    app.apply_layer_action(LayerAction::ToggleVisibility(newest));
    let work = app.pending_raster_work().expect("re-showing replans");
    assert_eq!(
        work.cog.map(|cog| cog.url),
        Some("https://example.org/newest.tif".to_string())
    );
}

// --- Editing v1.4 E1: a refused install is visible, and retryable ---

#[test]
fn a_refusal_is_visible_while_it_suppresses_the_plan() {
    let mut app = OxigisApp::new();
    let work = app.pending_raster_work().expect("the first plan");
    app.settle_raster_work(work, Err("no worker pool".to_string()));
    assert!(
        app.pending_raster_work().is_none(),
        "the memo still suppresses the rebuild spin"
    );
    assert_eq!(
        app.raster_refusal(),
        Some("no worker pool"),
        "and the very plan it suppresses is what the banner names"
    );
    assert_eq!(app.provider_refusal().as_deref(), Some("no worker pool"));
}

#[test]
fn a_stale_refusal_reports_nothing() {
    let mut app = OxigisApp::new();
    let work = app.pending_raster_work().expect("the first plan");
    app.settle_raster_work(work, Err("no worker pool".to_string()));
    // The project moves the desire past the refused plan: the memo can no
    // longer describe what the map is doing, so it must not badge it.
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/scene.tif".to_string(),
    ));
    assert!(app.pending_raster_work().is_some());
    assert!(app.raster_refusal().is_none());
    assert!(app.provider_refusal().is_none());
    assert!(
        !app.retry_refused_installs(),
        "a Retry that cannot misfire: nothing visible was cleared"
    );
}

#[test]
fn retry_re_offers_the_refused_plan_exactly_once() {
    let mut app = OxigisApp::new();
    let work = app.pending_raster_work().expect("the first plan");
    app.settle_raster_work(work.clone(), Err("no worker pool".to_string()));
    assert!(app.retry_refused_installs());
    assert_eq!(
        app.pending_raster_work(),
        Some(work.clone()),
        "the refused plan is outstanding again"
    );
    assert!(
        app.provider_refusal().is_none(),
        "and the banner is gone the moment it is retried"
    );
    // Click spam: the second click has nothing left to clear. Cost is bounded
    // by the human click rate — no clock is consulted anywhere.
    assert!(!app.retry_refused_installs());
    app.settle_raster_work(work, Ok(()));
    assert!(app.pending_raster_work().is_none());
}

#[test]
fn retry_preserves_the_installed_mirror() {
    let mut app = OxigisApp::new();
    let bare = app.pending_raster_work().expect("the first plan");
    app.settle_raster_work(bare.clone(), Ok(()));
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.org/scene.tif".to_string(),
    ));
    let composite = app.pending_raster_work().expect("the COG plan");
    app.settle_raster_work(composite, Err("out of memory".to_string()));

    app.retry_refused_installs();
    assert_eq!(
        app.raster_installed.as_ref(),
        Some(&bare),
        "clearing the installed mirror would blank and re-fetch every visible \
         tile — a retry only forgets the refusal"
    );
}

#[test]
fn retry_clears_both_refusals_with_one_click() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    let raster = app.pending_raster_work().expect("the raster plan");
    app.settle_raster_work(raster, Err("no worker pool".to_string()));
    let vector = app.pending_vector_work().expect("the vector plan");
    app.settle_vector_work(vector, Err("no decoder".to_string()));

    assert_eq!(
        app.provider_refusal().as_deref(),
        Some("no worker pool \u{b7} no decoder"),
        "one banner, both slots, joined by the credit-line separator"
    );
    assert!(app.retry_refused_installs());
    assert!(app.provider_refusal().is_none());
    assert!(app.pending_raster_work().is_some());
    assert!(app.pending_vector_work().is_some());
}

#[test]
fn retry_touches_no_undo_history() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddDemoXyzLayer);
    let id = app.selection().expect("the added layer is selected");
    app.apply_layer_action(LayerAction::SetOpacity(id, 0.5));
    let depth = app.undo.depth();
    assert_eq!(depth, (2, 0), "the add and the opacity change");

    let work = app.pending_raster_work().expect("a plan is outstanding");
    app.settle_raster_work(work, Err("no worker pool".to_string()));
    app.apply_layer_action(LayerAction::RetryRefusedInstalls);
    assert_eq!(
        app.undo.depth(),
        depth,
        "a retry is a command, not an edit: it records nothing and evicts \
         nothing"
    );
    // The coalescing window is left open too, so an opacity drag interrupted
    // by a Retry click still folds into ONE undo step.
    app.apply_layer_action(LayerAction::SetOpacity(id, 0.25));
    assert_eq!(app.undo.depth(), depth);
    assert!(app.undo_once());
    assert_eq!(
        app.project()
            .layers
            .get(id)
            .map(oxigis_core::Layer::opacity),
        Some(1.0),
        "one Ctrl+Z undoes the whole drag, retry click and all"
    );
}

#[test]
fn attaching_the_gpu_map_clears_a_stale_refusal() {
    // Both shells settle `Err("the GPU map is not attached")` when
    // `replace_provider` finds no map, and the desire is unchanged
    // afterwards — so without a clear at the attach seam that memo would
    // suppress the SAME plan for ever. A `wgpu::RenderState` cannot be built
    // headlessly, so this pins the seam `attach_gpu_map_with` calls.
    let mut app = OxigisApp::new();
    let work = app
        .pending_raster_work()
        .expect("the first plan a shell sees");
    app.settle_raster_work(work.clone(), Err("the GPU map is not attached".to_string()));
    assert!(app.pending_raster_work().is_none());

    // No render state means no attach, and therefore no clear.
    app.attach_gpu_map(None);
    assert!(app.pending_raster_work().is_none());

    app.clear_refused_installs();
    assert_eq!(
        app.pending_raster_work(),
        Some(work),
        "once the map IS attached the refusal is provably stale"
    );
}

#[test]
fn nothing_to_retry_says_so() {
    let mut app = OxigisApp::new();
    let work = app.pending_raster_work().expect("the first plan");
    app.settle_raster_work(work, Ok(()));
    app.apply_layer_action(LayerAction::RetryRefusedInstalls);
    assert_eq!(app.status.as_deref(), Some("Nothing to retry."));
}

#[test]
fn a_deferred_install_has_nothing_to_retry() {
    // A plan outstanding because no frame has had a render state yet is not
    // a refusal: there is no banner and nothing for Retry to clear.
    let mut app = OxigisApp::new();
    assert!(app.pending_raster_work().is_some());
    assert!(app.provider_refusal().is_none());
    app.apply_layer_action(LayerAction::RetryRefusedInstalls);
    assert_eq!(app.status.as_deref(), Some("Nothing to retry."));
    assert!(
        app.pending_raster_work().is_some(),
        "and the deferred plan is still outstanding"
    );
}

#[test]
fn a_load_reproducing_the_refused_plan_stays_refused() {
    let mut app = OxigisApp::new();
    let work = app.pending_raster_work().expect("the first plan");
    app.settle_raster_work(work.clone(), Err("no worker pool".to_string()));

    // The memo is GPU state and survives a load, exactly as the installed
    // mirror does — so a project that reproduces the refused plan shows the
    // banner immediately instead of silently drawing nothing.
    let mut project = Project::new("same basemap");
    project.basemap = Some((&BasemapConfig::default()).into());
    app.load_project(project);
    assert_eq!(app.raster_refusal(), Some("no worker pool"));
    assert!(app.pending_raster_work().is_none());

    app.apply_layer_action(LayerAction::RetryRefusedInstalls);
    assert_eq!(app.pending_raster_work(), Some(work));
}

#[test]
fn hiding_the_only_vector_layer_detaches_its_source() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    let id = app.selection().expect("selected");
    settle_everything(&mut app);

    app.apply_layer_action(LayerAction::ToggleVisibility(id));
    assert_eq!(
        app.pending_vector_work(),
        Some(VectorWork::Detach),
        "an invisible layer is not drawn — the checkbox means what it says"
    );
    app.settle_vector_work(VectorWork::Detach, Ok(()));

    app.apply_layer_action(LayerAction::ToggleVisibility(id));
    assert!(
        matches!(app.pending_vector_work(), Some(VectorWork::Install(_))),
        "showing it again re-installs it"
    );
    // Visibility is RECORDED now (project ops v1.7): the checkbox goes through
    // `toggle_layer_visibility`, so Ctrl+Z genuinely re-shows the layer. What
    // stays derived is what the *map* draws — the detach above is still a
    // consequence of the flag, never a second recorded step.
    assert_eq!(
        app.undo.depth(),
        (3, 0),
        "the add plus one entry per toggle, and nothing extra for the detach"
    );
}

#[test]
fn a_failed_gpu_map_attach_is_visible_and_retryable_like_any_other_refusal() {
    // The latch is normally written by a real `map_gpu::install` failure, which
    // needs a live `wgpu` device; what this pins is everything downstream of
    // it — that the reason reaches the banner, that Retry clears the latch so
    // the next frame re-attempts the install, and that the fallback painter is
    // told which of its two situations this is.
    let mut app = OxigisApp::new();
    settle_everything(&mut app);
    assert!(app.provider_refusal().is_none());
    assert!(app.map_attach_refusal().is_none());

    app.map_gpu_failed = Some("the surface format is unsupported".to_string());
    assert_eq!(
        app.map_attach_refusal(),
        Some("the surface format is unsupported")
    );
    let Some(banner) = app.provider_refusal() else {
        panic!("an attach failure must reach the banner");
    };
    assert!(banner.contains("the surface format is unsupported"));

    // The latch also suppresses the retry of the install itself, so Retry has
    // to clear it or the banner would carry a button that changes nothing.
    assert!(app.retry_refused_installs());
    assert!(app.map_attach_refusal().is_none());
    assert!(app.provider_refusal().is_none());
}

#[test]
fn the_credit_line_agrees_with_the_configs_it_summarises() {
    // `credit_line` derives from borrows rather than from `desired_raster` /
    // `desired_vector`, which is what keeps a per-frame paint off the
    // config-building path — so the two derivations are pinned to each other
    // here, over every layer kind that contributes a credit.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/scene.tif".to_string(),
    ));
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    let expect_raster = {
        let work = app.desired_raster();
        work.cog
            .map(|cog| cog.attribution)
            .or_else(|| work.archive.map(|archive| archive.attribution))
            .unwrap_or_default()
    };
    let expect_vector = app
        .desired_vector()
        .map(|config| config.attribution)
        .unwrap_or_default();
    assert_eq!(app.cog_attribution(), expect_raster);
    assert_eq!(app.vector_attribution(), expect_vector);
    assert_eq!(app.vector_attribution(), oxigis_ui_maplibre_attribution());
    let parts = [
        app.drawn_basemap().attribution,
        expect_raster,
        expect_vector,
    ];
    let expected = parts
        .iter()
        .filter(|part| !part.is_empty())
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" \u{b7} ");
    assert_eq!(app.credit_line(), expected);

    // And a non-demo MVT source credits nobody, exactly as `config_for` says.
    let mut plain = OxigisApp::new();
    plain.apply_layer_action(LayerAction::AddVectorTileLayer(
        "https://tiles.example.test/{z}/{x}/{y}.pbf".to_string(),
    ));
    assert_eq!(
        plain.vector_attribution(),
        plain
            .desired_vector()
            .map(|config| config.attribution)
            .unwrap_or_default()
    );
    assert!(plain.vector_attribution().is_empty());
}

#[test]
fn a_promoted_layers_credit_replaces_the_services_in_both_derivations() {
    let mut app = OxigisApp::new();
    // A service whose credit differs from the promoted layer's, so "replaces"
    // is observable rather than a coincidence of two identical strings.
    app.set_basemap(BasemapConfig {
        url_template: "https://tiles.example.test/{z}/{x}/{y}.png".to_string(),
        subdomains: Vec::new(),
        attribution: "\u{a9} Example".to_string(),
    });
    app.apply_layer_action(LayerAction::AddDemoXyzLayer);
    let Some(id) = app.selection() else {
        panic!("the add seam selects its layer");
    };
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    assert!(app.draws_as_basemap(id));
    assert_eq!(
        app.credit_line(),
        app.drawn_basemap().attribution,
        "the borrow path and the config path must name the same basemap"
    );
    assert_ne!(app.drawn_basemap().attribution, app.basemap().attribution);
}

// --- The drawn stack (compositing v1.6): every visible tiled layer draws, in
// --- stack order, at the opacity the panel shows.

/// The stack a shell would report back from `map_gpu::installed_tile_stack`
/// after confirming everything the project currently implies — the mirror the
/// pure `tile_stack_work` diff is taken against.
fn settle_stack(app: &OxigisApp) -> Vec<TileLayerPlan> {
    let installed = app.desired_tile_stack().entries;
    assert_eq!(
        app.tile_stack_work(&installed),
        None,
        "a stack that matches the project owes no work"
    );
    installed
}

/// The layer ids of a stack plan, bottom-up.
fn stack_ids(entries: &[TileLayerPlan]) -> Vec<oxigis_core::LayerId> {
    entries.iter().map(|entry| entry.layer).collect()
}

#[test]
fn two_raster_layers_both_draw_and_in_stack_order() {
    // THE defect: `desired_raster` `find_map`s the top-most visible raster
    // layer, so an orthophoto under a hillshade — or two adjacent Sentinel
    // scenes — drew exactly one of the two, with the other listed and ticked in
    // the panel.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/lower.tif".to_string(),
    ));
    let lower = app.selection().expect("the add seam selects its layer");
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/upper.tif".to_string(),
    ));
    let upper = app.selection().expect("the add seam selects its layer");

    let stack = app.desired_tile_stack();
    assert_eq!(
        stack_ids(&stack.entries),
        vec![lower, upper],
        "both raster layers draw, bottom-up"
    );
    assert!(stack.undrawn.is_empty());
    assert!(stack.draws(lower) && stack.draws(upper));
    match (&stack.entries[0].source, &stack.entries[1].source) {
        (TileLayerSource::Cog(bottom), TileLayerSource::Cog(top)) => {
            assert_eq!(bottom.url, "https://example.test/lower.tif");
            assert_eq!(top.url, "https://example.test/upper.tif");
        }
        other => panic!("two COG layers must yield two COG entries, got {other:?}"),
    }

    // The legacy single slot still answers "the top-most one", untouched — the
    // shells that have not migrated keep drawing exactly what they drew.
    assert_eq!(
        app.desired_raster().cog.map(|cog| cog.url),
        Some("https://example.test/upper.tif".to_string())
    );

    // Reordering the panel reorders the passes, and nothing else.
    app.apply_layer_action(LayerAction::MoveUp(lower));
    assert_eq!(
        stack_ids(&app.desired_tile_stack().entries),
        vec![upper, lower]
    );
}

#[test]
fn the_stack_interleaves_raster_and_vector_tile_layers() {
    // Two vector-tile sources AND two raster ones, alternating: before the
    // stack, exactly one of each drew and the interleaving was unrepresentable.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/base.tif".to_string(),
    ));
    let raster_lower = app.selection().expect("selected");
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        "https://example.test/roads/{z}/{x}/{y}.pbf".to_string(),
    ));
    let vector_lower = app.selection().expect("selected");
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/hillshade.tif".to_string(),
    ));
    let raster_upper = app.selection().expect("selected");
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        "https://example.test/labels/{z}/{x}/{y}.pbf".to_string(),
    ));
    let vector_upper = app.selection().expect("selected");

    let stack = app.desired_tile_stack();
    assert_eq!(
        stack_ids(&stack.entries),
        vec![raster_lower, vector_lower, raster_upper, vector_upper],
        "one list, both pipelines, in the order the panel shows"
    );
    assert!(stack.entries[0].source.is_raster());
    assert!(stack.entries[1].source.is_vector());
    assert!(stack.entries[2].source.is_raster());
    assert!(stack.entries[3].source.is_vector());
    assert!(!stack.entries[1].source.is_raster());
}

#[test]
fn an_opacity_change_on_a_tiled_layer_reaches_the_paint_and_rebuilds_nothing() {
    // The discriminating test for the whole design: a slider drag emits every
    // frame, so if opacity were part of the install plan the shell would
    // rebuild the provider — and `set_provider` clears every resident texture —
    // once per frame, blanking and re-fetching the map for the length of the
    // drag. Opacity must reach the GPU as a tint and NOTHING else.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/scene.tif".to_string(),
    ));
    let cog = app.selection().expect("selected");
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    let mvt = app.selection().expect("selected");
    settle_everything(&mut app);
    let installed = settle_stack(&app);
    assert_eq!(app.tile_layer_opacity(cog), 1.0);
    assert_eq!(app.tile_layer_opacity(mvt), 1.0);

    for (target, opacity) in [(cog, 0.25_f32), (mvt, 0.5)] {
        app.apply_layer_action(LayerAction::SetOpacity(target, opacity));
        assert!(
            (app.tile_layer_opacity(target) - opacity).abs() < f32::EPSILON,
            "the value the panel wrote is the value the paint reads"
        );
        assert!(
            app.pending_raster_work().is_none(),
            "a fade must not rebuild the raster provider"
        );
        assert!(
            app.pending_vector_work().is_none(),
            "a fade must not rebuild the vector source"
        );
        assert_eq!(
            app.tile_stack_work(&installed),
            None,
            "a fade must not reinstall a single stack entry"
        );
    }
    // The plan really is unchanged — identity is the source, not the fade.
    assert_eq!(app.desired_tile_stack().entries, installed);

    // …and the whole drag is honoured, including the extremes.
    app.apply_layer_action(LayerAction::SetOpacity(cog, 0.0));
    assert_eq!(app.tile_layer_opacity(cog), 0.0);
    assert_eq!(app.tile_stack_work(&installed), None);
    assert_eq!(
        app.tile_layer_opacity(oxigis_core::LayerId::from_raw(0)),
        1.0
    );
}

#[test]
fn an_opacity_change_on_a_tiled_layer_is_one_undoable_step() {
    // The panel already recorded `ProjectOp::SetOpacity` for every layer — the
    // defect was that nothing downstream READ it for a tiled one. Now that the
    // paint does, the undo has to put the paint back too.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/scene.tif".to_string(),
    ));
    let cog = app.selection().expect("selected");
    settle_everything(&mut app);
    let installed = settle_stack(&app);

    // One drag: several frames of slider events, coalesced into one entry.
    for step in [0.9_f32, 0.7, 0.5, 0.3] {
        app.apply_layer_action(LayerAction::SetOpacity(cog, step));
    }
    assert!((app.tile_layer_opacity(cog) - 0.3).abs() < f32::EPSILON);

    assert!(app.undo_once(), "the drag is undoable");
    assert_eq!(
        app.tile_layer_opacity(cog),
        1.0,
        "one Ctrl+Z undoes the whole drag, not one frame of it"
    );
    assert_eq!(
        app.tile_stack_work(&installed),
        None,
        "an undone fade rebuilds nothing either"
    );
    assert!(app.pending_raster_work().is_none());
}

#[test]
fn hiding_and_removing_tiled_layers_takes_them_out_of_the_stack() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/lower.tif".to_string(),
    ));
    let lower = app.selection().expect("selected");
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        "https://example.test/{z}/{x}/{y}.pbf".to_string(),
    ));
    let upper = app.selection().expect("selected");
    let installed = settle_stack(&app);
    assert_eq!(installed.len(), 2);

    // A hidden layer leaves the stack, so the reconciliation drops its entry —
    // the checkbox means what it says for a tiled layer too.
    app.apply_layer_action(LayerAction::ToggleVisibility(upper));
    assert_eq!(
        app.tile_stack_work(&installed),
        Some(TileStackWork::Remove(vec![upper]))
    );
    let after_hide = settle_stack(&app);
    assert_eq!(stack_ids(&after_hide), vec![lower]);

    // Showing it again re-installs it, and then the ORDER is the second unit of
    // work — a reorder rebuilds nothing.
    app.apply_layer_action(LayerAction::ToggleVisibility(upper));
    let Some(TileStackWork::Install(plan)) = app.tile_stack_work(&after_hide) else {
        panic!("re-showing a layer must offer its install");
    };
    assert_eq!(plan.layer, upper);
    let mut mirror = after_hide.clone();
    mirror.push(plan);
    assert_eq!(
        app.tile_stack_work(&mirror),
        None,
        "an append that already lands in the right place needs no reorder"
    );

    // Removing the layer outright drops it as well.
    app.apply_layer_action(LayerAction::Remove(lower));
    assert_eq!(
        app.tile_stack_work(&mirror),
        Some(TileStackWork::Remove(vec![lower]))
    );

    // And closing the project takes the WHOLE stack off in one unit of work,
    // not one layer per frame: a File ▸ New must not show the previous
    // project's layers vanishing one by one.
    let mut closing = OxigisApp::new();
    for index in 0..3 {
        closing.apply_layer_action(LayerAction::AddCogLayer(format!(
            "https://example.test/{index}.tif"
        )));
    }
    let held = settle_stack(&closing);
    closing.new_project();
    let Some(TileStackWork::Remove(dropped)) = closing.tile_stack_work(&held) else {
        panic!("File \u{25b8} New must drop the old stack");
    };
    assert_eq!(dropped, stack_ids(&held));
    assert_eq!(closing.tile_stack_work(&[]), None);
}

#[test]
fn the_stack_work_converges_and_a_reorder_rebuilds_nothing() {
    let mut app = OxigisApp::new();
    for index in 0..3 {
        app.apply_layer_action(LayerAction::AddCogLayer(format!(
            "https://example.test/{index}.tif"
        )));
    }
    // Drive the loop exactly as a shell does: one unit of work per frame, the
    // mirror read back from what is "installed" each time.
    let mut installed: Vec<TileLayerPlan> = Vec::new();
    let mut frames = 0;
    while let Some(work) = app.tile_stack_work(&installed) {
        frames += 1;
        assert!(frames < 16, "the reconciliation must converge");
        match work {
            TileStackWork::Install(plan) => {
                match installed.iter().position(|entry| entry.layer == plan.layer) {
                    Some(index) => installed[index] = plan,
                    None => installed.push(plan),
                }
            }
            TileStackWork::Remove(layers) => {
                installed.retain(|entry| !layers.contains(&entry.layer));
            }
            TileStackWork::Reorder(order) => {
                installed.sort_by_key(|entry| {
                    order
                        .iter()
                        .position(|id| *id == entry.layer)
                        .unwrap_or(usize::MAX)
                });
            }
        }
    }
    assert_eq!(installed, app.desired_tile_stack().entries);
    assert_eq!(frames, 3, "three layers, three installs, no wasted frame");

    // A pure reorder is ONE unit of work and rebuilds nothing: the plans that
    // come back are the very ones already installed.
    let bottom = installed[0].layer;
    app.apply_layer_action(LayerAction::MoveUp(bottom));
    let Some(TileStackWork::Reorder(order)) = app.tile_stack_work(&installed) else {
        panic!("a reorder must be offered as a reorder, not as installs");
    };
    assert_eq!(order.len(), 3);
    assert_eq!(order[1], bottom);
    installed.sort_by_key(|entry| {
        order
            .iter()
            .position(|id| *id == entry.layer)
            .unwrap_or(usize::MAX)
    });
    assert_eq!(app.tile_stack_work(&installed), None);
}

#[test]
fn changing_a_layers_source_reinstalls_that_entry_in_place() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/one.tif".to_string(),
    ));
    let first = app.selection().expect("selected");
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/two.tif".to_string(),
    ));
    let mut installed = settle_stack(&app);

    // Rewrite the lower layer's URL behind the panel, exactly as a project load
    // of an edited file would: the entry must be re-installed, not appended.
    if let Some(layer) = app.project.layers.get_mut(first) {
        layer.kind = oxigis_core::LayerKind::Raster(oxigis_core::RasterSource::Cog {
            url: "https://example.test/moved.tif".to_string(),
        });
    }
    let Some(TileStackWork::Install(plan)) = app.tile_stack_work(&installed) else {
        panic!("a changed source must be re-installed");
    };
    assert_eq!(plan.layer, first);
    let index = installed
        .iter()
        .position(|entry| entry.layer == first)
        .expect("the entry is installed");
    installed[index] = plan;
    assert_eq!(
        app.tile_stack_work(&installed),
        None,
        "replacing in place keeps the stack position — it is not a reorder"
    );
    assert_eq!(installed.len(), 2);
}

#[test]
fn the_drawn_stack_is_capped_and_names_the_layers_it_left_out() {
    // A bound, not a cliff: every entry is a whole renderer with its own cache
    // and its own share of the VRAM budget, so the count has to stop somewhere
    // — and where it stops has to be visible rather than silent.
    let mut app = OxigisApp::new();
    let mut added = Vec::new();
    for index in 0..(MAX_DRAWN_TILE_LAYERS + 3) {
        app.apply_layer_action(LayerAction::AddCogLayer(format!(
            "https://example.test/{index}.tif"
        )));
        added.push(app.selection().expect("selected"));
    }
    let stack = app.desired_tile_stack();
    assert_eq!(stack.entries.len(), MAX_DRAWN_TILE_LAYERS);
    assert_eq!(stack.undrawn.len(), 3);

    // The cap keeps the TOP of the stack — the layers the pre-stack "newest
    // wins" rule drew — and buries the bottom ones.
    assert_eq!(stack.undrawn, added[..3].to_vec());
    assert_eq!(stack_ids(&stack.entries), added[3..].to_vec());
    for id in &added[..3] {
        assert!(stack.hides(*id) && !stack.draws(*id));
    }

    let Some(notice) = stack.notice() else {
        panic!("a bitten cap must say so");
    };
    assert!(notice.contains("3 more tiled layers"));
    assert!(notice.contains(&MAX_DRAWN_TILE_LAYERS.to_string()));
    // …and it is NOT folded into the credit line, which is an attribution the
    // exported page reprints and not a place for an apology about a budget.
    assert!(!app.credit_line().contains("not drawn"));

    // Hiding one of the drawn layers lets a buried one through.
    app.apply_layer_action(LayerAction::ToggleVisibility(added[MAX_DRAWN_TILE_LAYERS]));
    let stack = app.desired_tile_stack();
    assert_eq!(stack.entries.len(), MAX_DRAWN_TILE_LAYERS);
    assert_eq!(stack.undrawn.len(), 2);
    assert!(stack.draws(added[2]));

    // With nothing over the cap there is nothing to apologise for.
    let quiet = OxigisApp::new();
    assert_eq!(quiet.desired_tile_stack().notice(), None);
    assert!(quiet.desired_tile_stack().entries.is_empty());
}

#[test]
fn an_unpromoted_xyz_layer_draws_as_a_stack_entry_and_a_promoted_one_does_not() {
    // Before the stack an XYZ layer that was not promoted to the basemap was
    // listed, ticked, and drew nothing at all.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddDemoXyzLayer);
    let xyz = app.selection().expect("selected");
    let stack = app.desired_tile_stack();
    assert_eq!(stack_ids(&stack.entries), vec![xyz]);
    assert!(matches!(stack.entries[0].source, TileLayerSource::Xyz(_)));
    assert!(stack.entries[0].source.is_raster());
    // The legacy raster slot is deliberately untouched by an XYZ layer: it
    // still means "the composited COG-or-archive layer".
    assert!(app.desired_raster().cog.is_none());
    assert!(app.desired_raster().archive.is_none());

    // Promoted, it is the basemap — drawing it as a stack entry too would draw
    // it twice, which is exactly what the promoted-basemap row already says
    // does not happen.
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(xyz)));
    assert!(app.draws_as_basemap(xyz));
    assert!(app.desired_tile_stack().entries.is_empty());

    // Demoted again, it goes back to being an ordinary overlay.
    app.apply_layer_action(LayerAction::SetBasemapLayer(None));
    assert_eq!(stack_ids(&app.desired_tile_stack().entries), vec![xyz]);
}

#[test]
fn the_stack_agrees_with_the_single_slot_derivations_it_generalises() {
    // Three scans read the project — `desired_raster`, `desired_vector` and
    // `desired_tile_stack` — and they share their classifiers precisely so they
    // cannot drift. This is the pin, over every kind that reaches a slot.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/lower.tif".to_string(),
    ));
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        "https://example.test/lower/{z}/{x}/{y}.pbf".to_string(),
    ));
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/upper.tif".to_string(),
    ));
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    app.apply_layer_action(LayerAction::AddDemoXyzLayer);

    let stack = app.desired_tile_stack();
    let top_raster = stack
        .entries
        .iter()
        .rev()
        .find_map(|entry| match &entry.source {
            TileLayerSource::Cog(cog) => Some(cog.clone()),
            _ => None,
        });
    assert_eq!(top_raster, app.desired_raster().cog);
    let top_vector = stack
        .entries
        .iter()
        .rev()
        .find_map(|entry| match &entry.source {
            TileLayerSource::Vector(config) => Some(config.clone()),
            _ => None,
        });
    assert_eq!(top_vector, app.desired_vector());

    // The credit line is derived from the same layers the stack draws, so an
    // entry the stack refuses cannot be credited either.
    assert_eq!(app.vector_attribution(), oxigis_ui_maplibre_attribution());
    assert!(app.credit_line().contains(oxigis_ui_maplibre_attribution()));
}

#[test]
fn a_loaded_project_draws_its_whole_tiled_stack() {
    // The load path the single-slot seams could not express at all: a saved
    // project with two rasters and two vector sources drew one of each.
    let mut saver = OxigisApp::new();
    saver.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/one.tif".to_string(),
    ));
    saver.apply_layer_action(LayerAction::AddVectorTileLayer(
        "https://example.test/one/{z}/{x}/{y}.pbf".to_string(),
    ));
    saver.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/two.tif".to_string(),
    ));
    let faded_by_hand = saver.selection().expect("selected");
    saver.apply_layer_action(LayerAction::SetOpacity(faded_by_hand, 0.4));
    saver.sync_project_view();
    let json = saver.project().to_json_string().expect("serialize");
    let expected: Vec<TileLayerSource> = saver
        .desired_tile_stack()
        .entries
        .iter()
        .map(|entry| entry.source.clone())
        .collect();

    let mut loader = OxigisApp::new();
    loader.load_project(Project::from_json_string(&json).expect("deserialize"));
    let stack = loader.desired_tile_stack();
    let loaded: Vec<TileLayerSource> = stack
        .entries
        .iter()
        .map(|entry| entry.source.clone())
        .collect();
    assert_eq!(loaded, expected, "the saved stack draws in the saved order");
    assert_eq!(stack.entries.len(), 3);
    // The serialized opacity is the one the paint reads — the value written to
    // disk is no longer a lie nothing consumes.
    let faded = stack.entries[2].layer;
    assert!((loader.tile_layer_opacity(faded) - 0.4).abs() < 1e-6);

    // A fresh shell owes exactly one install per entry, and nothing else.
    let mut installed: Vec<TileLayerPlan> = Vec::new();
    for _ in 0..stack.entries.len() {
        let Some(TileStackWork::Install(plan)) = loader.tile_stack_work(&installed) else {
            panic!("every saved tiled layer must be offered");
        };
        installed.push(plan);
    }
    assert_eq!(loader.tile_stack_work(&installed), None);
}

#[test]
fn a_refused_stack_entry_is_named_and_a_stale_refusal_says_nothing() {
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddCogLayer(
        "https://example.test/scene.tif".to_string(),
    ));
    let cog = app.selection().expect("selected");
    assert_eq!(app.tile_layer_refusal_line(&[]), None);

    let name = app
        .project()
        .layers
        .get(cog)
        .map(|layer| layer.name.clone())
        .expect("just added");
    let Some(line) = app.tile_layer_refusal_line(&[(cog, "no range worker pool".to_string())])
    else {
        panic!("a refused entry must be reportable");
    };
    assert!(line.contains(&name), "the user has to find the row");
    assert!(line.contains("no range worker pool"));

    // A refusal naming a layer the project no longer holds is about a plan the
    // desire has moved past, so it must not badge a map that is drawing fine —
    // the same staleness rule `raster_refusal` follows.
    app.apply_layer_action(LayerAction::Remove(cog));
    assert_eq!(
        app.tile_layer_refusal_line(&[(cog, "no range worker pool".to_string())]),
        None
    );
    assert!(app.desired_tile_stack().entries.is_empty());
}

#[test]
fn the_cheap_tiled_probe_agrees_with_the_config_building_classifier() {
    // `desired_tile_stack` consults `draws_as_tile_layer` for everything past
    // the cap so it never builds a config it would drop; the two verdicts have
    // to be the same one, over every layer kind that can reach the stack.
    use crate::app::providers::draws_as_tile_layer;
    use oxigis_core::{ArchiveFormat, ArchiveRef, Layer, LayerKind, RasterSource, VectorSource};

    let kinds = [
        LayerKind::Raster(RasterSource::Cog {
            url: "https://example.test/scene.tif".to_string(),
        }),
        LayerKind::Raster(RasterSource::Xyz {
            url_template: "https://example.test/{z}/{x}/{y}.png".to_string(),
            attribution: "\u{a9} Example".to_string(),
        }),
        // An XYZ template needing a `{s}` host list cannot resolve, because a
        // layer records no subdomains — neither verdict may accept it.
        LayerKind::Raster(RasterSource::Xyz {
            url_template: "https://{s}.example.test/{z}/{x}/{y}.png".to_string(),
            attribution: String::new(),
        }),
        LayerKind::Raster(RasterSource::TileArchive {
            archive: ArchiveRef::Url {
                url: "https://example.test/tiles.pmtiles".to_string(),
            },
            format: ArchiveFormat::PmTiles,
            attribution: String::new(),
        }),
        // MBTiles over HTTP cannot be read, so it is not a stack entry either.
        LayerKind::Raster(RasterSource::TileArchive {
            archive: ArchiveRef::Url {
                url: "https://example.test/tiles.mbtiles".to_string(),
            },
            format: ArchiveFormat::MbTiles,
            attribution: String::new(),
        }),
        LayerKind::Vector(VectorSource::MvtTiles {
            url_template: "https://example.test/{z}/{x}/{y}.pbf".to_string(),
            paints: Vec::new(),
        }),
        LayerKind::Vector(VectorSource::TileArchive {
            archive: ArchiveRef::Url {
                url: "https://example.test/vector.pmtiles".to_string(),
            },
            format: ArchiveFormat::PmTiles,
            paints: Vec::new(),
            attribution: String::new(),
        }),
        LayerKind::Vector(VectorSource::TileArchive {
            archive: ArchiveRef::Url {
                url: "https://example.test/vector.mbtiles".to_string(),
            },
            format: ArchiveFormat::MbTiles,
            paints: Vec::new(),
            attribution: String::new(),
        }),
        // A local dataset is not tiled at all: neither verdict may claim it.
        LayerKind::Vector(VectorSource::LocalGeoJson {
            path: "/tmp/points.geojson".to_string(),
        }),
    ];
    for kind in kinds {
        let mut app = OxigisApp::new();
        let mut layer = Layer::new("probe", kind.clone());
        layer.visible = true;
        let id = app.project.layers.add(layer);
        let drawn = app.desired_tile_stack().draws(id);
        let Some(stored) = app.project.layers.get(id) else {
            panic!("the layer was just added");
        };
        assert_eq!(
            draws_as_tile_layer(stored),
            drawn,
            "the cheap probe and the stack disagree about {kind:?}"
        );
    }
}

// --- Compositing v1.6 / project ops v1.7: the migrated shell contract ---

/// Puts the camera at `zoom` without touching anything else — the one lever
/// the zoom-aware derivations read.
fn zoom_to(app: &mut OxigisApp, zoom: f64) {
    let view = app.map_view().with_zoom(zoom);
    app.map_panel.set_view(view);
}

/// Adds a COG layer directly, returning its id.
fn add_cog(app: &mut OxigisApp, url: &str) -> oxigis_core::LayerId {
    app.project.layers.add(oxigis_core::Layer::new(
        url,
        oxigis_core::LayerKind::Raster(oxigis_core::RasterSource::Cog {
            url: url.to_string(),
        }),
    ))
}

#[test]
fn a_stack_shell_is_offered_the_basemap_alone_and_never_the_vector_slot() {
    // THE regression the flag exists to prevent: before it, adding a COG
    // changed `desired_raster()`, so `pending_raster_work()` answered `Some`
    // and the shell called `replace_provider` — blanking and re-fetching every
    // visible basemap tile to install a provider that came out identical.
    let mut app = OxigisApp::new();
    app.set_tile_stack_shell(true);
    assert!(app.tile_stack_shell());
    settle_everything(&mut app);

    let cog = add_cog(&mut app, "https://example.test/scene.tif");
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));

    assert_eq!(
        app.pending_raster_work(),
        None,
        "the basemap did not change, so nothing is owed the raster slot"
    );
    assert_eq!(
        app.pending_vector_work(),
        None,
        "the stack owns the vector layer; filling the legacy slot too would draw it twice"
    );
    // Both layers are the stack's, and the legacy plan still *describes* the
    // COG — the derivations stay the same function of the project for every
    // shell, which is what the print snapshot and the credit line rely on.
    assert!(app.desired_tile_stack().draws(cog));
    assert_eq!(
        app.desired_raster().cog.map(|cog| cog.url).as_deref(),
        Some("https://example.test/scene.tif")
    );
}

#[test]
fn a_stack_shells_basemap_refusal_survives_an_unrelated_layer_add() {
    // `raster_refusal` compares the memo against `desired_raster_work`, not
    // `desired_raster`. Compared against the latter, adding a COG would move
    // the desire past the memo and the banner would silently drop a basemap
    // that is still not drawing.
    let mut app = OxigisApp::new();
    app.set_tile_stack_shell(true);
    let work = app
        .pending_raster_work()
        .expect("a fresh app owes a basemap");
    app.settle_raster_work(work, Err("no worker thread started".to_string()));
    assert_eq!(app.raster_refusal(), Some("no worker thread started"));

    add_cog(&mut app, "https://example.test/scene.tif");
    assert_eq!(
        app.raster_refusal(),
        Some("no worker thread started"),
        "the basemap is still the one that was refused"
    );
    assert_eq!(
        app.pending_raster_work(),
        None,
        "and the memo still suppresses the rebuild spin"
    );
}

#[test]
fn a_stack_entrys_refusal_reaches_the_one_banner() {
    let mut app = OxigisApp::new();
    let cog = add_cog(&mut app, "https://example.test/scene.tif");
    assert!(app.provider_refusal().is_none() || app.raster_refusal().is_some());

    app.set_tile_layer_refusals(vec![(
        cog,
        "the range worker pool did not start".to_string(),
    )]);
    let banner = app.provider_refusal().expect("a refused entry is visible");
    assert!(banner.contains("is not drawing"), "{banner}");
    assert!(
        banner.contains("the range worker pool did not start"),
        "{banner}"
    );

    // Retry raises the flag the shell drains to clear the map's own memos, and
    // it is take-once.
    app.apply_layer_action(LayerAction::RetryRefusedInstalls);
    assert!(app.take_tile_layer_retry());
    assert!(!app.take_tile_layer_retry());
}

#[test]
fn a_scale_limited_layer_stops_drawing_when_the_camera_leaves_its_range() {
    let mut app = OxigisApp::new();
    let cog = add_cog(&mut app, "https://example.test/detail.tif");
    app.project
        .layers
        .set_zoom_range(cog, Some(12.0), None)
        .expect("the layer was just added");

    zoom_to(&mut app, 14.0);
    assert!(app.desired_tile_stack().draws(cog), "inside the range");
    assert!(
        app.desired_raster().cog.is_some(),
        "and the legacy slot agrees"
    );

    zoom_to(&mut app, 8.0);
    assert!(
        !app.desired_tile_stack().draws(cog),
        "the camera left the range, so the layer stops drawing"
    );
    assert!(
        app.desired_raster().cog.is_none(),
        "leaving one derivation on the bare `visible` flag is how the two drift"
    );
}

#[test]
fn the_credit_line_follows_the_scale_range_too() {
    // `raster_credit`/`vector_credit` are pinned to the derivations they
    // summarise; a range honoured by one and not the other would credit a
    // source that is not on screen.
    let mut app = OxigisApp::new();
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    let id = app.selection().expect("selected");
    app.project
        .layers
        .set_zoom_range(id, None, Some(10.0))
        .expect("the layer was just added");

    zoom_to(&mut app, 6.0);
    assert!(app.desired_vector().is_some());
    assert!(app.credit_line().contains(oxigis_ui_maplibre_attribution()));

    zoom_to(&mut app, 12.0);
    assert!(app.desired_vector().is_none(), "past `max_zoom`");
    assert!(
        !app.credit_line().contains(oxigis_ui_maplibre_attribution()),
        "crediting tiles that are not drawn is as wrong as omitting the credit for tiles that are"
    );
}

#[test]
fn a_scale_limited_promoted_basemap_falls_back_to_the_service() {
    // `promoted_basemap_layer` culls through the SAME `visible_at` predicate,
    // so the promoted slot is not the one place a range is ignored. The
    // pointer is untouched — only what *draws* moves.
    let mut app = OxigisApp::new();
    let id = app.project.layers.add(oxigis_core::Layer::new(
        "Carto",
        oxigis_core::LayerKind::Raster(oxigis_core::RasterSource::Xyz {
            url_template: "https://tiles.example.org/{z}/{x}/{y}.png".to_string(),
            attribution: "\u{a9} Carto".to_string(),
        }),
    ));
    app.apply_layer_action(LayerAction::SetBasemapLayer(Some(id)));
    app.project
        .layers
        .set_zoom_range(id, Some(5.0), None)
        .expect("the layer was just added");

    zoom_to(&mut app, 9.0);
    assert!(app.draws_as_basemap(id));
    assert_eq!(
        app.drawn_basemap().url_template,
        "https://tiles.example.org/{z}/{x}/{y}.png"
    );
    assert!(
        !app.desired_tile_stack().draws(id),
        "the promoted layer is already under everything; drawing it twice would be wrong"
    );

    zoom_to(&mut app, 2.0);
    assert_eq!(
        app.project().basemap_layer,
        Some(id),
        "the pointer is recorded state and does not move with the camera"
    );
    assert!(
        !app.draws_as_basemap(id),
        "out of range, so the service is what is on screen"
    );
    assert_eq!(app.drawn_basemap(), *app.basemap());
    assert!(
        !app.desired_tile_stack().draws(id),
        "and it must not reappear as an ordinary stack entry either"
    );
}

#[test]
fn the_print_snapshot_carries_the_whole_drawn_stack_not_just_the_top_two() {
    // THE defect the field exists to close: before it, an export read
    // `cog`/`archive`/`vector` — one raster layer and one vector-tile layer —
    // while the screen composited N, so a project with an orthophoto under a
    // hillshade printed a different map from the one on screen.
    let mut app = OxigisApp::new();
    let lower = add_cog(&mut app, "https://example.test/ortho.tif");
    let upper = add_cog(&mut app, "https://example.test/hillshade.tif");
    app.apply_layer_action(LayerAction::AddVectorTileLayer(
        VectorTileConfig::maplibre_demo().url_template,
    ));
    let vector = app.selection().expect("selected");
    app.project
        .layers
        .set_opacity(lower, 0.25)
        .expect("the layer was just added");

    app.request_print();
    let request = app.take_pending_print().expect("a print must be queued");
    assert_eq!(
        request
            .stack
            .iter()
            .map(|entry| entry.layer)
            .collect::<Vec<_>>(),
        vec![lower, upper, vector],
        "bottom-up, exactly the order the passes run in"
    );
    // The snapshot is the LIVE derivation, so it cannot disagree with what the
    // map reconciles against.
    assert_eq!(
        request
            .stack
            .iter()
            .map(|entry| &entry.source)
            .collect::<Vec<_>>(),
        app.desired_tile_stack()
            .entries
            .iter()
            .map(|entry| &entry.source)
            .collect::<Vec<_>>(),
    );
    // Opacity is baked in, unlike in a `TileLayerPlan`: a PDF image stream
    // carries no slider.
    assert!((request.stack[0].opacity - 0.25).abs() < 1e-6);
    assert!((request.stack[1].opacity - 1.0).abs() < 1e-6);

    // The three legacy fields still describe the top-most of each kind, so an
    // export is unchanged for a project that has one of each.
    assert_eq!(
        request.cog.as_ref().map(|cog| cog.url.as_str()),
        Some("https://example.test/hillshade.tif"),
    );
    assert!(request.vector.is_some());
}

/// A tiny point collection — enough to make a real local vector layer.
const POINTS: &str = r#"{"type":"FeatureCollection","features":[
    {"type":"Feature","properties":{"name":"Tokyo"},
     "geometry":{"type":"Point","coordinates":[139.767,35.681]}}]}"#;

#[test]
fn ctrl_z_after_hiding_a_layer_genuinely_re_shows_it() {
    // The whole point of routing the checkbox through
    // `toggle_layer_visibility`: the old arm moved the flag WITHOUT recording,
    // so a later Ctrl+Z undid the wrong step. Asserting the undo DEPTH is not
    // enough — what the user is promised is that the layer comes back, on the
    // map as well as in the panel.
    let mut app = OxigisApp::new();
    let id = app
        .add_geojson_layer_from_text("cities", POINTS, None)
        .expect("valid GeoJSON");
    let _ = app.take_pending_local_ops();

    app.apply_layer_action(LayerAction::ToggleVisibility(id));
    assert!(
        app.project
            .layers
            .get(id)
            .is_some_and(|layer| !layer.visible),
        "the checkbox is off"
    );
    let ops = app.take_pending_local_ops();
    assert!(
        ops.iter()
            .any(|op| matches!(op, LocalLayerOp::SetVisibility(other, false) if *other == id)),
        "the GPU mirror is told, or the layer would keep drawing: {ops:?}"
    );

    assert!(app.undo_once());
    assert!(
        app.project
            .layers
            .get(id)
            .is_some_and(|layer| layer.visible),
        "Ctrl+Z re-shows the layer in the panel"
    );
    let ops = app.take_pending_local_ops();
    assert!(
        ops.iter()
            .any(|op| matches!(op, LocalLayerOp::SetVisibility(other, true) if *other == id)),
        "and on the map: the applier queues the mirror op in BOTH directions, \
         which is what makes an undo actually re-show it: {ops:?}"
    );
}
