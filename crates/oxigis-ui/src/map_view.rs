//! Central map viewport: pan/zoom camera state, driven every frame from
//! `egui` input, and the two ways it gets painted.
//!
//! # The two paint paths
//!
//! * **GPU (default).** [`MapPanelState::paint_gpu`] pushes the
//!   `egui_wgpu` paint callback built by [`crate::map_gpu::paint_callback`],
//!   which draws [`oxigis_render::MapRenderer`]'s tiles inside egui's own
//!   render pass. This is the real map.
//! * **Fallback.** [`MapPanelState::paint_fallback`] paints an egui-native
//!   grid (one coloured rectangle per
//!   [`oxigis_render::MapView::visible_placements`] entry) plus a note. It is
//!   used only when there is no `wgpu` render state at all — a `glow` host, or
//!   a headless [`egui::Context`] in tests — because pushing a `wgpu` callback
//!   there would silently paint nothing.
//!
//! Which one runs is decided by [`crate::app::OxigisApp`] from whether
//! [`crate::map_gpu::install`] succeeded; the camera half of this module is
//! identical either way.
//!
//! # Units
//!
//! [`MapPanelState::allocate`] keeps [`oxigis_render::MapView::size_px`] equal
//! to the panel rect's size in *physical* pixels (`rect.size() *
//! pixels_per_point`). That is not cosmetic: it is the invariant that makes the
//! renderer's NDC conversion land exactly on the callback rect (see
//! [`crate::map_gpu`]'s geometry contract).
//!
//! # The scale bar
//!
//! Both paint paths draw the ground-distance bar [`crate::scalebar`] derives
//! (View ▸ Scale bar, on by default). Only the painting is here; the maths,
//! and why it is Web Mercator arithmetic rather than the ellipsoidal geodesy
//! [`crate::measure`] does, are documented there.

use crate::scalebar::{
    FALLBACK_NOTE_ROW, SCALE_BAR_FONT, SCALE_BAR_HEIGHT, SCALE_BAR_MARGIN, SCALE_BAR_PAD,
    ScreenScaleBar, screen_scale_bar,
};
use egui::{
    Align2, Color32, CornerRadius, FontId, Rect, Response, Sense, Stroke, StrokeKind, Ui, Vec2,
};
use oxigis_render::{LonLat, MAX_LATITUDE_DEG, MapView, TileId};

/// Zoom level used whenever a valid camera can't be derived from the
/// caller's inputs (matches [`oxigis_core::View::default`]'s zoom).
const DEFAULT_ZOOM: f64 = 2.0;

/// What [`MapPanelState::paint_fallback`] says when nothing more specific was
/// set: the fallback is up because this host handed over no `wgpu` render
/// state at all. Any other cause must arrive through
/// [`MapPanelState::set_fallback_reason`], or the note would state something
/// false about a render state that does exist.
pub const NO_RENDER_STATE_NOTE: &str = "no wgpu render state \u{2014} tile placement preview only";

/// Scroll-wheel units per one full zoom level, tuned so a few notches of a
/// typical mouse wheel move about one zoom level.
const SCROLL_UNITS_PER_ZOOM_LEVEL: f32 = 200.0;

/// Exponential decay time constant for kinetic panning, in seconds: velocity
/// falls to `1/e` of its value every `PAN_INERTIA_TAU_S`. Chosen in the
/// 0.15-0.3s range typical of slippy maps — fast enough that the map doesn't
/// feel like it is sliding on ice, slow enough that a flick still carries the
/// view a visible distance.
const PAN_INERTIA_TAU_S: f32 = 0.2;

/// Velocity magnitude (physical px/s) below which kinetic panning stops and
/// [`PanInertia::is_active`] goes false, so [`MapPanelState::allocate`] can
/// stop calling [`egui::Context::request_repaint`] and let the UI go idle
/// again instead of animating forever at an imperceptible crawl.
const PAN_INERTIA_STOP_PX_PER_S: f32 = 4.0;

/// Upper bound on the `dt` fed to [`PanInertia::tick`] in a single step.
/// Frame-time spikes (a stall, a debugger pause, the very first frame after
/// the window regains focus) would otherwise hand the decay math a huge `dt`
/// and either overshoot the pan by a large jump or, through the `exp`
/// underflowing to zero, cut the animation off with a visible snap; clamping
/// keeps a single [`PanInertia::tick`] call physically plausible regardless
/// of what egui reports.
const PAN_INERTIA_MAX_DT_S: f32 = 0.1;

/// Hard ceiling on the fling speed handed to kinetic panning, in physical
/// px/s. A vigorous human flick peaks around 1500–3000 px/s; anything far
/// beyond that is an input glitch, not a gesture. The cap matters because
/// pointer stacks are not trustworthy at this boundary — WSLg/XWayland in
/// particular can interleave window-local and screen-global coordinates, and
/// the resulting position jump reads as an enormous velocity that would
/// carry the camera `v·τ` = tens of thousands of pixels, slamming it into
/// the antimeridian and leaving the user staring at gray. Capped, the worst
/// glitch glides at most `PAN_INERTIA_MAX_PX_PER_S * PAN_INERTIA_TAU_S`
/// ≈ 800 px and stops.
const PAN_INERTIA_MAX_PX_PER_S: f32 = 4_000.0;

/// A single drag frame whose own speed exceeds this multiple of
/// [`PAN_INERTIA_MAX_PX_PER_S`] is treated as a pointer glitch and excluded
/// from the release-velocity estimate entirely (2× keeps genuinely violent
/// human flicks, which land between 1× and 2×, in the estimate — they are
/// then capped by [`PanInertia::start`] anyway).
const DRAG_GLITCH_PX_PER_S: f32 = PAN_INERTIA_MAX_PX_PER_S * 2.0;

/// How far back, in seconds of drag time, the release-velocity estimate
/// looks. Short on purpose: the fling should continue the motion of the last
/// few frames before the pointer lifted, not the average of the whole drag.
const DRAG_VELOCITY_WINDOW_S: f32 = 0.12;

/// Upper bound on the number of drag frames the release-velocity estimate
/// retains, so a high-refresh display cannot grow the sample window.
const DRAG_VELOCITY_MAX_SAMPLES: usize = 8;

/// Whether the map panel may consume this frame's primary drag as a camera
/// pan.
///
/// Returned by the closure [`MapPanelState::allocate_gated`] calls *after* the
/// panel rect and its [`Response`] exist, which is what makes the decision a
/// real hit test rather than a guess from the previous frame's hover state —
/// the distinction that matters on touch, where the first frame the pointer
/// exists at all is already the press frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PanGate {
    /// The camera owns the primary drag: pan, fling and inertia behave exactly
    /// as [`MapPanelState::allocate`] has always made them behave.
    #[default]
    Allow,
    /// Something else owns the drag (an edit-mode vertex handle). The camera
    /// stays put, no fling velocity is accumulated, and any live inertia is
    /// cancelled, so releasing the vertex cannot launch the map.
    ///
    /// Wheel and pinch zoom are **never** suppressed — see
    /// [`MapPanelState::allocate_gated`].
    Suppress,
}

/// Release-velocity estimator for drag-to-pan: a tiny window of the drag's
/// own most recent `(pan delta, dt)` frames.
///
/// This exists because `egui`'s `pointer.velocity()` proved untrustworthy on
/// at least WSLg/XWayland (see [`PAN_INERTIA_MAX_PX_PER_S`]): estimating
/// from the deltas that actually panned the map keeps the fling consistent
/// with the motion the user saw, and lets a lone glitch frame be discarded
/// ([`DRAG_GLITCH_PX_PER_S`]) instead of poisoning the estimate.
#[derive(Debug, Default)]
struct DragVelocityTracker {
    /// Most recent drag frames, oldest first: pan delta in physical px, and
    /// the frame's `dt` in seconds (always finite and positive — enforced by
    /// [`Self::push`]).
    samples: Vec<(Vec2, f32)>,
}

impl DragVelocityTracker {
    /// Records one drag frame. Non-finite input or a non-positive `dt` is
    /// dropped at this boundary so every stored sample is safe to divide by.
    fn push(&mut self, delta_px: Vec2, dt_s: f32) {
        if !delta_px.x.is_finite() || !delta_px.y.is_finite() || !dt_s.is_finite() || dt_s <= 0.0 {
            return;
        }
        self.samples.push((delta_px, dt_s));
        while self.samples.len() > DRAG_VELOCITY_MAX_SAMPLES {
            self.samples.remove(0);
        }
        // Drop frames the time window no longer needs (always keeping the
        // newest), so a slow, stately drag doesn't fling from stale motion.
        loop {
            let total: f32 = self.samples.iter().map(|(_, dt)| dt).sum();
            if self.samples.len() <= 1 || total <= DRAG_VELOCITY_WINDOW_S {
                break;
            }
            if total - self.samples[0].1 >= DRAG_VELOCITY_WINDOW_S {
                self.samples.remove(0);
            } else {
                break;
            }
        }
    }

    /// Forgets the current drag (new pointer-down, or the fling was handed
    /// off).
    fn clear(&mut self) {
        self.samples.clear();
    }

    /// The velocity to fling with when the drag releases: windowed average of
    /// the recorded frames, excluding glitch frames whose individual speed
    /// exceeds [`DRAG_GLITCH_PX_PER_S`]. No usable frames → zero (no fling),
    /// which is also what a drag that ended stationary produces, since its
    /// trailing frames carry zero deltas.
    fn release_velocity(&self) -> Vec2 {
        let mut delta_sum = Vec2::ZERO;
        let mut dt_sum = 0.0_f32;
        for &(delta, dt) in &self.samples {
            if delta.length() / dt > DRAG_GLITCH_PX_PER_S {
                continue;
            }
            delta_sum += delta;
            dt_sum += dt;
        }
        if dt_sum <= 0.0 {
            return Vec2::ZERO;
        }
        delta_sum / dt_sum
    }
}

/// Kinetic-panning physics: a screen-space velocity that decays
/// exponentially toward zero once a drag is released, so the map keeps
/// gliding for a moment instead of stopping dead when the pointer lifts.
///
/// Deliberately just a velocity plus decay/stop constants and no dependency
/// on `egui::Context` or `Response`, so the decay curve, the stop epsilon,
/// cancellation, and non-finite-input handling are all testable as plain
/// struct math (see the `pan_inertia_*` tests below) without a UI harness.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PanInertia {
    /// Current screen-space velocity, in physical px/s, using the same sign
    /// convention as [`Response::drag_delta`] (and therefore
    /// [`MapPanelState::apply_pan`]'s `delta_px`): positive x/y is the
    /// direction the pointer is moving, which is also the direction the
    /// camera pans.
    velocity_px_per_s: Vec2,
}

impl Default for PanInertia {
    fn default() -> Self {
        Self {
            velocity_px_per_s: Vec2::ZERO,
        }
    }
}

impl PanInertia {
    /// Starts (or replaces) the kinetic pan with `velocity_px_per_s`,
    /// typically the pointer's velocity at the moment a drag is released.
    ///
    /// Non-finite input (should not happen from egui, but this is the
    /// boundary where an outside value enters the physics) is treated as no
    /// velocity at all rather than propagating NaN/inf into every future
    /// [`Self::tick`]. A velocity already below the stop threshold is
    /// likewise dropped immediately, so a released drag that ended almost
    /// stationary doesn't spend one frame "active" for no visible motion.
    /// A speed above [`PAN_INERTIA_MAX_PX_PER_S`] is capped to it (direction
    /// preserved) — see that constant for why implausible speeds must not
    /// reach the physics.
    fn start(&mut self, velocity_px_per_s: Vec2) {
        let finite = velocity_px_per_s.x.is_finite() && velocity_px_per_s.y.is_finite();
        self.velocity_px_per_s = if finite {
            let speed = velocity_px_per_s.length();
            if speed < PAN_INERTIA_STOP_PX_PER_S {
                Vec2::ZERO
            } else if speed > PAN_INERTIA_MAX_PX_PER_S {
                velocity_px_per_s * (PAN_INERTIA_MAX_PX_PER_S / speed)
            } else {
                velocity_px_per_s
            }
        } else {
            Vec2::ZERO
        };
    }

    /// Zeroes the velocity on the axes where the camera has hit the edge of
    /// the world ([`MapPanelState::apply_pan`]'s return value), so a fling
    /// doesn't keep grinding against the antimeridian or the Mercator
    /// cut-off — requesting repaints for motion that can no longer happen —
    /// while the other axis keeps gliding. Falls below the stop threshold →
    /// the glide ends exactly as natural decay would.
    fn stop_axis(&mut self, x: bool, y: bool) {
        if x {
            self.velocity_px_per_s.x = 0.0;
        }
        if y {
            self.velocity_px_per_s.y = 0.0;
        }
        if self.velocity_px_per_s.length() < PAN_INERTIA_STOP_PX_PER_S {
            self.velocity_px_per_s = Vec2::ZERO;
        }
    }

    /// Stops the kinetic pan immediately (new pointer-down, wheel/pinch
    /// zoom, or any other input that should take over from inertia).
    fn cancel(&mut self) {
        self.velocity_px_per_s = Vec2::ZERO;
    }

    /// Whether the kinetic pan is still carrying the view — i.e. whether the
    /// caller needs to keep calling [`Self::tick`] and requesting repaints.
    fn is_active(&self) -> bool {
        self.velocity_px_per_s != Vec2::ZERO
    }

    /// Advances the decay by `dt` seconds and returns how far to pan the
    /// camera this step, in the same physical-px, [`Response::drag_delta`]
    /// sign convention as [`Self::velocity_px_per_s`] — i.e. the caller
    /// passes the result straight to [`MapPanelState::apply_pan`].
    ///
    /// `dt` is clamped to `0.0..=`[`PAN_INERTIA_MAX_DT_S`] (non-finite `dt`
    /// becomes `0.0`) before it reaches the exponential, so a pathological
    /// frame time can neither jump the pan by an implausible distance nor
    /// underflow the decay to an instant stop. Velocity decays as
    /// `v *= exp(-dt/tau)`; once it falls below
    /// [`PAN_INERTIA_STOP_PX_PER_S`] it snaps to zero, ending the animation
    /// and turning [`Self::is_active`] false.
    fn tick(&mut self, dt: f32) -> Vec2 {
        if !self.is_active() {
            return Vec2::ZERO;
        }
        let dt = if dt.is_finite() {
            dt.clamp(0.0, PAN_INERTIA_MAX_DT_S)
        } else {
            0.0
        };
        if dt <= 0.0 {
            return Vec2::ZERO;
        }
        let step = self.velocity_px_per_s * dt;
        let decay = (-dt / PAN_INERTIA_TAU_S).exp();
        self.velocity_px_per_s *= decay;
        if self.velocity_px_per_s.length() < PAN_INERTIA_STOP_PX_PER_S {
            self.velocity_px_per_s = Vec2::ZERO;
        }
        step
    }
}

/// Pan/zoom camera state for the central map panel, plus both of its paint
/// paths (see the module docs).
pub struct MapPanelState {
    /// The camera. The single source of truth handed to the GPU renderer.
    view: MapView,
    /// Kinetic panning left over from the most recent drag release; ticked
    /// forward each frame in [`Self::allocate`] and cancelled by any new
    /// pointer-down or wheel/pinch zoom.
    inertia: PanInertia,
    /// Release-velocity estimate for the drag currently in progress; feeds
    /// [`Self::inertia`] when the pointer lifts.
    drag_velocity: DragVelocityTracker,
    /// One-line input diagnostic for the readout, present only while egui
    /// believes a pointer button is down, a touch is active, or the panel is
    /// being dragged. Exists because phantom input is real on some pointer
    /// stacks (WSLg/XWayland): when "the map moves with no button held" is
    /// reported, this line shows which button state egui is acting on, so
    /// the mechanism is diagnosable from a screenshot instead of guesswork.
    ///
    /// Always [`None`] unless [`Self::set_input_diagnostics`] turned the
    /// mechanism on — it is a developer aid, not part of the map.
    input_diag: Option<String>,
    /// Whether [`Self::input_diag`] is collected at all. Off by default: the
    /// line is orange debug text over the user's map, and building it costs a
    /// `format!` on every frame of every drag.
    input_diagnostics: bool,
    /// Why the fallback is up, when it is up for a reason other than "this host
    /// has no `wgpu`" — see [`Self::set_fallback_reason`].
    fallback_reason: Option<String>,
    /// Whether the ground-distance bar is drawn in the bottom-left corner.
    ///
    /// **On** by default, unlike every other overlay switch in this module: a
    /// map with no scale reference is not an analytical view, and a user who
    /// has to go and find the toggle before the numbers on screen mean
    /// anything has already read the map wrong once. View ▸ Scale bar turns it
    /// off for the rare case where the corner is wanted for something else.
    scale_bar: bool,
}

impl MapPanelState {
    /// Builds a panel state centered at `center_lon`/`center_lat`, at
    /// `zoom`, sized for `size_px` physical pixels.
    ///
    /// Falls back to the OxiGIS default view (`0°, 0°`, zoom
    /// `DEFAULT_ZOOM`) if the inputs are rejected by
    /// [`oxigis_render::MapView::new`] (e.g. a non-finite zoom, or the
    /// zero-size viewport egui can report on the very first layout pass).
    #[must_use]
    pub fn new(center_lon: f64, center_lat: f64, zoom: f64, size_px: [f32; 2]) -> Self {
        let size = clamp_size(size_px);
        let zoom = if zoom.is_finite() { zoom } else { DEFAULT_ZOOM };
        let view = match MapView::new(LonLat::new(center_lon, center_lat), zoom, size) {
            Ok(view) => view,
            Err(_) => default_view(size),
        };
        Self {
            view,
            inertia: PanInertia::default(),
            drag_velocity: DragVelocityTracker::default(),
            input_diag: None,
            input_diagnostics: false,
            fallback_reason: None,
            scale_bar: true,
        }
    }

    /// Whether the on-screen scale bar is drawn (default: yes).
    #[must_use]
    pub fn scale_bar_visible(&self) -> bool {
        self.scale_bar
    }

    /// Shows or hides the on-screen scale bar — View ▸ Scale bar.
    pub fn set_scale_bar_visible(&mut self, visible: bool) {
        self.scale_bar = visible;
    }

    /// The scale bar as it would be drawn into a panel `panel_width` logical
    /// pixels wide, or [`None`] when it is switched off or the geometry gives
    /// nothing worth drawing.
    ///
    /// The seam a test reads instead of a screenshot, and the same call
    /// `Self::paint_scale_bar` makes.
    #[must_use]
    pub fn scale_bar(&self, panel_width: f32, ppp: f32) -> Option<ScreenScaleBar> {
        self.scale_bar
            .then(|| screen_scale_bar(&self.view, panel_width, ppp))
            .flatten()
    }

    /// The current camera.
    #[must_use]
    pub fn view(&self) -> MapView {
        self.view
    }

    /// Turns the pointer-state diagnostic line over the map on or off.
    ///
    /// Off by default, and deliberately a runtime switch rather than a
    /// `cfg!(debug_assertions)` one: the input glitches it exists to diagnose
    /// (phantom pointer buttons on WSLg/XWayland) only reproduce on real
    /// pointer stacks, so a
    /// user on a release build has to be able to turn it on. A shell flips it
    /// from whatever it uses for developer switches — this crate compiles to
    /// `wasm32` and reads no environment of its own.
    pub fn set_input_diagnostics(&mut self, enabled: bool) {
        self.input_diagnostics = enabled;
        if !enabled {
            self.input_diag = None;
        }
    }

    /// Whether the pointer-state diagnostic line is being collected.
    #[must_use]
    pub fn input_diagnostics(&self) -> bool {
        self.input_diagnostics
    }

    /// Replaces the reason [`Self::paint_fallback`] gives for not drawing the
    /// real map.
    ///
    /// [`None`] restores [`NO_RENDER_STATE_NOTE`], which is only true when
    /// there is no `wgpu` render state at all. A shell whose GPU map *install*
    /// failed passes the failure here, because the default note would then be
    /// a false statement about a real render state and would misdirect the
    /// diagnosis.
    /// Called once per fallback frame, so it compares before it allocates.
    pub fn set_fallback_reason(&mut self, reason: Option<&str>) {
        if self.fallback_reason.as_deref() == reason {
            return;
        }
        self.fallback_reason = reason.map(str::to_owned);
    }

    /// Replaces the camera outright (e.g. restoring a saved project's
    /// [`oxigis_core::View`]).
    pub fn set_view(&mut self, view: MapView) {
        self.view = view;
    }

    /// Allocates the panel's rect, applies drag-to-pan and scroll-to-zoom
    /// from this frame's input, and returns the rect plus the interaction
    /// response.
    ///
    /// Uses physical pixels throughout (`rect.size() * pixels_per_point()`),
    /// matching [`oxigis_render::MapView`]'s documented unit contract, so a
    /// shell driving the real GPU pipeline can feed [`MapPanelState::view`]
    /// straight into [`oxigis_render::MapRenderer::begin_frame`] without any
    /// unit conversion.
    ///
    /// Exactly [`Self::allocate_gated`] with a gate that always answers
    /// [`PanGate::Allow`] — a caller that wants to hand the primary drag to
    /// something else (edit-mode vertex handles) uses that entry point.
    pub fn allocate(&mut self, ui: &mut Ui) -> (Rect, Response) {
        self.allocate_gated(ui, |_rect, _response, _ppp| PanGate::Allow)
    }

    /// Allocates the rect and applies wheel/pinch zoom exactly as
    /// [`Self::allocate`] does, but asks `gate` — after the rect and the
    /// [`Response`] exist, and before any pan is applied — whether this
    /// frame's primary drag belongs to the camera.
    ///
    /// `gate` receives the panel rect, the frame's response and the context's
    /// `pixels_per_point`, so it can hit-test the live pointer position
    /// itself. A [`PanGate::Suppress`] result guards **both** the
    /// `dragged_by(Primary)` pan branch and the `drag_stopped_by(Primary)`
    /// fling handoff, records no drag velocity, and cancels any inertia still
    /// in flight.
    ///
    /// Wheel and pinch zoom are never gated: zooming during a vertex drag is
    /// useful and harmless, because an editing gesture re-derives its position
    /// in lon/lat every frame rather than holding it in pixels.
    ///
    /// [`Self::allocate`] is exactly `allocate_gated(ui, |_, _, _|
    /// PanGate::Allow)`, so nothing that does not opt in changes behaviour.
    pub fn allocate_gated(
        &mut self,
        ui: &mut Ui,
        gate: impl FnOnce(Rect, &Response, f32) -> PanGate,
    ) -> (Rect, Response) {
        let rect = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(rect, Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point();
        let size_px = clamp_size([rect.width() * ppp, rect.height() * ppp]);
        if let Ok(resized) = self.view.with_size_px(size_px) {
            self.view = resized;
        }

        // Any fresh pointer-down on the panel (starting a new drag, or just a
        // click) takes over from whatever kinetic pan is still running.
        if response.drag_started()
            || (response.hovered() && ui.input(|i| i.pointer.primary_pressed()))
        {
            self.inertia.cancel();
            self.drag_velocity.clear();
        }

        // The gate runs here — after the resize and the inertia-cancel block,
        // before a single pixel of pan is applied — so its verdict is made
        // from this frame's real rect and response.
        let pan_gate = gate(rect, &response, ppp);

        // Pan strictly on PRIMARY-button drags. `Response::dragged()` is
        // true for a drag by ANY button (egui 0.35 `response.rs:416`), and
        // phantom button state is real on some pointer stacks — WSLg/
        // XWayland has been observed feeding egui a held-down button the
        // user never pressed, which made the map follow bare mouse motion.
        // Restricting to `dragged_by(Primary)` both matches the intended
        // gesture (secondary is reserved for future context menus) and
        // immunises panning against phantom secondary/middle/touch presses.
        let primary = egui::PointerButton::Primary;
        if pan_gate == PanGate::Suppress {
            // Someone else owns this drag. Nothing is panned, nothing is
            // remembered as fling velocity (so the release below cannot
            // hand one off), and a glide already in flight stops dead —
            // otherwise the map would keep sliding under the handle the
            // user is trying to place.
            self.inertia.cancel();
            self.drag_velocity.clear();
        } else if response.dragged_by(primary) {
            let delta = response.drag_delta() * ppp;
            let dt = ui.ctx().input(|i| i.unstable_dt);
            self.drag_velocity.push(delta, dt);
            self.apply_pan([delta.x, delta.y]);
        } else if response.drag_stopped_by(primary) {
            // The pointer just lifted: hand off the drag's exit velocity to
            // inertia so the pan keeps gliding instead of stopping dead.
            // Deliberately NOT `pointer.velocity()`: that estimate comes
            // straight from the pointer stack, and on WSLg/XWayland it can
            // report a coordinate-space jump as an enormous velocity — the
            // observed failure was every drag release flinging the map to
            // the antimeridian, leaving only gray. The tracker estimates
            // from the deltas that actually panned the map instead.
            self.inertia.start(self.drag_velocity.release_velocity());
            self.drag_velocity.clear();
        }

        // Input diagnostic (see the field docs): captured while any button/
        // touch/drag is live, cleared to nothing the moment input goes idle.
        // Nothing at all is read — let alone formatted — while the switch is
        // off, which is the state a released build runs in.
        self.input_diag = if !self.input_diagnostics {
            None
        } else {
            let (p, s, m, touch, any_down) = ui.input(|i| {
                (
                    i.pointer.button_down(egui::PointerButton::Primary),
                    i.pointer.button_down(egui::PointerButton::Secondary),
                    i.pointer.button_down(egui::PointerButton::Middle),
                    i.any_touches(),
                    i.pointer.any_down(),
                )
            });
            (response.dragged() || any_down || touch).then(|| {
                format!(
                    "input: drag={} P={} S={} M={} touch={}",
                    u8::from(response.dragged()),
                    u8::from(p),
                    u8::from(s),
                    u8::from(m),
                    u8::from(touch),
                )
            })
        };

        if response.hovered() {
            let (scroll_y, pinch) =
                ui.input(|input| (input.smooth_scroll_delta().y, input.zoom_delta()));
            let levels = zoom_levels(scroll_y, pinch);
            if levels.is_finite() && levels != 0.0 {
                // A deliberate zoom takes over from any kinetic pan in
                // flight, same as a fresh pointer-down.
                self.inertia.cancel();
            }
            // Physical pixels from the panel's north-west corner — the frame
            // `MapView`'s screen coordinates are expressed in.
            let cursor = response.hover_pos().map(|pos| {
                let local = pos - rect.min;
                [local.x * ppp, local.y * ppp]
            });
            self.apply_zoom_at(levels, cursor);
        }

        // Kinetic panning: only while nothing else is actively driving the
        // camera this frame (an active drag already applied its own delta
        // above, and takes priority even on the same frame a drag stops).
        if !response.dragged() && self.inertia.is_active() {
            let dt = ui.ctx().input(|i| i.unstable_dt);
            let delta = self.inertia.tick(dt);
            if delta != Vec2::ZERO {
                let walls = self.apply_pan([delta.x, delta.y]);
                self.inertia.stop_axis(walls[0], walls[1]);
            }
            if self.inertia.is_active() {
                ui.ctx().request_repaint();
            }
        }

        (rect, response)
    }

    /// Pans the camera so the view scrolls by `delta_px` physical pixels
    /// (positive x drags the map right, positive y drags it down), matching
    /// how [`Self::allocate`] interprets `response.drag_delta()`. Extracted
    /// from [`Self::allocate`] so drag-to-pan is unit-testable without a
    /// simulated egui drag gesture.
    ///
    /// The camera saturates at the edge of the world instead of leaving it:
    /// [`oxigis_render::MapView::screen_to_lon_lat`] clamps to the Mercator
    /// unit square, so the centre stops at ±180° / ±[`MAX_LATITUDE_DEG`] and
    /// the map always remains reachable on screen (the tile grid does not
    /// repeat across the antimeridian, so wrapping the centre would teleport
    /// the whole map a world-width sideways). Returns, per axis, whether
    /// this pan pressed against that edge — `[x_wall, y_wall]` — so a
    /// kinetic fling can stop instead of grinding against it.
    fn apply_pan(&mut self, delta_px: [f32; 2]) -> [bool; 2] {
        let center_px = self.view.lon_lat_to_screen(self.view.center());
        let new_center = self
            .view
            .screen_to_lon_lat([center_px[0] - delta_px[0], center_px[1] - delta_px[1]]);
        self.view = self.view.with_center(new_center);
        // Positive delta x reveals territory to the west (longitude falls
        // toward -180°); positive delta y reveals the north (latitude rises
        // toward the cut-off). "At the wall" = at the saturated coordinate
        // while still pushing outward.
        const EDGE_EPS: f64 = 1e-9;
        let at = self.view.center();
        let x_wall = (at.lon <= -180.0 + EDGE_EPS && delta_px[0] > 0.0)
            || (at.lon >= 180.0 - EDGE_EPS && delta_px[0] < 0.0);
        let y_wall = (at.lat >= MAX_LATITUDE_DEG - EDGE_EPS && delta_px[1] > 0.0)
            || (at.lat <= -(MAX_LATITUDE_DEG - EDGE_EPS) && delta_px[1] < 0.0);
        [x_wall, y_wall]
    }

    /// Zooms the camera by `levels` zoom levels, keeping the geographic point
    /// under `cursor_px` (physical pixels from the panel's north-west corner)
    /// pinned to that same pixel — the usual slippy-map wheel behaviour.
    /// `None` zooms about the viewport centre instead.
    ///
    /// The zoom itself is clamped to `0..=`[`oxigis_render::MAX_ZOOM`] by
    /// [`oxigis_render::MapView::with_zoom`], and the anchor correction is
    /// computed *after* that clamp, so a wheel notch at either end of the range
    /// cannot drift the centre. Extracted from [`Self::allocate`] so the maths
    /// is unit-testable without simulated egui input.
    fn apply_zoom_at(&mut self, levels: f64, cursor_px: Option<[f32; 2]>) {
        if !levels.is_finite() || levels == 0.0 {
            return;
        }
        let zoomed = self.view.with_zoom(self.view.zoom() + levels);
        let Some(cursor) = cursor_px else {
            self.view = zoomed;
            return;
        };
        // Where the point that *was* under the cursor ended up, and how far the
        // camera has to move to put it back.
        let anchor = self.view.screen_to_lon_lat(cursor);
        let anchor_px = zoomed.lon_lat_to_screen(anchor);
        let center_px = zoomed.lon_lat_to_screen(zoomed.center());
        let corrected = zoomed.screen_to_lon_lat([
            center_px[0] + anchor_px[0] - cursor[0],
            center_px[1] + anchor_px[1] - cursor[1],
        ]);
        self.view = zoomed.with_center(corrected);
    }

    /// Pushes the `egui_wgpu` paint callback that draws the real tile map into
    /// `rect` (see [`crate::map_gpu`]).
    ///
    /// A background fill goes down first, because the tile pipeline only covers
    /// the tiles it actually has, and the coordinate readout goes on top.
    pub fn paint_gpu(&self, ui: &Ui, rect: Rect) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::ZERO, Color32::from_gray(18));
        // The repainting variant: the local vector stack budgets its
        // re-tessellation, so a frame can end with work still owed and must be
        // able to ask for the frame that finishes it.
        painter.add(egui::Shape::Callback(
            crate::map_gpu::paint_callback_repainting(rect, self.view, ui.ctx()),
        ));
        self.paint_readout(&painter, rect);
        self.paint_scale_bar(&painter, rect, ui.ctx().pixels_per_point(), 0.0);
    }

    /// Paints the non-GPU fallback: one colored rectangle per currently visible
    /// tile (see the module docs), plus a coordinate readout and a note saying
    /// this is not the real renderer.
    pub fn paint_fallback(&self, ui: &Ui, rect: Rect) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, CornerRadius::ZERO, Color32::from_gray(30));

        let ppp = ui.ctx().pixels_per_point();
        for placement in self.view.visible_placements() {
            let origin = rect.min + Vec2::new(placement.x / ppp, placement.y / ppp);
            let size = placement.size / ppp;
            let tile_rect = Rect::from_min_size(origin, Vec2::splat(size)).intersect(rect);
            if tile_rect.width() <= 0.0 || tile_rect.height() <= 0.0 {
                continue;
            }
            painter.rect_filled(tile_rect, CornerRadius::ZERO, tile_color(placement.tile));
            painter.rect_stroke(
                tile_rect,
                CornerRadius::ZERO,
                Stroke::new(1.0_f32, Color32::from_gray(60)),
                StrokeKind::Inside,
            );
            painter.text(
                tile_rect.center(),
                Align2::CENTER_CENTER,
                format!(
                    "{}/{}/{}",
                    placement.tile.z, placement.tile.x, placement.tile.y
                ),
                FontId::monospace(11.0),
                Color32::from_gray(220),
            );
        }

        self.paint_readout(&painter, rect);
        // Lifted clear of the note below, which owns the same corner in this
        // paint path and would otherwise be drawn straight through the plate.
        self.paint_scale_bar(&painter, rect, ppp, FALLBACK_NOTE_ROW);
        let note = match self.fallback_reason.as_deref() {
            Some(reason) => format!(
                "the GPU map could not be attached: {reason} \u{2014} tile placement preview only"
            ),
            None => NO_RENDER_STATE_NOTE.to_owned(),
        };
        painter.text(
            rect.left_bottom() + Vec2::new(6.0, -6.0),
            Align2::LEFT_BOTTOM,
            note,
            FontId::monospace(11.0),
            Color32::from_rgb(220, 160, 90),
        );
    }

    /// Paints the `lon / lat / zoom` readout in the panel's top-left corner,
    /// plus the input-diagnostic line whenever one is live (see
    /// [`Self::input_diag`]).
    fn paint_readout(&self, painter: &egui::Painter, rect: Rect) {
        let center = self.view.center();
        painter.text(
            rect.left_top() + Vec2::new(6.0, 6.0),
            Align2::LEFT_TOP,
            format!(
                "lon {:.4}  lat {:.4}  zoom {:.2}",
                center.lon,
                center.lat,
                self.view.zoom()
            ),
            FontId::monospace(12.0),
            Color32::WHITE,
        );
        if let Some(diag) = &self.input_diag {
            painter.text(
                rect.left_top() + Vec2::new(6.0, 22.0),
                Align2::LEFT_TOP,
                diag,
                FontId::monospace(11.0),
                Color32::from_rgb(220, 160, 90),
            );
        }
    }

    /// Draws the ground-distance bar in the panel's bottom-left corner: a dark
    /// plate, a rule with an end tick at each end, and the round distance the
    /// rule spans.
    ///
    /// `lift` raises the plate off the bottom edge, for a paint path that has
    /// already claimed that corner ([`Self::paint_fallback`]'s note). Nothing
    /// is drawn when the bar is switched off or the geometry gives no bar
    /// worth drawing — see [`Self::scale_bar`].
    fn paint_scale_bar(&self, painter: &egui::Painter, rect: Rect, ppp: f32, lift: f32) {
        let Some(bar) = self.scale_bar(rect.width(), ppp) else {
            return;
        };
        let galley = painter.layout_no_wrap(
            bar.label.clone(),
            FontId::proportional(SCALE_BAR_FONT),
            Color32::from_rgb(0xF0, 0xF4, 0xFA),
        );
        // The plate is as wide as the bar OR its label, whichever needs more:
        // a `1 km` label under a short rule must not hang off the plate.
        let content_width = bar.width.max(galley.size().x);
        let plate_size = Vec2::new(
            content_width + SCALE_BAR_PAD * 2.0,
            galley.size().y + SCALE_BAR_HEIGHT + SCALE_BAR_PAD * 2.5,
        );
        let plate = Rect::from_min_size(
            egui::pos2(
                rect.left() + SCALE_BAR_MARGIN,
                rect.bottom() - SCALE_BAR_MARGIN - lift - plate_size.y,
            ),
            plate_size,
        );
        if !rect.contains_rect(plate) {
            // A panel too small to hold the plate gets no plate: half a scale
            // bar clipped at the panel edge states a distance it does not span.
            return;
        }
        painter.rect_filled(plate, 3.0, Color32::from_black_alpha(0xB0));
        let bar_left = plate.left() + SCALE_BAR_PAD;
        let bar_bottom = plate.bottom() - SCALE_BAR_PAD;
        let bar_top = bar_bottom - SCALE_BAR_HEIGHT;
        let ink = Color32::from_rgb(0xF0, 0xF4, 0xFA);
        let stroke = Stroke::new(1.5, ink);
        // The rule, with an upward tick at each end: the classic form, and the
        // one that says unambiguously where the measured span begins and ends.
        painter.line_segment(
            [
                egui::pos2(bar_left, bar_bottom),
                egui::pos2(bar_left + bar.width, bar_bottom),
            ],
            stroke,
        );
        for x in [bar_left, bar_left + bar.width] {
            painter.line_segment([egui::pos2(x, bar_bottom), egui::pos2(x, bar_top)], stroke);
        }
        painter.galley(
            egui::pos2(bar_left, plate.top() + SCALE_BAR_PAD * 0.5),
            galley,
            Color32::PLACEHOLDER,
        );
    }

    /// Convenience: [`Self::allocate`] followed by the paint path `gpu` selects
    /// ([`Self::paint_gpu`] when `true`, [`Self::paint_fallback`] otherwise).
    ///
    /// [`crate::app::OxigisApp::ui`] drives the central panel through
    /// [`Self::allocate`] directly so it can record the rect; this exists for
    /// hosts that embed the map panel on its own.
    pub fn ui(&mut self, ui: &mut Ui, gpu: bool) -> Response {
        let (rect, response) = self.allocate(ui);
        if gpu {
            self.paint_gpu(ui, rect);
        } else {
            self.paint_fallback(ui, rect);
        }
        response
    }
}

impl Default for MapPanelState {
    /// Starts at `0°, 0°`, zoom `DEFAULT_ZOOM`, sized for a nominal
    /// 800x600 window; [`MapPanelState::allocate`] immediately corrects the
    /// size to the panel's real rect on the first frame.
    fn default() -> Self {
        Self::new(0.0, 0.0, DEFAULT_ZOOM, [800.0, 600.0])
    }
}

/// Rewrites non-finite or out-of-range dimensions into
/// [`oxigis_render::MapView::new`]'s accepted range, so viewport
/// construction never fails on a degenerate egui layout (e.g. a
/// zero-size rect on the first frame of a hidden panel).
fn clamp_size(size_px: [f32; 2]) -> [f32; 2] {
    let clamp_one = |value: f32| -> f32 {
        if value.is_finite() {
            value.clamp(1.0, oxigis_render::viewport::MAX_VIEWPORT_PX)
        } else {
            1.0
        }
    };
    [clamp_one(size_px[0]), clamp_one(size_px[1])]
}

/// A view centered on `0°, 0°` at [`DEFAULT_ZOOM`], for `size_px` (already
/// passed through [`clamp_size`]).
///
/// `size_px` being pre-clamped and `DEFAULT_ZOOM` being a finite literal mean
/// the only two failure modes of [`MapView::new`] are already excluded, so
/// the first attempt always succeeds; the second attempt (a known-good 1x1
/// surface) exists only so this function has no `.unwrap()`/`.expect()` call
/// and still returns `MapView` (not `Result`) to callers, per COOLJAPAN
/// policy.
fn default_view(size_px: [f32; 2]) -> MapView {
    match MapView::new(LonLat::new(0.0, 0.0), DEFAULT_ZOOM, size_px) {
        Ok(view) => view,
        Err(_) => match MapView::new(LonLat::new(0.0, 0.0), DEFAULT_ZOOM, [1.0, 1.0]) {
            Ok(view) => view,
            // Unreachable: a 1x1 surface and a finite zoom always pass
            // `MapView::new`'s validation (see its doc comment).
            Err(_) => unreachable!("a 1x1 viewport at a finite zoom is always valid"),
        },
    }
}

/// Total zoom-level change one frame of input asks for.
///
/// `scroll_y` is egui's smoothed wheel delta (converted at
/// [`SCROLL_UNITS_PER_ZOOM_LEVEL`]) and `pinch` is
/// [`egui::InputState::zoom_delta`], a *multiplier* — one zoom level is a
/// factor of two, hence `log2`. The two never double-count: egui routes wheel
/// input into `zoom_delta` and zeroes `smooth_scroll_delta` when the zoom
/// modifier is held (`egui-0.34.3` `input_state/mod.rs`, `is_zoom` branch), and
/// feeds `zoom_delta` from multi-touch pinch otherwise.
fn zoom_levels(scroll_y: f32, pinch: f32) -> f64 {
    let from_scroll = if scroll_y.is_finite() {
        f64::from(scroll_y) / f64::from(SCROLL_UNITS_PER_ZOOM_LEVEL)
    } else {
        0.0
    };
    let from_pinch = if pinch.is_finite() && pinch > 0.0 {
        f64::from(pinch).log2()
    } else {
        0.0
    };
    from_scroll + from_pinch
}

/// A deterministic pseudo-random tint per tile, so adjacent placeholder
/// tiles in the checkerboard are visually distinguishable.
fn tile_color(tile: TileId) -> Color32 {
    let hash = u32::from(tile.z) ^ tile.x.wrapping_mul(2_654_435_761) ^ tile.y.wrapping_mul(40_503);
    let r = 40 + (hash & 0x3f) as u8;
    let g = 60 + ((hash >> 6) & 0x3f) as u8;
    let b = 90 + ((hash >> 12) & 0x3f) as u8;
    Color32::from_rgb(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pan_inertia_starts_inactive() {
        let inertia = PanInertia::default();
        assert!(!inertia.is_active());
    }

    #[test]
    fn pan_inertia_tick_while_inactive_is_a_no_op() {
        let mut inertia = PanInertia::default();
        assert_eq!(inertia.tick(0.05), Vec2::ZERO);
        assert!(!inertia.is_active());
    }

    #[test]
    fn pan_inertia_start_below_stop_threshold_stays_inactive() {
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(PAN_INERTIA_STOP_PX_PER_S * 0.5, 0.0));
        assert!(!inertia.is_active());
    }

    #[test]
    fn pan_inertia_start_makes_it_active_and_ticking_moves_in_velocity_direction() {
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(500.0, -200.0));
        assert!(inertia.is_active());
        let step = inertia.tick(0.01);
        // First step should move (roughly) in the same direction as the
        // starting velocity, i.e. positive x, negative y.
        assert!(step.x > 0.0);
        assert!(step.y < 0.0);
    }

    #[test]
    fn pan_inertia_decays_by_half_after_one_half_life() {
        // Solving `exp(-dt/tau) = 0.5` for `dt` gives tau * ln(2): the decay
        // interval at which the *velocity* (not the per-step delta) should
        // have halved.
        let mut inertia = PanInertia::default();
        let start_velocity = Vec2::new(1_000.0, 0.0);
        inertia.start(start_velocity);
        // Split the half-life across two ticks, each safely under
        // `PAN_INERTIA_MAX_DT_S`, so the clamp in `tick` doesn't distort the
        // decay this test is checking.
        let half_life = PAN_INERTIA_TAU_S * std::f32::consts::LN_2;
        inertia.tick(half_life / 2.0);
        inertia.tick(half_life / 2.0);
        assert!(
            (inertia.velocity_px_per_s.x - start_velocity.x * 0.5).abs() < 1.0,
            "expected velocity to roughly halve, got {}",
            inertia.velocity_px_per_s.x
        );
    }

    #[test]
    fn pan_inertia_stops_below_epsilon_and_goes_inactive() {
        let mut inertia = PanInertia::default();
        // Start just above the stop threshold and let a single, fairly long
        // tick decay it past the epsilon.
        inertia.start(Vec2::new(PAN_INERTIA_STOP_PX_PER_S * 1.5, 0.0));
        // Several ticks at the max clamped dt guarantee enough elapsed time
        // for the decay to cross the stop threshold regardless of tau.
        for _ in 0..50 {
            inertia.tick(PAN_INERTIA_MAX_DT_S);
            if !inertia.is_active() {
                break;
            }
        }
        assert!(!inertia.is_active());
        assert_eq!(inertia.tick(0.05), Vec2::ZERO);
    }

    #[test]
    fn pan_inertia_cancel_stops_it_immediately() {
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(1_000.0, 1_000.0));
        assert!(inertia.is_active());
        inertia.cancel();
        assert!(!inertia.is_active());
        assert_eq!(inertia.tick(0.05), Vec2::ZERO);
    }

    #[test]
    fn pan_inertia_start_rejects_non_finite_velocity() {
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(f32::NAN, 0.0));
        assert!(!inertia.is_active());
        inertia.start(Vec2::new(f32::INFINITY, 0.0));
        assert!(!inertia.is_active());
    }

    #[test]
    fn pan_inertia_tick_clamps_non_finite_or_out_of_range_dt() {
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(1_000.0, 0.0));
        // NaN dt must not propagate into the velocity or the returned delta.
        let step = inertia.tick(f32::NAN);
        assert_eq!(step, Vec2::ZERO);
        assert!(inertia.velocity_px_per_s.x.is_finite());

        // A huge dt is clamped to `PAN_INERTIA_MAX_DT_S`, not applied as-is
        // — otherwise a single frame stall could fling the camera an
        // implausible distance.
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(1_000.0, 0.0));
        let clamped_step = inertia.tick(PAN_INERTIA_MAX_DT_S);
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(1_000.0, 0.0));
        let huge_step = inertia.tick(1_000.0);
        assert_eq!(huge_step, clamped_step);
        assert!(huge_step.x.is_finite() && huge_step.y.is_finite());
    }

    #[test]
    fn pan_inertia_tick_negative_dt_is_a_no_op() {
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(1_000.0, 0.0));
        let before = inertia.velocity_px_per_s;
        assert_eq!(inertia.tick(-1.0), Vec2::ZERO);
        assert_eq!(inertia.velocity_px_per_s, before);
    }

    #[test]
    fn pan_inertia_start_caps_implausible_speeds_but_keeps_the_direction() {
        // A glitching pointer stack (WSLg) can report absurd velocities; the
        // fling must be bounded or the camera slams into the world edge.
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(1.0e9, 0.0));
        assert!(inertia.is_active());
        assert!((inertia.velocity_px_per_s.length() - PAN_INERTIA_MAX_PX_PER_S).abs() < 1e-3);
        assert!(inertia.velocity_px_per_s.x > 0.0);
        assert_eq!(inertia.velocity_px_per_s.y, 0.0);

        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(0.0, -1.0e9));
        assert!((inertia.velocity_px_per_s.y + PAN_INERTIA_MAX_PX_PER_S).abs() < 1e-3);

        // A plausible fling passes through uncapped.
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(1_500.0, 0.0));
        assert_eq!(inertia.velocity_px_per_s, Vec2::new(1_500.0, 0.0));
    }

    #[test]
    fn pan_inertia_stop_axis_zeroes_one_axis_and_can_end_the_glide() {
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(1_000.0, 500.0));
        inertia.stop_axis(false, true);
        assert_eq!(inertia.velocity_px_per_s, Vec2::new(1_000.0, 0.0));
        assert!(inertia.is_active());
        inertia.stop_axis(true, false);
        assert!(!inertia.is_active());

        // Zeroing one axis can drop the remaining speed below the stop
        // threshold, which must end the glide entirely.
        let mut inertia = PanInertia::default();
        inertia.start(Vec2::new(1_000.0, PAN_INERTIA_STOP_PX_PER_S * 0.5));
        inertia.stop_axis(true, false);
        assert!(!inertia.is_active());
    }

    #[test]
    fn apply_pan_saturates_at_the_antimeridian_and_reports_the_wall() {
        let mut panel = MapPanelState::new(0.0, 0.0, 2.0, [512.0, 512.0]);
        // Dragging the map right far beyond the world reveals the west; the
        // centre must stop at -180°, not wrap to the eastern hemisphere
        // (which would teleport the map a world-width sideways — the
        // "map slides right into gray" failure).
        let walls = panel.apply_pan([1.0e6, 0.0]);
        assert!((panel.view().center().lon - (-180.0)).abs() < 1e-6);
        assert!(walls[0], "a pan that saturates must report the x wall");

        // Pressing further outward stays pinned and keeps reporting the wall.
        let walls = panel.apply_pan([10.0, 0.0]);
        assert!((panel.view().center().lon - (-180.0)).abs() < 1e-6);
        assert!(walls[0]);

        // Backing away from the wall works and is not reported as a wall.
        let walls = panel.apply_pan([-50.0, 0.0]);
        assert!(panel.view().center().lon > -180.0);
        assert!(!walls[0]);
    }

    #[test]
    fn apply_pan_saturates_at_the_mercator_cutoff_vertically() {
        let mut panel = MapPanelState::new(0.0, 0.0, 2.0, [512.0, 512.0]);
        // Dragging down reveals the north; the centre stops at the cut-off.
        let walls = panel.apply_pan([0.0, 1.0e6]);
        assert!((panel.view().center().lat - MAX_LATITUDE_DEG).abs() < 1e-6);
        assert!(walls[1], "a pan that saturates must report the y wall");

        let walls = panel.apply_pan([0.0, -50.0]);
        assert!(panel.view().center().lat < MAX_LATITUDE_DEG);
        assert!(!walls[1]);
    }

    #[test]
    fn a_runaway_fling_always_stops_inside_the_world() {
        // Replays `allocate`'s inertia branch against the worst velocity the
        // cap allows, and asserts the two invariants the reported bug broke:
        // the camera never leaves the world, and the glide terminates.
        let mut panel = MapPanelState::new(0.0, 0.0, 2.0, [512.0, 512.0]);
        panel.inertia.start(Vec2::new(1.0e9, -1.0e9));
        let mut ticks = 0;
        while panel.inertia.is_active() {
            let delta = panel.inertia.tick(0.016);
            if delta != Vec2::ZERO {
                let walls = panel.apply_pan([delta.x, delta.y]);
                panel.inertia.stop_axis(walls[0], walls[1]);
            }
            ticks += 1;
            assert!(ticks < 1_000, "fling did not terminate");
            let center = panel.view().center();
            assert!((-180.0..=180.0).contains(&center.lon));
            assert!(center.lat.abs() <= MAX_LATITUDE_DEG + 1e-9);
        }
    }

    #[test]
    fn drag_velocity_tracker_averages_a_steady_drag() {
        let mut tracker = DragVelocityTracker::default();
        for _ in 0..5 {
            tracker.push(Vec2::new(10.0, -4.0), 0.016);
        }
        let velocity = tracker.release_velocity();
        assert!((velocity.x - 10.0 / 0.016).abs() < 1.0);
        assert!((velocity.y + 4.0 / 0.016).abs() < 1.0);
    }

    #[test]
    fn drag_velocity_tracker_excludes_glitch_frames() {
        let mut tracker = DragVelocityTracker::default();
        for _ in 0..4 {
            tracker.push(Vec2::new(10.0, 0.0), 0.016);
        }
        // One frame claiming ~300k px/s (a WSLg coordinate-space jump) must
        // not poison the estimate.
        tracker.push(Vec2::new(5_000.0, 0.0), 0.016);
        let velocity = tracker.release_velocity();
        assert!((velocity.x - 10.0 / 0.016).abs() < 1.0);

        // A drag consisting ONLY of glitch frames produces no fling at all.
        let mut tracker = DragVelocityTracker::default();
        tracker.push(Vec2::new(5_000.0, 0.0), 0.016);
        assert_eq!(tracker.release_velocity(), Vec2::ZERO);
    }

    #[test]
    fn drag_velocity_tracker_forgets_stale_motion() {
        let mut tracker = DragVelocityTracker::default();
        // A fast start followed by more than the window's worth of standing
        // still must not fling: only the trailing still frames survive.
        tracker.push(Vec2::new(50.0, 0.0), 0.016);
        for _ in 0..10 {
            tracker.push(Vec2::ZERO, 0.016);
        }
        assert_eq!(tracker.release_velocity(), Vec2::ZERO);
    }

    #[test]
    fn drag_velocity_tracker_rejects_degenerate_samples_and_clears() {
        let mut tracker = DragVelocityTracker::default();
        tracker.push(Vec2::new(f32::NAN, 0.0), 0.016);
        tracker.push(Vec2::new(10.0, 0.0), 0.0);
        tracker.push(Vec2::new(10.0, 0.0), f32::NAN);
        assert_eq!(tracker.release_velocity(), Vec2::ZERO);

        tracker.push(Vec2::new(10.0, 0.0), 0.016);
        assert!(tracker.release_velocity().x > 0.0);
        tracker.clear();
        assert_eq!(tracker.release_velocity(), Vec2::ZERO);
    }

    #[test]
    fn new_falls_back_to_a_valid_view_on_zero_size() {
        let panel = MapPanelState::new(139.7, 35.7, 9.0, [0.0, 0.0]);
        assert!(panel.view().size_px()[0] >= 1.0);
        assert!(panel.view().size_px()[1] >= 1.0);
    }

    #[test]
    fn new_falls_back_to_a_valid_view_on_non_finite_zoom() {
        let panel = MapPanelState::new(0.0, 0.0, f64::NAN, [800.0, 600.0]);
        assert!(panel.view().zoom().is_finite());
    }

    #[test]
    fn default_view_is_the_documented_starting_camera() {
        let panel = MapPanelState::default();
        assert_eq!(panel.view().center(), LonLat::new(0.0, 0.0));
        assert!((panel.view().zoom() - DEFAULT_ZOOM).abs() < 1e-9);
    }

    #[test]
    fn set_view_replaces_the_camera() {
        let mut panel = MapPanelState::default();
        let Ok(replacement) = MapView::new(LonLat::new(10.0, 20.0), 5.0, [640.0, 480.0]) else {
            panic!("view construction failed");
        };
        panel.set_view(replacement);
        assert_eq!(panel.view(), replacement);
    }

    #[test]
    fn allocate_updates_size_without_simulated_input() {
        egui::__run_test_ui(|ui| {
            let mut panel = MapPanelState::new(0.0, 0.0, 2.0, [400.0, 300.0]);
            let before = panel.view().center();
            let (rect, _response) = panel.allocate(ui);
            assert!(rect.width() > 0.0);
            // Simulated headless input has no drag in progress, so the
            // center must be unchanged by `allocate` alone (pan/zoom
            // transitions themselves are covered directly by the
            // `apply_pan_*`/`apply_zoom_*` tests above).
            assert_eq!(panel.view().center(), before);
        });
    }

    #[test]
    fn apply_pan_moves_the_center_west_when_dragging_right() {
        let mut panel = MapPanelState::new(0.0, 0.0, 4.0, [512.0, 512.0]);
        let before = panel.view().center();
        // Dragging the map to the right (positive x) reveals territory to
        // the west, so the camera center's longitude must decrease.
        panel.apply_pan([100.0, 0.0]);
        assert!(panel.view().center().lon < before.lon);
    }

    #[test]
    fn apply_pan_moves_the_center_north_when_dragging_down() {
        let mut panel = MapPanelState::new(0.0, 0.0, 4.0, [512.0, 512.0]);
        let before = panel.view().center();
        // Dragging the map down (positive y) reveals territory to the
        // north, so the camera center's latitude must increase.
        panel.apply_pan([0.0, 100.0]);
        assert!(panel.view().center().lat > before.lat);
    }

    #[test]
    fn apply_pan_with_zero_delta_is_a_no_op() {
        let mut panel = MapPanelState::new(10.0, 20.0, 4.0, [512.0, 512.0]);
        let before = panel.view().center();
        panel.apply_pan([0.0, 0.0]);
        assert!((panel.view().center().lon - before.lon).abs() < 1e-9);
        assert!((panel.view().center().lat - before.lat).abs() < 1e-9);
    }

    #[test]
    fn scroll_units_convert_to_zoom_levels() {
        assert!((zoom_levels(SCROLL_UNITS_PER_ZOOM_LEVEL, 1.0) - 1.0).abs() < 1e-12);
        assert!((zoom_levels(-SCROLL_UNITS_PER_ZOOM_LEVEL, 1.0) + 1.0).abs() < 1e-12);
        assert!(zoom_levels(0.0, 1.0).abs() < 1e-12);
    }

    #[test]
    fn pinch_multiplier_converts_to_zoom_levels() {
        // A pinch that doubles the scale is exactly one zoom level in.
        assert!((zoom_levels(0.0, 2.0) - 1.0).abs() < 1e-9);
        assert!((zoom_levels(0.0, 0.5) + 1.0).abs() < 1e-9);
        // Non-finite / non-positive inputs contribute nothing rather than NaN.
        assert!(zoom_levels(f32::NAN, 1.0).abs() < 1e-12);
        assert!(zoom_levels(0.0, 0.0).abs() < 1e-12);
        assert!(zoom_levels(0.0, f32::NAN).abs() < 1e-12);
    }

    #[test]
    fn scroll_and_pinch_contributions_add() {
        let combined = zoom_levels(SCROLL_UNITS_PER_ZOOM_LEVEL, 2.0);
        assert!((combined - 2.0).abs() < 1e-9);
    }

    #[test]
    fn apply_zoom_scrolling_up_increases_zoom_by_about_one_level() {
        let mut panel = MapPanelState::new(0.0, 0.0, 4.0, [512.0, 512.0]);
        let before = panel.view().zoom();
        panel.apply_zoom_at(zoom_levels(SCROLL_UNITS_PER_ZOOM_LEVEL, 1.0), None);
        assert!((panel.view().zoom() - (before + 1.0)).abs() < 1e-9);
    }

    #[test]
    fn apply_zoom_scrolling_down_decreases_zoom() {
        let mut panel = MapPanelState::new(0.0, 0.0, 4.0, [512.0, 512.0]);
        let before = panel.view().zoom();
        panel.apply_zoom_at(zoom_levels(-SCROLL_UNITS_PER_ZOOM_LEVEL, 1.0), None);
        assert!(panel.view().zoom() < before);
    }

    #[test]
    fn apply_zoom_about_the_center_leaves_the_center_put() {
        let mut panel = MapPanelState::new(12.0, 34.0, 4.0, [512.0, 512.0]);
        let before = panel.view().center();
        panel.apply_zoom_at(1.0, None);
        assert!((panel.view().center().lon - before.lon).abs() < 1e-9);
        assert!((panel.view().center().lat - before.lat).abs() < 1e-9);
    }

    #[test]
    fn apply_zoom_clamps_at_the_top_of_the_valid_range() {
        let mut panel =
            MapPanelState::new(0.0, 0.0, f64::from(oxigis_render::MAX_ZOOM), [512.0, 512.0]);
        panel.apply_zoom_at(zoom_levels(SCROLL_UNITS_PER_ZOOM_LEVEL * 10.0, 1.0), None);
        assert!((panel.view().zoom() - f64::from(oxigis_render::MAX_ZOOM)).abs() < 1e-9);
    }

    #[test]
    fn apply_zoom_clamps_at_the_bottom_of_the_valid_range() {
        let mut panel = MapPanelState::new(0.0, 0.0, 0.5, [512.0, 512.0]);
        panel.apply_zoom_at(-10.0, None);
        assert!(panel.view().zoom().abs() < 1e-9);
    }

    #[test]
    fn apply_zoom_below_epsilon_is_a_no_op() {
        let mut panel = MapPanelState::new(0.0, 0.0, 4.0, [512.0, 512.0]);
        let before = panel.view().zoom();
        panel.apply_zoom_at(0.0, None);
        assert!((panel.view().zoom() - before).abs() < 1e-12);
        panel.apply_zoom_at(f64::NAN, None);
        assert!((panel.view().zoom() - before).abs() < 1e-12);
    }

    #[test]
    fn apply_zoom_at_pins_the_point_under_the_cursor() {
        // The invariant that matters: whatever was under the cursor before the
        // zoom is still under it afterwards (a sign error in the anchor
        // correction sends it the other way by twice the offset).
        let cursor = [96.0_f32, 400.0_f32];
        let anchor = MapPanelState::new(10.0, 20.0, 6.0, [512.0, 512.0])
            .view()
            .screen_to_lon_lat(cursor);

        for levels in [1.0_f64, -1.0, 0.375] {
            let mut panel = MapPanelState::new(10.0, 20.0, 6.0, [512.0, 512.0]);
            panel.apply_zoom_at(levels, Some(cursor));
            let after = panel.view().lon_lat_to_screen(anchor);
            assert!(
                (after[0] - cursor[0]).abs() < 0.05 && (after[1] - cursor[1]).abs() < 0.05,
                "anchor drifted to {after:?} after {levels} levels"
            );
        }

        // And the camera really did move (i.e. the loop above is not passing
        // because nothing happened).
        let mut panel = MapPanelState::new(10.0, 20.0, 6.0, [512.0, 512.0]);
        let before = panel.view().center();
        panel.apply_zoom_at(1.0, Some(cursor));
        assert!((panel.view().center().lon - before.lon).abs() > 1e-6);
        assert!((panel.view().center().lat - before.lat).abs() > 1e-6);
    }

    #[test]
    fn apply_zoom_at_the_viewport_center_does_not_move_the_camera() {
        let mut panel = MapPanelState::new(10.0, 20.0, 6.0, [512.0, 512.0]);
        let before = panel.view().center();
        panel.apply_zoom_at(1.0, Some([256.0, 256.0]));
        assert!((panel.view().center().lon - before.lon).abs() < 1e-6);
        assert!((panel.view().center().lat - before.lat).abs() < 1e-6);
    }

    #[test]
    fn allocate_matches_the_view_size_to_the_rect_in_physical_pixels() {
        // The geometry contract `crate::map_gpu` depends on: NDC -1..1 covers
        // the callback rect only if `size_px == rect.size() * ppp`.
        egui::__run_test_ui(|ui| {
            let mut panel = MapPanelState::new(0.0, 0.0, 2.0, [400.0, 300.0]);
            let (rect, _response) = panel.allocate(ui);
            let ppp = ui.ctx().pixels_per_point();
            let size = panel.view().size_px();
            assert!((size[0] - rect.width() * ppp).abs() < 1e-3);
            assert!((size[1] - rect.height() * ppp).abs() < 1e-3);
        });
    }

    #[test]
    fn paint_fallback_does_not_panic_on_a_typical_rect() {
        egui::__run_test_ui(|ui| {
            let panel = MapPanelState::new(139.7, 35.7, 5.0, [512.0, 512.0]);
            let rect = Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(512.0, 512.0));
            panel.paint_fallback(ui, rect);
        });
    }

    #[test]
    fn the_input_diagnostic_is_off_until_a_shell_asks_for_it() {
        let ctx = egui::Context::default();
        let mut panel = MapPanelState::new(0.0, 0.0, 3.0, [600.0, 400.0]);
        assert!(!panel.input_diagnostics());
        // A frame with a button held is exactly the state that used to paint
        // the orange line over every user's map.
        let held = vec![egui::Event::PointerButton {
            pos: egui::Pos2::new(100.0, 100.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        }];
        let _ = gate_frame(&ctx, &mut panel, PanGate::Allow, held.clone());
        assert!(panel.input_diag.is_none(), "no diagnostic while off");

        panel.set_input_diagnostics(true);
        let _ = gate_frame(&ctx, &mut panel, PanGate::Allow, held);
        assert!(
            panel.input_diag.is_some(),
            "the mechanism must still be reachable for the bug it exists to diagnose"
        );
        // Turning it back off clears the line the same frame.
        panel.set_input_diagnostics(false);
        assert!(panel.input_diag.is_none());
    }

    #[test]
    fn the_fallback_note_names_the_install_failure_when_there_was_one() {
        let mut panel = MapPanelState::new(0.0, 0.0, 3.0, [512.0, 512.0]);
        assert!(panel.fallback_reason.is_none());
        panel.set_fallback_reason(Some("surface format Rgba8Unorm is unsupported"));
        assert_eq!(
            panel.fallback_reason.as_deref(),
            Some("surface format Rgba8Unorm is unsupported"),
            "the default note asserts there was no render state, which is false here"
        );
        panel.set_fallback_reason(None);
        assert!(panel.fallback_reason.is_none());
        // Painting either shape is harmless in a headless context.
        egui::__run_test_ui(|ui| {
            let rect = Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(512.0, 512.0));
            panel.paint_fallback(ui, rect);
            panel.set_fallback_reason(Some("boom"));
            panel.paint_fallback(ui, rect);
        });
    }

    #[test]
    fn paint_gpu_pushes_a_callback_without_a_gpu() {
        // No wgpu renderer is listening in a headless context; pushing the
        // callback shape must still be harmless.
        egui::__run_test_ui(|ui| {
            let panel = MapPanelState::new(139.7, 35.7, 5.0, [512.0, 512.0]);
            let rect = Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(512.0, 512.0));
            panel.paint_gpu(ui, rect);
        });
    }

    #[test]
    fn ui_selects_a_paint_path_without_panicking() {
        for gpu in [true, false] {
            egui::__run_test_ui(|ui| {
                let mut panel = MapPanelState::new(0.0, 0.0, 3.0, [512.0, 512.0]);
                let _response = panel.ui(ui, gpu);
            });
        }
    }

    /// The screen the gate tests lay out in.
    fn gate_screen() -> Rect {
        Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(600.0, 400.0))
    }

    /// Runs one frame through `allocate_gated` with `gate` and `events`,
    /// returning the panel rect.
    fn gate_frame(
        ctx: &egui::Context,
        panel: &mut MapPanelState,
        gate: PanGate,
        events: Vec<egui::Event>,
    ) -> Rect {
        let raw_input = egui::RawInput {
            screen_rect: Some(gate_screen()),
            events,
            ..Default::default()
        };
        let mut rect = gate_screen();
        let _output = ctx.run_ui(raw_input, |ui| {
            let (allocated, _response) = panel.allocate_gated(ui, |_rect, _response, _ppp| gate);
            rect = allocated;
        });
        rect
    }

    /// Press at `from`, drag to `to`, release — four frames, the first of which
    /// only registers the widget so egui has something to hit-test against.
    ///
    /// egui only calls a press-then-move a *drag* once the pointer has left the
    /// click radius **on a later frame than the press**, which is why the press
    /// and the motion cannot share one frame.
    fn drag_gesture(panel: &mut MapPanelState, gate: PanGate, from: egui::Pos2, to: egui::Pos2) {
        let ctx = egui::Context::default();
        gate_frame(&ctx, panel, gate, Vec::new());
        gate_frame(
            &ctx,
            panel,
            gate,
            vec![
                egui::Event::PointerMoved(from),
                egui::Event::PointerButton {
                    pos: from,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
        );
        gate_frame(&ctx, panel, gate, vec![egui::Event::PointerMoved(to)]);
        gate_frame(
            &ctx,
            panel,
            gate,
            vec![egui::Event::PointerButton {
                pos: to,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        );
    }

    #[test]
    fn allocate_gated_suppressed_does_not_move_the_camera_on_a_primary_drag() {
        // The control: an allowed drag really does pan, so the suppressed case
        // below is not passing because nothing happened.
        let mut allowed = MapPanelState::new(10.0, 20.0, 5.0, [600.0, 400.0]);
        let before = allowed.view().center();
        drag_gesture(
            &mut allowed,
            PanGate::Allow,
            egui::pos2(300.0, 200.0),
            egui::pos2(180.0, 260.0),
        );
        assert!(
            (allowed.view().center().lon - before.lon).abs() > 1e-6,
            "the harness must produce a real drag"
        );

        let mut suppressed = MapPanelState::new(10.0, 20.0, 5.0, [600.0, 400.0]);
        drag_gesture(
            &mut suppressed,
            PanGate::Suppress,
            egui::pos2(300.0, 200.0),
            egui::pos2(180.0, 260.0),
        );
        assert!((suppressed.view().center().lon - before.lon).abs() < 1e-12);
        assert!((suppressed.view().center().lat - before.lat).abs() < 1e-12);
    }

    #[test]
    fn allocate_gated_suppressed_accumulates_no_fling_velocity_and_cancels_inertia() {
        let mut panel = MapPanelState::new(10.0, 20.0, 5.0, [600.0, 400.0]);
        // A glide is already in flight when the gesture starts.
        panel.inertia.start(Vec2::new(2_000.0, 0.0));
        assert!(panel.inertia.is_active());
        let before = panel.view().center();

        drag_gesture(
            &mut panel,
            PanGate::Suppress,
            egui::pos2(300.0, 200.0),
            egui::pos2(120.0, 200.0),
        );

        assert!(
            !panel.inertia.is_active(),
            "a suppressed drag must cancel any live inertia"
        );
        assert_eq!(
            panel.drag_velocity.release_velocity(),
            Vec2::ZERO,
            "no drag frame may be recorded as fling velocity"
        );
        assert!((panel.view().center().lon - before.lon).abs() < 1e-12);

        // And no glide starts afterwards either: further idle frames move
        // nothing.
        let ctx = egui::Context::default();
        for _ in 0..5 {
            gate_frame(&ctx, &mut panel, PanGate::Suppress, Vec::new());
        }
        assert!((panel.view().center().lon - before.lon).abs() < 1e-12);
    }

    #[test]
    fn allocate_gated_suppressed_still_applies_wheel_zoom() {
        let mut panel = MapPanelState::new(10.0, 20.0, 5.0, [600.0, 400.0]);
        let ctx = egui::Context::default();
        let center = gate_screen().center();
        // One frame to register the rect, then hover it and pinch by 2x — one
        // whole zoom level.
        gate_frame(&ctx, &mut panel, PanGate::Suppress, Vec::new());
        gate_frame(
            &ctx,
            &mut panel,
            PanGate::Suppress,
            vec![egui::Event::PointerMoved(center), egui::Event::Zoom(2.0)],
        );
        assert!(
            (panel.view().zoom() - 6.0).abs() < 1e-6,
            "zoom is never gated, got {}",
            panel.view().zoom()
        );
    }

    #[test]
    fn allocate_delegates_with_allow_and_is_behaviourally_unchanged() {
        // Same gesture through `allocate` and through `allocate_gated` with an
        // `Allow` gate must land the camera in exactly the same place.
        let mut gated = MapPanelState::new(10.0, 20.0, 5.0, [600.0, 400.0]);
        drag_gesture(
            &mut gated,
            PanGate::Allow,
            egui::pos2(300.0, 200.0),
            egui::pos2(180.0, 260.0),
        );

        let mut plain = MapPanelState::new(10.0, 20.0, 5.0, [600.0, 400.0]);
        let ctx = egui::Context::default();
        let frames: Vec<Vec<egui::Event>> = vec![
            Vec::new(),
            vec![
                egui::Event::PointerMoved(egui::pos2(300.0, 200.0)),
                egui::Event::PointerButton {
                    pos: egui::pos2(300.0, 200.0),
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::NONE,
                },
            ],
            vec![egui::Event::PointerMoved(egui::pos2(180.0, 260.0))],
            vec![egui::Event::PointerButton {
                pos: egui::pos2(180.0, 260.0),
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            }],
        ];
        for events in frames {
            let raw_input = egui::RawInput {
                screen_rect: Some(gate_screen()),
                events,
                ..Default::default()
            };
            let _output = ctx.run_ui(raw_input, |ui| {
                let _allocated = plain.allocate(ui);
            });
        }

        assert_eq!(plain.view().center(), gated.view().center());
        assert!((plain.view().zoom() - gated.view().zoom()).abs() < 1e-12);
        assert_eq!(
            plain.inertia.is_active(),
            gated.inertia.is_active(),
            "the fling handoff must be identical too"
        );
    }

    #[test]
    fn tile_color_is_deterministic() {
        let Ok(tile) = TileId::new(3, 1, 2) else {
            panic!("tile construction failed");
        };
        assert_eq!(tile_color(tile), tile_color(tile));
    }

    #[test]
    fn scalebar_is_on_by_default_and_the_toggle_silences_it() {
        let mut panel = MapPanelState::new(139.7, 35.7, 12.0, [1024.0, 768.0]);
        assert!(panel.scale_bar_visible(), "a map ships with a scale");
        assert!(panel.scale_bar(800.0, 1.0).is_some());
        panel.set_scale_bar_visible(false);
        assert!(!panel.scale_bar_visible());
        assert_eq!(panel.scale_bar(800.0, 1.0), None);
        panel.set_scale_bar_visible(true);
        assert!(panel.scale_bar(800.0, 1.0).is_some());
    }

    #[test]
    fn painting_the_scale_bar_does_not_panic_on_either_path_or_a_tiny_rect() {
        egui::__run_test_ui(|ui| {
            let panel = MapPanelState::new(139.7, 35.7, 10.0, [512.0, 512.0]);
            let painter = ui.painter();
            for size in [
                Vec2::new(1024.0, 768.0),
                Vec2::new(60.0, 40.0),
                Vec2::new(1.0, 1.0),
            ] {
                let rect = Rect::from_min_size(egui::Pos2::ZERO, size);
                panel.paint_scale_bar(painter, rect, 1.0, 0.0);
                panel.paint_scale_bar(painter, rect, 2.0, FALLBACK_NOTE_ROW);
            }
            // The fallback path draws the bar too, and must survive a panel
            // whose corner it also writes a note into.
            let rect = Rect::from_min_size(egui::Pos2::ZERO, Vec2::new(512.0, 512.0));
            panel.paint_fallback(ui, rect);
        });
    }
}
