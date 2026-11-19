// SPDX-License-Identifier: GPL-3.0-or-later
//! 3D View (stage 1): the active document shown as a flat sheet floating in 3D, seen through
//! an orbit camera. It only DISPLAYS the document; nothing here changes the document.
//!
//! The orbit-camera model and the ray / drawing-plane intersection follow the approach of
//! Sketch3D (https://github.com/remarcus/sketch3d, GPL-3.0-or-later, src/geometry.rs).
//! Because of that this file is GPL-3.0-or-later; a build that includes it must be
//! distributed under the GPL with its source.
//!
//! Open it from the View menu: View > 3D View (it shows a check mark while it is open).
//!   - one finger / left mouse drag: orbit
//!   - two fingers: pan and pinch-zoom (mouse: right drag pans, wheel zooms)
//!
//! Stage 2 will use [`Cam::screen_to_doc`] to turn a pointer position into a point on the
//! sheet, so the normal brush pipeline can paint through this view.

use crate::PhotocraftApp;
use egui::{Color32, Pos2, Rect, Sense, pos2, vec2};
use std::sync::atomic::{AtomicBool, Ordering};

/// Vertical field of view (45 degrees).
const FOV_Y: f32 = std::f32::consts::FRAC_PI_4;
/// The sheet is drawn as a GRID x GRID mesh so the perspective is right across it.
const GRID: usize = 24;
/// Points closer to the camera than this are not drawn.
const NEAR: f32 = 1.0;
/// Keeps the camera from looking exactly along the up axis.
const MAX_PITCH: f32 = 1.5;

type V3 = [f32; 3];

fn dot(a: V3, b: V3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: V3, k: f32) -> V3 {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn normalize(a: V3) -> V3 {
    let len = dot(a, a).sqrt();
    if len > 1e-9 { scale(a, 1.0 / len) } else { [0.0, 0.0, 1.0] }
}

/// The document size and the viewport size, both in pixels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Vp {
    pub doc_w: f32,
    pub doc_h: f32,
    pub w: f32,
    pub h: f32,
}

/// An orbit camera looking at a point on the sheet. The sheet lies in the world plane z = 0,
/// x to the right and y up, centred on the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cam {
    /// Radians; positive moves the camera to the right.
    pub yaw: f32,
    /// Radians; positive moves the camera up.
    pub pitch: f32,
    /// Distance multiplier relative to "the whole sheet fits".
    pub zoom: f32,
    /// The point on the sheet (world x, y) the camera looks at.
    pub target: [f32; 2],
}

impl Default for Cam {
    fn default() -> Self {
        Self { yaw: 0.0, pitch: 0.0, zoom: 1.0, target: [0.0, 0.0] }
    }
}

struct Frame {
    eye: V3,
    forward: V3,
    right: V3,
    up: V3,
}

impl Cam {
    fn focal(vp: Vp) -> f32 {
        (vp.h * 0.5) / (FOV_Y * 0.5).tan()
    }

    /// Distance from the target to the camera.
    fn distance(&self, vp: Vp) -> f32 {
        let half = (vp.doc_h * 0.5).max(vp.doc_w * 0.5 * vp.h / vp.w.max(1.0));
        (half / (FOV_Y * 0.5).tan() * 1.15 * self.zoom).max(NEAR * 2.0)
    }

    fn frame(&self, vp: Vp) -> Frame {
        let toward_eye = [self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), self.pitch.cos() * self.yaw.cos()];
        let target = [self.target[0], self.target[1], 0.0];
        let eye = add(target, scale(toward_eye, self.distance(vp)));
        let forward = scale(toward_eye, -1.0);
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        Frame { eye, forward, right, up }
    }

    /// Document pixel (x right, y down) to a world point on the sheet.
    pub fn doc_to_world(&self, vp: Vp, x: f32, y: f32) -> V3 {
        [x - vp.doc_w * 0.5, vp.doc_h * 0.5 - y, 0.0]
    }

    /// A world point to a viewport position in pixels from the viewport's top-left, or `None`
    /// when the point is behind the camera.
    pub fn project(&self, vp: Vp, p: V3) -> Option<[f32; 2]> {
        let f = self.frame(vp);
        let v = sub(p, f.eye);
        let depth = dot(v, f.forward);
        if depth <= NEAR {
            return None;
        }
        let focal = Self::focal(vp);
        Some([vp.w * 0.5 + dot(v, f.right) / depth * focal, vp.h * 0.5 - dot(v, f.up) / depth * focal])
    }

    /// A viewport position (pixels from the top-left) to the document pixel under it, by
    /// intersecting the camera ray with the sheet. `None` when the ray misses the sheet's
    /// plane. The result can lie outside the document's bounds.
    #[allow(dead_code)]
    pub fn screen_to_doc(&self, vp: Vp, s: [f32; 2]) -> Option<[f32; 2]> {
        let f = self.frame(vp);
        let focal = Self::focal(vp);
        let x = (s[0] - vp.w * 0.5) / focal;
        let y = -(s[1] - vp.h * 0.5) / focal;
        let ray = add(f.forward, add(scale(f.right, x), scale(f.up, y)));
        if ray[2].abs() < 1e-6 {
            return None;
        }
        let t = -f.eye[2] / ray[2];
        if t <= 0.0 {
            return None;
        }
        let hit = add(f.eye, scale(ray, t));
        Some([hit[0] + vp.doc_w * 0.5, vp.doc_h * 0.5 - hit[1]])
    }

    /// Slide the view by a screen drag of `d` pixels.
    fn pan(&mut self, vp: Vp, d: egui::Vec2) {
        let f = self.frame(vp);
        let per_pixel = self.distance(vp) / Self::focal(vp);
        self.target[0] += (-f.right[0] * d.x + f.up[0] * d.y) * per_pixel;
        self.target[1] += (-f.right[1] * d.x + f.up[1] * d.y) * per_pixel;
    }

    fn orbit(&mut self, d: egui::Vec2) {
        self.yaw -= d.x * 0.008;
        self.pitch = (self.pitch + d.y * 0.008).clamp(-MAX_PITCH, MAX_PITCH);
    }

    fn zoom_by(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.zoom = (self.zoom / factor).clamp(0.1, 10.0);
        }
    }
}

/// Whether the 3D View window is open. A plain flag so the View menu can read and flip it.
static OPEN: AtomicBool = AtomicBool::new(false);

/// For the View > 3D View check mark.
pub fn is_open() -> bool {
    OPEN.load(Ordering::Relaxed)
}

/// For the View > 3D View command.
pub fn toggle() {
    OPEN.fetch_xor(true, Ordering::Relaxed);
}

fn cam_id() -> egui::Id {
    egui::Id::unique("photocraft-3d-view-cam")
}

/// Called once per frame by the app: draws the window while it is open.
pub fn show(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if !is_open() {
        return;
    }
    let mut cam: Cam = ctx.data(|d| d.get_temp(cam_id())).unwrap_or_default();
    let mut open = true;
    egui::Window::new("3D View").id(cam_id().with("window")).open(&mut open).default_size(vec2(520.0, 460.0)).resizable(true).show(ctx, |ui| {
        body(app, ui, &mut cam);
    });
    ctx.data_mut(|d| d.insert_temp(cam_id(), cam));
    if !open {
        OPEN.store(false, Ordering::Relaxed);
    }
}

fn to_screen(frame: Rect, p: [f32; 2]) -> Pos2 {
    pos2(frame.min.x + p[0], frame.min.y + p[1])
}

fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, cam: &mut Cam) {
    let Some(idx) = app.session.active_index() else {
        ui.label("No document open");
        return;
    };
    let size = app.session.documents()[idx].doc.size;
    ui.horizontal(|ui| {
        if ui.button("Reset").clicked() {
            *cam = Cam::default();
        }
        ui.label("Drag: orbit. Two fingers: pan and zoom.");
    });
    let avail = ui.available_size();
    let vw = if avail.x.is_finite() { avail.x.clamp(120.0, 4096.0) } else { 480.0 };
    let vh = if avail.y.is_finite() { avail.y.clamp(120.0, 4096.0) } else { 360.0 };
    let (frame, resp) = ui.allocate_exact_size(vec2(vw, vh), Sense::click_and_drag());
    let vp = Vp { doc_w: size.width.max(1) as f32, doc_h: size.height.max(1) as f32, w: frame.width(), h: frame.height() };

    let touch = ui.input(|i| i.multi_touch().map(|m| (m.translation_delta, m.zoom_delta)));
    if let Some((translation, pinch)) = touch {
        cam.pan(vp, translation);
        cam.zoom_by(pinch);
    } else if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
        cam.pan(vp, resp.drag_delta());
    } else if resp.dragged() {
        cam.orbit(resp.drag_delta());
    }
    if touch.is_none() && resp.hovered() {
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        cam.zoom_by((scroll * 0.002).exp());
        cam.zoom_by(pinch);
    }

    let painter = ui.painter_at(frame);
    painter.rect_filled(frame, 0.0, Color32::from_gray(28));
    let ctx = ui.ctx().clone();
    let Some(tex) = crate::canvas::navigator_texture(app, &ctx, idx) else {
        painter.text(frame.center(), egui::Align2::CENTER_CENTER, "Preparing...", egui::FontId::proportional(14.0), Color32::from_gray(180));
        return;
    };

    let mut mesh = egui::Mesh::with_texture(tex);
    let mut visible = vec![true; (GRID + 1) * (GRID + 1)];
    for j in 0..=GRID {
        for i in 0..=GRID {
            let (s, t) = (i as f32 / GRID as f32, j as f32 / GRID as f32);
            let world = cam.doc_to_world(vp, s * vp.doc_w, t * vp.doc_h);
            let pos = match cam.project(vp, world) {
                Some(p) => to_screen(frame, p),
                None => {
                    visible[j * (GRID + 1) + i] = false;
                    frame.center()
                }
            };
            mesh.vertices.push(egui::epaint::Vertex { pos, uv: pos2(s, t), color: Color32::WHITE });
        }
    }
    for j in 0..GRID {
        for i in 0..GRID {
            let a = j * (GRID + 1) + i;
            let (b, c, d) = (a + 1, a + GRID + 2, a + GRID + 1);
            if visible[a] && visible[b] && visible[c] && visible[d] {
                mesh.add_triangle(a as u32, b as u32, c as u32);
                mesh.add_triangle(a as u32, c as u32, d as u32);
            }
        }
    }
    painter.add(egui::Shape::mesh(mesh));

    let corners: Vec<Pos2> = [(0.0, 0.0), (vp.doc_w, 0.0), (vp.doc_w, vp.doc_h), (0.0, vp.doc_h)]
        .iter()
        .filter_map(|&(x, y)| cam.project(vp, cam.doc_to_world(vp, x, y)).map(|p| to_screen(frame, p)))
        .collect();
    if corners.len() == 4 {
        painter.add(egui::Shape::closed_line(corners, egui::Stroke::new(1.5, Color32::from_gray(200))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VP: Vp = Vp { doc_w: 800.0, doc_h: 600.0, w: 500.0, h: 400.0 };

    #[test]
    fn front_view_puts_the_document_centre_at_the_viewport_centre() {
        let cam = Cam::default();
        let p = cam.project(VP, cam.doc_to_world(VP, 400.0, 300.0));
        let p = p.unwrap_or([f32::NAN; 2]);
        assert!((p[0] - 250.0).abs() < 1e-3 && (p[1] - 200.0).abs() < 1e-3, "{p:?}");
    }

    #[test]
    fn document_top_is_above_the_bottom_on_screen() {
        let cam = Cam::default();
        let top = cam.project(VP, cam.doc_to_world(VP, 400.0, 0.0)).unwrap_or([0.0, f32::MAX]);
        let bottom = cam.project(VP, cam.doc_to_world(VP, 400.0, 600.0)).unwrap_or([0.0, f32::MIN]);
        assert!(top[1] < bottom[1]);
    }

    #[test]
    fn a_screen_point_round_trips_to_the_same_document_point() {
        for (yaw, pitch, zoom) in [(0.0, 0.0, 1.0), (0.7, 0.4, 0.8), (-1.1, 0.9, 2.0), (0.3, -0.8, 1.3)] {
            let cam = Cam { yaw, pitch, zoom, target: [20.0, -15.0] };
            for (x, y) in [(10.0, 20.0), (400.0, 300.0), (790.0, 590.0), (123.0, 456.0)] {
                let Some(s) = cam.project(VP, cam.doc_to_world(VP, x, y)) else { continue };
                let Some(d) = cam.screen_to_doc(VP, s) else { panic_free_fail() };
                assert!((d[0] - x).abs() < 0.05 && (d[1] - y).abs() < 0.05, "{yaw} {pitch}: {d:?} vs {x},{y}");
            }
        }
    }

    fn panic_free_fail() -> ! {
        unreachable!("a point in front of the camera must hit the sheet")
    }

    #[test]
    fn a_point_behind_the_camera_is_not_projected() {
        let cam = Cam::default();
        assert!(cam.project(VP, [0.0, 0.0, 1.0e7]).is_none());
    }

    #[test]
    fn orbit_clamps_pitch_and_zoom_is_bounded() {
        let mut cam = Cam::default();
        cam.orbit(vec2(0.0, 1.0e6));
        assert!(cam.pitch <= MAX_PITCH);
        cam.zoom_by(1.0e-9);
        assert!(cam.zoom <= 10.0);
        cam.zoom_by(1.0e9);
        assert!(cam.zoom >= 0.1);
    }
}
