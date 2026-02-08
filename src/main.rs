use eframe::egui;
use opencv::{core, imgproc, opencv_has_inherent_feature_algorithm_hint, prelude::*, videoio};
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, atomic};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SerializableRect {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct VideoRange {
    start_time: f64,
    end_time: f64,
    crop_rect_norm: Option<SerializableRect>,
    note: String,
}

enum PlayState {
    Playing,
    PlayingUntil(f64),
    NotPlaying,
}

struct VideoApp {
    input_folder: Option<PathBuf>,
    output_folder: Option<PathBuf>,
    videos: Vec<PathBuf>,
    selected_video_idx: Option<usize>,
    cap: Option<videoio::VideoCapture>,
    video_texture: Option<egui::TextureHandle>,
    current_time: f64,
    duration: f64,
    play_state: PlayState,
    native_fps: f64,
    ranges: Vec<VideoRange>,
    current_range_idx: usize,
    drag_start_norm: Option<egui::Pos2>,
    is_exporting: Arc<AtomicBool>,
    export_error: Arc<Mutex<Option<String>>>,
}

impl Default for VideoApp {
    fn default() -> Self {
        Self {
            input_folder: None,
            output_folder: None,
            videos: Vec::new(),
            selected_video_idx: None,
            cap: None,
            video_texture: None,
            current_time: 0.0,
            duration: 0.0,
            play_state: PlayState::NotPlaying,
            native_fps: 30.0,
            ranges: vec![VideoRange {
                start_time: 0.0,
                end_time: 0.0,
                crop_rect_norm: None,
                note: String::new(),
            }],
            current_range_idx: 0,
            drag_start_norm: None,
            is_exporting: Arc::new(AtomicBool::new(false)),
            export_error: Arc::new(Mutex::new(None)),
        }
    }
}

impl VideoApp {
    fn is_playing(&self) -> bool {
        match self.play_state {
            PlayState::Playing | PlayState::PlayingUntil(_) => true,
            _ => false,
        }
    }

    fn pause_play(&mut self) {
        self.play_state = match self.play_state {
            PlayState::NotPlaying => PlayState::Playing,
            PlayState::Playing => PlayState::NotPlaying,
            PlayState::PlayingUntil(_) => PlayState::NotPlaying, // TODO?
        };
    }

    fn prev_frame(&mut self, ctx: &egui::Context) {
        self.current_time -= 1.0 / self.native_fps;
        self.update_frame(ctx);
    }
    fn next_frame(&mut self, ctx: &egui::Context) {
        self.current_time += 1.0 / self.native_fps;
        self.update_frame(ctx);
    }

    fn update_frame(&mut self, ctx: &egui::Context) {
        if let Some(ref mut cap) = self.cap {
            let frame_pos = (self.current_time * self.native_fps) as i32;
            let _ = cap.set(videoio::CAP_PROP_POS_FRAMES, frame_pos as f64);
            let mut frame = core::Mat::default();
            if cap.read(&mut frame).unwrap_or(false) && !frame.empty() {
                let mut rgb_frame = core::Mat::default();

                opencv_has_inherent_feature_algorithm_hint! { {
                        let _ = imgproc::cvt_color(
                            &frame,
                            &mut rgb_frame,
                            imgproc::COLOR_BGR2RGB,
                            0,
                            core::AlgorithmHint::ALGO_HINT_DEFAULT,
                        );
                    } else {
                        let _ = imgproc::cvt_color(
                            &frame,
                            &mut rgb_frame,
                            imgproc::COLOR_BGR2RGB,
                            0
                        );
                    }
                }
                let size = rgb_frame.size().unwrap();
                let data = rgb_frame.data_bytes().unwrap();
                let color_image =
                    egui::ColorImage::from_rgb([size.width as usize, size.height as usize], data);
                self.video_texture =
                    Some(ctx.load_texture("video-frame", color_image, Default::default()));
            }
        }
    }

    fn run_export(&self) {
        let (Some(idx), Some(out_dir)) = (self.selected_video_idx, &self.output_folder) else {
            return;
        };
        let input_path = self.videos[idx].clone();
        let stem = input_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        // Clone data for the background thread
        let ranges = self.ranges.clone();
        let out_dir = out_dir.clone();

        // Get dimensions for crop math
        let (vid_w, vid_h) = if let Some(ref cap) = self.cap {
            (
                cap.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(1920.0),
                cap.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(1080.0),
            )
        } else {
            (1920.0, 1080.0)
        };

        self.is_exporting
            .store(true, std::sync::atomic::Ordering::SeqCst);
        *self.export_error.lock().unwrap() = None; // Clear previous errors

        let exp_err = self.export_error.clone();
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(false, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let guard = DropGuard(self.is_exporting.clone());
        std::thread::spawn(move || {
            let _guard = guard;

            for (i, range) in ranges.iter().enumerate() {
                let out_base = out_dir.join(format!("{}_range{}", stem, i));
                println!("DBG: {} vs {:?}", format!("{}_range{}", stem, i), out_base);

                // 1. Write the per-range note to its own text file
                if !range.note.is_empty() {
                    let _ = std::fs::write(out_base.with_added_extension("txt"), &range.note);
                }

                // 2. Setup FFmpeg
                let mut cmd = Command::new("ffmpeg");
                cmd.arg("-y")
                    .arg("-ss")
                    .arg(range.start_time.to_string())
                    .arg("-to")
                    .arg(range.end_time.to_string())
                    .arg("-i")
                    .arg(&input_path);

                let mut filters = vec!["fps=16".to_string()];
                if let Some(ref norm) = range.crop_rect_norm {
                    // Ensure dimensions are even numbers (requirement for many encoders)
                    let cw = ((norm.max_x - norm.min_x).abs() as f64 * vid_w) as i32 & !1;
                    let ch = ((norm.max_y - norm.min_y).abs() as f64 * vid_h) as i32 & !1;
                    let cx = (norm.min_x.min(norm.max_x) as f64 * vid_w) as i32;
                    let cy = (norm.min_y.min(norm.max_y) as f64 * vid_h) as i32;
                    filters.push(format!("crop={}:{}:{}:{}", cw, ch, cx, cy));
                }

                cmd.arg("-vf")
                    .arg(filters.join(","))
                    .arg("-c:v")
                    .arg("libx264")
                    .arg("-preset")
                    .arg("ultrafast")
                    .arg(out_base.with_added_extension("mp4"));

                println!(
                    "Exporting Range {}: {} to {}, file {:?}",
                    i,
                    range.start_time,
                    range.end_time,
                    out_base.with_added_extension("mp4")
                );

                match cmd.status() {
                    Ok(status) if !status.success() => {
                        let err_msg = format!(
                            "FFmpeg failed on range {} with exit code: {:?}",
                            i,
                            status.code()
                        );
                        *exp_err.lock().unwrap() = Some(err_msg);
                        break; // Stop exporting further ranges on error
                    }
                    Err(e) => {
                        *exp_err.lock().unwrap() = Some(format!("Failed to start FFmpeg: {}", e));
                        break;
                    }
                    _ => {} // Success
                }
            }
            println!("All exports finished.");
        });
    }
}

impl eframe::App for VideoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut video_to_load = None;

        // Keyboard Logic
        if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
            self.pause_play();
        }
        if !self.ranges.is_empty() {
            if ctx.input(|i| i.key_pressed(egui::Key::I)) {
                self.ranges[self.current_range_idx].start_time = self.current_time;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::O)) {
                self.ranges[self.current_range_idx].end_time = self.current_time;
            }
            if ctx.input(|i| i.key_pressed(egui::Key::R)) {
                let range = &self.ranges[self.current_range_idx];
                self.current_time = range.start_time;
                self.play_state = PlayState::PlayingUntil(range.end_time);
            }
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
            self.prev_frame(ctx);
        }
        if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
            self.next_frame(ctx);
        }

        // Panels
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📁 Input Folder").clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_folder() {
                        self.input_folder = Some(p.clone());
                        self.videos = std::fs::read_dir(p)
                            .unwrap()
                            .filter_map(|e| e.ok())
                            .map(|e| e.path())
                            .filter(|p| {
                                p.extension().map_or(false, |ext| {
                                    let ext = ext.to_ascii_lowercase();
                                    ext == "mp4" || ext == "mkv" || ext == "avi" || ext == "mov"
                                })
                            })
                            .collect();
                    }
                }
                ui.label(format!(
                    "In: {}",
                    self.input_folder
                        .as_deref()
                        .unwrap_or(Path::new("None"))
                        .display()
                ));
                ui.separator();
                if ui.button("💾 Output Folder").clicked() {
                    self.output_folder = rfd::FileDialog::new().pick_folder();
                }
                ui.label(format!(
                    "Out: {}",
                    self.output_folder
                        .as_deref()
                        .unwrap_or(Path::new("None"))
                        .display()
                ));
            });
        });

        egui::SidePanel::left("left")
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.heading("Videos");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.set_min_width(400.0);

                    for (i, v) in self.videos.iter().enumerate() {
                        let name = v.file_name().unwrap().to_string_lossy();
                        if ui
                            .selectable_label(self.selected_video_idx == Some(i), name)
                            .clicked()
                        {
                            video_to_load = Some(i);
                        }
                    }
                });
            });

        egui::SidePanel::right("right")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("Active Ranges");
                if ui.button("➕ Add Range").clicked() {
                    self.ranges.push(VideoRange {
                        start_time: self.current_time,
                        end_time: self.duration,
                        crop_rect_norm: None,
                        note: String::new(),
                    });
                    self.current_range_idx = self.ranges.len() - 1;
                }
                ui.separator();
                let mut to_remove = None;
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for i in 0..self.ranges.len() {
                        let range = &self.ranges[i];
                        let duration = range.end_time - range.start_time;
                        let frame_count_16fps = (duration * 16.0).round() as i32;

                        // Calculate frame indices based on native FPS for the second line
                        let start_frame = (range.start_time * self.native_fps).round() as i32;
                        let end_frame = (range.end_time * self.native_fps).round() as i32;

                        let label_text = format!(
                            "R{}: {:.1}s - {:.1}s ({:.1}s)\n      {} - {} ({} frames)",
                            i,
                            range.start_time,
                            range.end_time,
                            duration,
                            start_frame,
                            end_frame,
                            frame_count_16fps
                        );

                        let is_selected = self.current_range_idx == i;
                        ui.horizontal(|ui| {
                            let btn = egui::Button::selectable(is_selected, label_text)
                                .min_size(egui::vec2(ui.available_width() - 50.0, 45.0));

                            if ui.add(btn).clicked() {
                                self.current_range_idx = i;
                            }
                            if ui.button("❌").clicked() {
                                to_remove = Some(i);
                            }
                        });
                    }
                });
                if let Some(idx) = to_remove {
                    self.ranges.remove(idx);
                    self.current_range_idx = self
                        .current_range_idx
                        .clamp(0, self.ranges.len().saturating_sub(1));
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let mut avail_w = ui.available_width();
            let avail_h = ui.available_height();
            if (avail_h - avail_w * 0.56) < 200.0 {
                avail_w = (avail_h - 300.0) / 0.56;
            }

            let (rect, response) = ui.allocate_at_least(
                egui::vec2(avail_w, avail_w * 0.56),
                egui::Sense::click_and_drag(),
            );

            if let Some(tex) = &self.video_texture {
                ui.painter().image(
                    tex.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            } else {
                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
            }

            // Crop Handling
            let to_norm = |p: egui::Pos2| {
                egui::pos2(
                    (p.x - rect.min.x) / rect.width(),
                    (p.y - rect.min.y) / rect.height(),
                )
            };
            let from_norm = |p: egui::Pos2| {
                egui::pos2(
                    p.x * rect.width() + rect.min.x,
                    p.y * rect.height() + rect.min.y,
                )
            };

            if !self.ranges.is_empty() {
                if response.drag_started() {
                    self.drag_start_norm = response.interact_pointer_pos().map(to_norm);
                }
                if response.dragged() {
                    if let (Some(start), Some(now)) = (
                        self.drag_start_norm,
                        response.interact_pointer_pos().map(to_norm),
                    ) {
                        let r = egui::Rect::from_two_pos(start, now);
                        self.ranges[self.current_range_idx].crop_rect_norm =
                            Some(SerializableRect {
                                min_x: r.min.x,
                                min_y: r.min.y,
                                max_x: r.max.x,
                                max_y: r.max.y,
                            });
                    }
                }

                if let Some(ref norm) = self.ranges[self.current_range_idx].crop_rect_norm {
                    let screen_rect = egui::Rect::from_min_max(
                        from_norm(egui::pos2(norm.min_x, norm.min_y)),
                        from_norm(egui::pos2(norm.max_x, norm.max_y)),
                    );
                    ui.painter().rect_stroke(
                        screen_rect,
                        0.0,
                        egui::Stroke::new(2.0, egui::Color32::RED),
                        egui::StrokeKind::Outside,
                    );
                }
            }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Native Frame: {}",
                    (self.current_time * self.native_fps) as i32
                ));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("Target 16FPS: {:.1}", self.current_time * 16.0));
                });
            });

            ui.spacing_mut().slider_width = avail_w - 60.0;
            // 1. Draw the Slider first
            let slider_res = ui.add(
                egui::Slider::new(&mut self.current_time, 0.0..=self.duration)
                    .show_value(true)
                    .suffix("s"),
            );
            if slider_res.changed() {
                self.update_frame(ctx);
            }

            // 2. Draw Markers on top of the Slider
            if !self.ranges.is_empty() {
                let range = &self.ranges[self.current_range_idx];
                let rect = slider_res.rect;

                // Helper to turn video time into a horizontal screen coordinate
                let time_to_x = |time: f64| {
                    let pct = (time / self.duration) as f32;
                    rect.min.x + pct * rect.width()
                };

                let painter = ui.painter();
                let stroke_start = egui::Stroke::new(2.0, egui::Color32::GREEN);
                let stroke_end = egui::Stroke::new(2.0, egui::Color32::RED);

                // Draw Start Marker (Green line)
                if range.start_time > 0.0 {
                    let x = time_to_x(range.start_time);
                    painter.line_segment(
                        [egui::pos2(x, rect.min.y), egui::pos2(x, rect.max.y)],
                        stroke_start,
                    );
                }

                // Draw End Marker (Red line)
                if range.end_time < self.duration {
                    let x = time_to_x(range.end_time);
                    painter.line_segment(
                        [egui::pos2(x, rect.min.y), egui::pos2(x, rect.max.y)],
                        stroke_end,
                    );
                }

                // Optional: Draw a subtle highlight between them
                let start_x = time_to_x(range.start_time);
                let end_x = time_to_x(range.end_time);
                painter.rect_filled(
                    egui::Rect::from_min_max(
                        egui::pos2(start_x, rect.center().y - 2.0),
                        egui::pos2(end_x, rect.center().y + 2.0),
                    ),
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40), // Faint white glow
                );
            }

            ui.horizontal(|ui| {
                if ui.button("⏪").clicked() {
                    self.prev_frame(ctx);
                }
                if ui
                    .button(if self.is_playing() { "⏸" } else { "▶" })
                    .clicked()
                {
                    self.pause_play();
                }
                if ui.button("⏩").clicked() {
                    self.next_frame(ctx);
                }
                ui.separator();
                if !self.ranges.is_empty() {
                    if ui.button("Set Start").clicked() {
                        self.ranges[self.current_range_idx].start_time = self.current_time;
                    }
                    if ui.button("Set End").clicked() {
                        self.ranges[self.current_range_idx].end_time = self.current_time;
                    }
                    if ui.button("Clear Crop").clicked() {
                        self.ranges[self.current_range_idx].crop_rect_norm = None;
                    }
                    ui.separator();
                    if ui.add(egui::Button::new("🔁 Play Range (R)")).clicked() {
                        let range = &self.ranges[self.current_range_idx];
                        self.current_time = range.start_time;
                        self.play_state = PlayState::PlayingUntil(range.end_time);
                    }
                }
            });

            if !self.ranges.is_empty() {
                ui.add_space(10.0);
                ui.label(format!("Note for Range {}:", self.current_range_idx));
                // KEY: Bound specifically to the current range's note
                ui.add(
                    egui::TextEdit::multiline(&mut self.ranges[self.current_range_idx].note)
                        .desired_width(avail_w)
                        .desired_rows(5),
                );
            }

            ui.add_space(10.0);
            let exporting = self.is_exporting.load(atomic::Ordering::SeqCst);

            // Change button appearance based on state
            ui.add_enabled_ui(!exporting, |ui| {
                let btn_text = if exporting {
                    "⏳ Exporting..."
                } else {
                    "🚀 RUN EXPORT ALL"
                };
                if ui
                    .add_sized([avail_w, 40.0], egui::Button::new(btn_text))
                    .clicked()
                {
                    self.run_export();
                }
            });

            // Optional: Show a small spinner next to the button if exporting
            if exporting {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Processing ranges with FFmpeg...");
                });
            }

            // Error Dialog Window
            let mut err_guard = self.export_error.lock().unwrap();
            if let Some(err) = err_guard.as_ref() {
                ui.label(err);
            }
        });

        if let Some(idx) = video_to_load {
            self.selected_video_idx = Some(idx);
            if let Ok(c) = videoio::VideoCapture::from_file(
                self.videos[idx].to_str().unwrap(),
                videoio::CAP_ANY,
            ) {
                self.native_fps = c.get(videoio::CAP_PROP_FPS).unwrap_or(30.0);
                self.duration =
                    c.get(videoio::CAP_PROP_FRAME_COUNT).unwrap_or(0.0) / self.native_fps;
                self.ranges = vec![VideoRange {
                    start_time: 0.0,
                    end_time: self.duration, // Now correctly spans the whole file
                    crop_rect_norm: None,
                    note: String::new(),
                }];
                self.current_range_idx = 0;
                self.current_time = 0.0;
                self.cap = Some(c);
                self.update_frame(ctx);
            }
        }

        if self.is_playing() {
            self.current_time += ctx.input(|i| i.stable_dt) as f64;
            if let PlayState::PlayingUntil(x) = self.play_state {
                if x < self.current_time {
                    self.play_state = PlayState::NotPlaying;
                }
            }
            if self.current_time >= self.duration {
                self.play_state = PlayState::NotPlaying;
            }
            self.update_frame(ctx);
            ctx.request_repaint();
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_maximized(true), // Starts maximized (windowed but fills screen)
        // .with_fullscreen(true), // Use this instead for true kiosk-style fullscreen
        ..Default::default()
    };
    eframe::run_native(
        "VidDataTrainCrop",
        options,
        Box::new(|_cc| Ok(Box::new(VideoApp::default()))),
    )
}
