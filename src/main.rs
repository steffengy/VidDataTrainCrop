use eframe::egui;
use opencv::{core, imgcodecs, imgproc, opencv_has_inherent_feature_algorithm_hint, prelude::*, videoio};
use std::cmp::Ordering;
use std::fs;
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

// 1. Introduce an enum to handle both Videos and static Images
enum MediaSource {
    Video(videoio::VideoCapture),
    Image(core::Mat),
}

struct VideoApp {
    input_folder: Option<PathBuf>,
    output_folder: Option<PathBuf>,
    videos: Vec<PathBuf>,
    selected_file_idx: Option<usize>,
    media: Option<MediaSource>, // Replaced `cap` with `media`
    is_image: bool,             // Quick flag to toggle UI elements
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
    frame_text: String,
}

impl Default for VideoApp {
    fn default() -> Self {
        Self {
            input_folder: None,
            output_folder: None,
            videos: Vec::new(),
            selected_file_idx: None,
            media: None,
            is_image: false,
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
            frame_text: "0".to_string(),
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
            PlayState::PlayingUntil(_) => PlayState::NotPlaying,
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
        let mut frame = core::Mat::default();
        let mut valid_frame = false;

        // 2. Safely read from either the VideoCapture or the static Image Mat
        if let Some(ref mut media) = self.media {
            match media {
                MediaSource::Video(cap) => {
                    let frame_pos = (self.current_time * self.native_fps) as i32;
                    let _ = cap.set(videoio::CAP_PROP_POS_FRAMES, frame_pos as f64);
                    if cap.read(&mut frame).unwrap_or(false) && !frame.empty() {
                        valid_frame = true;
                    }
                }
                MediaSource::Image(mat) => {
                    if !mat.empty() {
                        mat.copy_to(&mut frame).unwrap();
                        valid_frame = true;
                    }
                }
            }
        }

        if valid_frame {
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

    fn run_export(&self) {
        let (Some(idx), Some(out_dir)) = (self.selected_file_idx, &self.output_folder) else {
            return;
        };
        let input_path = self.videos[idx].clone();
        let stem = input_path
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();

        let ext = input_path
            .extension()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        let is_img = matches!(
            ext.as_str(),
            "jpg" | "jpeg" | "png" | "bmp" | "webp"
        );

        let ranges = self.ranges.clone();
        let out_dir = out_dir.clone();

        // Get dimensions for crop math depending on media source
        let (vid_w, vid_h) = if let Some(ref media) = self.media {
            match media {
                MediaSource::Video(cap) => (
                    cap.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(1920.0),
                    cap.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(1080.0),
                ),
                MediaSource::Image(mat) => {
                    let size = mat.size().unwrap();
                    (size.width as f64, size.height as f64)
                }
            }
        } else {
            (1920.0, 1080.0)
        };

        self.is_exporting
            .store(true, std::sync::atomic::Ordering::SeqCst);
        *self.export_error.lock().unwrap() = None;

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
                let out_base = if ranges.len() > 1 {
                    out_dir.join(format!("{}_range{}", &stem, i))
                } else {
                    out_dir.join(&stem)
                };
                println!("DBG: {:?}", out_base);

                if !range.note.is_empty() {
                    let _ = std::fs::write(out_base.with_added_extension("txt"), &range.note);
                }

                // 3. Conditional FFmpeg command construction based on if it's an image
                let mut cmd = Command::new("ffmpeg");
                cmd.arg("-y");

                if !is_img {
                    cmd.arg("-ss")
                        .arg(range.start_time.to_string())
                        .arg("-to")
                        .arg(range.end_time.to_string());
                }

                cmd.arg("-i").arg(&input_path);

                let mut filters = vec![];
                if !is_img {
                    filters.push("fps=16".to_string());
                }

                if let Some(ref norm) = range.crop_rect_norm {
                    let cw = ((norm.max_x - norm.min_x).abs() as f64 * vid_w) as i32 & !1;
                    let ch = ((norm.max_y - norm.min_y).abs() as f64 * vid_h) as i32 & !1;
                    let cx = (norm.min_x.min(norm.max_x) as f64 * vid_w) as i32;
                    let cy = (norm.min_y.min(norm.max_y) as f64 * vid_h) as i32;
                    filters.push(format!("crop={}:{}:{}:{}", cw, ch, cx, cy));
                }

                if !filters.is_empty() {
                    cmd.arg("-vf").arg(filters.join(","));
                }

                let out_ext = if is_img { ext.to_string() } else { "mp4".to_string() };
                let out_file = out_base.with_added_extension(&out_ext);

                if !is_img {
                    cmd.arg("-c:v")
                        .arg("libx264")
                        .arg("-preset")
                        .arg("ultrafast");
                }

                cmd.arg(&out_file);

                println!("Exporting Range {}: file {:?}", i, out_file);

                match cmd.status() {
                    Ok(status) if !status.success() => {
                        let err_msg = format!(
                            "FFmpeg failed on range {} with exit code: {:?}",
                            i,
                            status.code()
                        );
                        *exp_err.lock().unwrap() = Some(err_msg);
                        break;
                    }
                    Err(e) => {
                        *exp_err.lock().unwrap() = Some(format!("Failed to start FFmpeg: {}", e));
                        break;
                    }
                    _ => {}
                }
            }
            println!("All exports finished.");
        });
    }
}

impl eframe::App for VideoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut file_idx_to_load = None;

        // Keyboard Logic (Disable for images to prevent accidental scrubbing)
        if !ctx.wants_keyboard_input() && !self.is_image {
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
                                    // 4. Added image extensions here
                                    ext == "mp4" || ext == "mkv" || ext == "avi" || ext == "mov" || ext == "webm" ||
                                    ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" || ext == "webp"
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
                ui.heading("Files");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.set_min_width(400.0);

                    for (i, v) in self.videos.iter().enumerate() {
                        let name = v.file_name().unwrap().to_string_lossy();
                        if ui
                            .selectable_label(self.selected_file_idx == Some(i), name)
                            .clicked()
                        {
                            file_idx_to_load = Some(i);
                        }
                    }
                });
            });

        egui::SidePanel::right("right")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading(if self.is_image { "Active Crops" } else { "Active Ranges" });
                if ui.button(if self.is_image { "➕ Add Crop" } else { "➕ Add Range" }).clicked() {
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

                        let label_text = if self.is_image {
                            format!("Crop {}", i)
                        } else {
                            let duration = range.end_time - range.start_time;
                            let frame_count_16fps = (duration * 16.0).round() as i32;
                            let start_frame = (range.start_time * self.native_fps).round() as i32;
                            let end_frame = (range.end_time * self.native_fps).round() as i32;

                            format!(
                                "R{}: {:.1}s - {:.1}s ({:.1}s)\n      {} - {} ({} frames)",
                                i,
                                range.start_time,
                                range.end_time,
                                duration,
                                start_frame,
                                end_frame,
                                frame_count_16fps
                            )
                        };

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
            let mut avail_size = ui.available_size();
            avail_size.y = avail_size.y - 280.0;
            let mut avail_w = avail_size.x;

            // 1. Determine the display rectangle based on texture aspect ratio
            let rect = if let Some(tex) = &self.video_texture {
                let tex_size = tex.size_vec2();
                let scale = (avail_size.x / tex_size.x).min(avail_size.y / tex_size.y);
                let display_size = tex_size * scale;

                // Center the image in the available space
                let left_top = ui.cursor().min + (avail_size - display_size) * 0.5;
                egui::Rect::from_min_size(left_top, display_size)
            } else {
                // Fallback if no video is loaded
                let fallback_h = avail_size.x * 0.5625;
                ui.allocate_exact_size(egui::vec2(avail_size.x, fallback_h), egui::Sense::hover()).0
            };

            // Allocate the interaction area at the calculated rect
            let response = ui.interact(rect, ui.id().with("video_interact"), egui::Sense::click_and_drag());

            // 2. Paint the background and the image
            if let Some(tex) = &self.video_texture {
                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK); // Black bars area
                ui.painter().image(
                    tex.id(),
                    rect,
                    egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                    egui::Color32::WHITE,
                );
            } else {
                ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);
            }

            // 3. Coordinate mapping (Now uses the correctly aspect-ratioed 'rect')
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

            // --- Crop Handling (Remains the same logic, but uses updated rect) ---
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
                        // Clamp to 0.0-1.0 to prevent cropping outside the image
                        self.ranges[self.current_range_idx].crop_rect_norm =
                            Some(SerializableRect {
                                min_x: r.min.x.clamp(0.0, 1.0),
                                min_y: r.min.y.clamp(0.0, 1.0),
                                max_x: r.max.x.clamp(0.0, 1.0),
                                max_y: r.max.y.clamp(0.0, 1.0),
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

            // 4. Playback Controls / UI below the video
            ui.advance_cursor_after_rect(rect);
            ui.add_space(8.0);

            // 5. Hide the timeline/playback info if we are looking at a static image
            if !self.is_image {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label("Native Frame:");

                    let response = ui.add(
                        egui::TextEdit::singleline(&mut self.frame_text)
                            .desired_width(80.0)
                    );

                    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Ok(frame_num) = self.frame_text.trim().parse::<i32>() {
                            self.current_time = (frame_num as f64) / self.native_fps;
                            self.current_time = self.current_time.clamp(0.0, self.duration);
                            self.update_frame(ctx);
                        }
                    }

                    if !response.has_focus() {
                        let current_frame = (self.current_time * self.native_fps) as i32;
                        self.frame_text = current_frame.to_string();
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!("Target 16FPS: {:.1}", self.current_time * 16.0));
                    });
                });

                let track_width = avail_w - 60.0;
                ui.spacing_mut().slider_width = track_width;

                let slider_res = ui.add(
                    egui::Slider::new(&mut self.current_time, 0.0..=self.duration)
                        .show_value(true)
                        .suffix("s"),
                );
                if slider_res.changed() {
                    self.update_frame(ctx);
                }

                if !self.ranges.is_empty() {
                    let range = &self.ranges[self.current_range_idx];
                    let rect = slider_res.rect;

                    let time_to_x = |time: f64| {
                        let pct = (time / self.duration) as f32;
                        rect.min.x + pct * track_width
                    };

                    let painter = ui.painter();
                    let stroke_start = egui::Stroke::new(2.0, egui::Color32::GREEN);
                    let stroke_end = egui::Stroke::new(2.0, egui::Color32::RED);

                    if range.start_time > 0.0 {
                        let x = time_to_x(range.start_time);
                        painter.line_segment(
                            [egui::pos2(x, rect.min.y), egui::pos2(x, rect.max.y)],
                            stroke_start,
                        );
                    }

                    if range.end_time < self.duration {
                        let x = time_to_x(range.end_time);
                        painter.line_segment(
                            [egui::pos2(x, rect.min.y), egui::pos2(x, rect.max.y)],
                            stroke_end,
                        );
                    }

                    let start_x = time_to_x(range.start_time);
                    let end_x = time_to_x(range.end_time);
                    painter.rect_filled(
                        egui::Rect::from_min_max(
                            egui::pos2(start_x, rect.center().y - 2.0),
                            egui::pos2(end_x, rect.center().y + 2.0),
                        ),
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(255, 255, 255, 40),
                    );
                }
            } // end if !self.is_image

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if !self.is_image {
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
                }

                if !self.ranges.is_empty() {
                    if !self.is_image {
                        if ui.button("Set Start").clicked() {
                            self.ranges[self.current_range_idx].start_time = self.current_time;
                        }
                        if ui.button("Set End").clicked() {
                            self.ranges[self.current_range_idx].end_time = self.current_time;
                        }
                    }
                    if ui.button("Clear Crop").clicked() {
                        self.ranges[self.current_range_idx].crop_rect_norm = None;
                    }
                    if !self.is_image {
                        ui.separator();
                        if ui.add(egui::Button::new("🔁 Play Range (R)")).clicked() {
                            let range = &self.ranges[self.current_range_idx];
                            self.current_time = range.start_time;
                            self.play_state = PlayState::PlayingUntil(range.end_time);
                        }
                    }
                }
            });

            if !self.ranges.is_empty() {
                ui.add_space(10.0);
                ui.label(if self.is_image {
                    format!("Note for Crop {}:", self.current_range_idx)
                } else {
                    format!("Note for Range {}:", self.current_range_idx)
                });

                ui.add(
                    egui::TextEdit::multiline(&mut self.ranges[self.current_range_idx].note)
                        .desired_width(avail_w)
                        .desired_rows(5),
                );
            }

            ui.add_space(10.0);
            let exporting = self.is_exporting.load(atomic::Ordering::SeqCst);

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

            if exporting {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Processing ranges with FFmpeg...");
                });
            }

            let mut err_guard = self.export_error.lock().unwrap();
            if let Some(err) = err_guard.as_ref() {
                ui.label(err);
            }
        });

        // 6. Handle loading the new media depending on its extension
        if let Some(idx) = file_idx_to_load {
            self.selected_file_idx = Some(idx);
            let path = &self.videos[idx];

            // Read note from .txt file if it already exists
            let p = path.with_extension("txt");
            let note = if p.exists() {
                fs::read_to_string(p).unwrap_or_default()
            } else {
                String::new()
            };

            let ext = path.extension().unwrap_or_default().to_string_lossy().to_lowercase();

            self.is_image = matches!(
                ext.as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "webp"
            );

            if self.is_image {
                // Load using imgcodecs instead of VideoCapture
                if let Ok(mat) = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR) {
                    self.native_fps = 1.0;
                    self.duration = 0.0;
                    self.ranges = vec![VideoRange {
                        start_time: 0.0,
                        end_time: 0.0,
                        crop_rect_norm: None,
                        note: note,
                    }];
                    self.current_range_idx = 0;
                    self.current_time = 0.0;
                    self.media = Some(MediaSource::Image(mat));
                    self.update_frame(ctx);
                }
            } else {
                if let Ok(c) = videoio::VideoCapture::from_file(
                    path.to_str().unwrap(),
                    videoio::CAP_ANY,
                ) {
                    self.native_fps = c.get(videoio::CAP_PROP_FPS).unwrap_or(30.0);
                    self.duration =
                        c.get(videoio::CAP_PROP_FRAME_COUNT).unwrap_or(0.0) / self.native_fps;
                    self.ranges = vec![VideoRange {
                        start_time: 0.0,
                        end_time: self.duration,
                        crop_rect_norm: None,
                        note: note,
                    }];
                    self.current_range_idx = 0;
                    self.current_time = 0.0;
                    self.media = Some(MediaSource::Video(c));
                    self.update_frame(ctx);
                }
            }
        }

        if self.is_playing() && !self.is_image {
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
        viewport: egui::ViewportBuilder::default().with_maximized(true),
        ..Default::default()
    };
    eframe::run_native(
        "VidDataTrainCrop",
        options,
        Box::new(|_cc| Ok(Box::new(VideoApp::default()))),
    )
}
