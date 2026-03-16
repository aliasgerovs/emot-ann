import gradio as gr
import os
import pandas as pd
from datetime import datetime
import shutil
import time
import gc
from contextlib import contextmanager
from moviepy import VideoFileClip
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

class VideoEmotionAnnotator:
    def __init__(self):
        self.base_dir = os.path.abspath(os.getcwd())
        self.clips_dir = None
        self.working_dir = os.path.join(self.base_dir, 'annotation_working')
        self.uploads_dir = os.path.join(self.base_dir, 'annotation_uploads')
        self.annotation_clips_dir = os.path.join(self.base_dir, 'annotation_clips')
        
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.annotation_clips_dir, exist_ok=True)
        
        print(f"Using working directory: {self.working_dir}")
        print(f"Using uploads directory: {self.uploads_dir}")
        print(f"Using annotation clips directory: {self.annotation_clips_dir}")
        
        self.ffmpeg_available = shutil.which('ffprobe') is not None
        
        if self.ffmpeg_available:
            print("FFmpeg detected - using for video info")
        else:
            print("FFmpeg not detected - falling back to MoviePy for video info")
        
        self.annotations = []
        self.current_clips = []
        self.participant_id = ""
        self.working_video_path = None
        self.presence_of_others = 0
        
    def __del__(self):
        try:
            if hasattr(self, 'working_dir') and os.path.exists(self.working_dir):
                for file in os.listdir(self.working_dir):
                    try:
                        file_path = os.path.join(self.working_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except:
                        pass
            
            if hasattr(self, 'clips_dir') and self.clips_dir and os.path.exists(self.clips_dir):
                for file in os.listdir(self.clips_dir):
                    if file.startswith('clip_'):
                        file_path = os.path.join(self.clips_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            time.sleep(0.1)
        except Exception as e:
            print(f"Error cleaning up files: {str(e)}")
    
    @contextmanager
    def video_clip_context(self, path):
        clip = None
        try:
            clip = VideoFileClip(path)
            yield clip
        finally:
            if clip is not None:
                clip.close()
            time.sleep(0.2)
    
    def _get_video_info_ffmpeg(self, path):
        if not self.ffmpeg_available:
            return 0, 0, 0
        
        try:
            cmd_duration = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                path
            ]
            duration_result = subprocess.run(cmd_duration, capture_output=True, text=True, check=True)
            duration = float(duration_result.stdout.strip())
            
            cmd_fps = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'csv=p=0',
                path
            ]
            fps_result = subprocess.run(cmd_fps, capture_output=True, text=True, check=True)
            r_frame_rate = fps_result.stdout.strip()
            if '/' in r_frame_rate:
                num, den = map(int, r_frame_rate.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(r_frame_rate) if r_frame_rate else 30.0
            
            frame_count = int(duration * fps)
            return duration, fps, frame_count
        except Exception as e:
            print(f"ffprobe error: {e}")
            return 0, 0, 0
    
    def get_video_info(self, video_path):
        if not video_path:
            return 0, 0, 0
        
        working_path = self.copy_video_to_working_dir(video_path)
        if not working_path:
            return 0, 0, 0
        
        duration, fps, frame_count = self._get_video_info_ffmpeg(working_path)
        if duration > 0:
            return duration, fps, frame_count
        
        try:
            time.sleep(0.3)
            with self.video_clip_context(working_path) as clip:
                duration = clip.duration
                fps = clip.fps
                frame_count = int(duration * fps)
                return duration, fps, frame_count
        except Exception as e:
            print(f"Error getting video info: {e}")
            import traceback
            traceback.print_exc()
        
        return 0, 0, 0

    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def copy_video_to_working_dir(self, video_path):
        if not video_path or not os.path.exists(video_path):
            print(f"Video path does not exist: {video_path}")
            return None
        
        try:
            filename = os.path.basename(video_path)
            timestamp = int(time.time() * 1000)
            upload_path = os.path.join(self.uploads_dir, f"upload_{timestamp}_{filename}")
            
            print(f"Copying from: {video_path}")
            print(f"Copying to: {upload_path}")
            
            with open(video_path, 'rb') as src:
                with open(upload_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
            
            if not os.path.exists(upload_path):
                print("Error: Upload file was not created")
                return None
            
            file_size = os.path.getsize(upload_path)
            print(f"Uploaded file size: {file_size} bytes")
            
            working_filename = f"working_{timestamp}_{filename}"
            working_path = os.path.join(self.working_dir, working_filename)
            
            print(f"Copying to working: {working_path}")
            shutil.copy2(upload_path, working_path)
            
            if os.path.exists(working_path):
                print(f"Successfully copied to working directory: {working_path}")
                time.sleep(0.3)
                return working_path
            else:
                print("Error: Working file was not created")
                return None
                
        except Exception as e:
            print(f"Error copying video: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_video(self, video_file, participant_id, start_time, interval, process_minutes, progress=gr.Progress()):
        if not video_file or not participant_id:
            return "Please upload a video and provide a Participant ID.", gr.update(choices=[])
        
        progress(0, desc="Starting video processing...")
        
        if hasattr(video_file, 'name'):
            video_path = video_file.name
        elif isinstance(video_file, str):
            video_path = video_file
        else:
            return "Invalid video file.", gr.update(choices=[])
        
        print(f"Received video path: {video_path}")
        
        if not os.path.exists(video_path):
            return f"Video file not found: {video_path}", gr.update(choices=[])
            
        self.participant_id = participant_id
        
        progress(0.05, desc="Cleaning up old files...")
        
        for clip in self.current_clips:
            try:
                if os.path.exists(clip['path']):
                    os.remove(clip['path'])
                    time.sleep(0.1)
            except Exception as e:
                print(f"Could not remove old clip: {str(e)}")
        
        if self.working_video_path and os.path.exists(self.working_video_path):
            try:
                os.remove(self.working_video_path)
                time.sleep(0.3)
            except Exception as e:
                print(f"Could not remove old working video: {str(e)}")
        
        progress(0.1, desc="Copying video to working directory...")
        
        self.working_video_path = self.copy_video_to_working_dir(video_path)
        if not self.working_video_path:
            return "Failed to prepare video file. Please try again.", gr.update(choices=[])
        
        video_filename = os.path.splitext(os.path.basename(self.working_video_path))[0]
        self.clips_dir = os.path.join(self.annotation_clips_dir, video_filename)
        os.makedirs(self.clips_dir, exist_ok=True)
        print(f"Saving clips to: {self.clips_dir}")
        
        time.sleep(0.5)
        
        progress(0.15, desc="Reading video information...")
        
        fps = 0
        duration = 0
        
        duration, fps, _ = self._get_video_info_ffmpeg(self.working_video_path)
        if duration == 0:
            try:
                with self.video_clip_context(self.working_video_path) as clip:
                    duration = clip.duration
                    fps = clip.fps
            except Exception as e:
                print(f"Error reading video: {e}")
                import traceback
                traceback.print_exc()
                return "Failed to read video file. Please try again.", gr.update(choices=[])
        
        print(f"Video info - Duration: {duration}s, FPS: {fps}")
        
        time.sleep(0.3)
        
        if duration == 0:
            return "Failed to read video file. Please try again.", gr.update(choices=[])
        
        progress(0.25, desc="Calculating clip parameters...")
        
        requested_seconds = (process_minutes or 0) * 60
        if requested_seconds <= 0:
            requested_seconds = 240
        end_time = min(start_time + requested_seconds, duration)
        
        if end_time <= start_time:
            return "Start time is too close to the end of the video. Please choose an earlier start time.", gr.update(choices=[])
        
        time_range_msg = f"Processing from {self.format_time(start_time)} to {self.format_time(end_time)} (duration: {self.format_time(end_time - start_time)})"
        self.annotations = []
        
        progress(0.3, desc="Starting clip extraction...")
        
        try:
            time.sleep(0.3)
            self.current_clips = []
            clip_params_list = []
            current_time = start_time
            clip_num = 1
            while current_time < end_time:
                clip_end_time = min(current_time + interval, end_time)
                clip_params_list.append((current_time, clip_end_time, clip_num))
                current_time += interval
                clip_num += 1
            
            total_clips = len(clip_params_list)
            print(f"Will create {total_clips} clips")

            def create_single_clip(clip_params):
                current_time, clip_end_time, clip_num = clip_params
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                clip_filename = f"clip_{self.participant_id}_{timestamp}_{clip_num:03d}.mp4"
                output_path = os.path.join(self.clips_dir, clip_filename)
                
                try:
                    with self.video_clip_context(self.working_video_path) as video:
                        subclip = video.subclipped(current_time, clip_end_time)
                        
                        subclip.write_videofile(
                            output_path,
                            codec='libx264',
                            audio_codec='aac',
                            ffmpeg_params=['-movflags', '+faststart'],
                            logger=None,
                            threads=1,
                            preset='ultrafast'
                        )
                        subclip.close()
                    
                    time.sleep(2.0)
                    
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        clip_label = f"Clip {clip_num:03d} ({self.format_time(current_time)} - {self.format_time(clip_end_time)})"
                        return {
                            'label': clip_label,
                            'path': output_path,
                            'start': current_time,
                            'end': clip_end_time,
                            'number': clip_num
                        }
                    else:
                        return None
                
                except Exception as e:
                    error_str = str(e)
                    if 'stdout' in error_str and 'NoneType' in error_str:
                        print(f"Ignoring known audio processing error for clip {clip_num}: {error_str}")
                        time.sleep(2.0)
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            clip_label = f"Clip {clip_num:03d} ({self.format_time(current_time)} - {self.format_time(clip_end_time)})"
                            return {
                                'label': clip_label,
                                'path': output_path,
                                'start': current_time,
                                'end': clip_end_time,
                                'number': clip_num
                            }
                        else:
                            return None
                    else:
                        print(f"Unexpected error creating clip {clip_num}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return None
            
            progress(0.3, desc="Starting parallel clip extraction...")
            
            with ThreadPoolExecutor(max_workers=30) as executor:
                future_to_clipnum = {executor.submit(create_single_clip, params): params[2] for params in clip_params_list}
                completed = 0
                for future in as_completed(future_to_clipnum):
                    try:
                        result = future.result()
                        if result:
                            self.current_clips.append(result)
                            print(f"Successfully created clip {result['number']}")
                        completed += 1
                        progress(
                            0.3 + (0.65 * completed / total_clips),
                            desc=f"Extracted {completed}/{total_clips} clips"
                        )
                    except Exception as exc:
                        print(f"Error in clip future: {exc}")
                        completed += 1
                        progress(
                            0.3 + (0.65 * completed / total_clips),
                            desc=f"Extracted {completed}/{total_clips} clips (some errors)"
                        )
            
            self.current_clips.sort(key=lambda x: x['number'])
        
        except Exception as e:
            print(f"Error during clip extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Failed to process video: {str(e)}", gr.update(choices=[])
        
        progress(0.95, desc="Finalizing...")
        time.sleep(0.5)
        
        if not self.current_clips:
            return "No clips were successfully created. Please try again.", gr.update(choices=[])
        
        clip_choices = [clip['label'] for clip in self.current_clips]
        
        progress(1.0, desc="Complete!")
        
        success_msg = f"""✅ **Video processed successfully!**

{time_range_msg}

📊 **Results:**
- Created {len(self.current_clips)} clips
- Clip interval: {interval} seconds
- Ready for annotation

Select a clip below to begin annotating."""
        
        return success_msg, gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None)

    def load_clip(self, clip_label):
        if not clip_label:
            return None
        
        for clip in self.current_clips:
            if clip['label'] == clip_label:
                if os.path.exists(clip['path']):
                    full_path = clip['path']
                    print(f"Loading clip full path: {full_path}")
                    return full_path
        
        return None
    
    def save_annotation(self, clip_label, has_no_emotion, task_type, annotator_name,
                       focused_stimulus, distraction, interaction_family, interaction_experimenter,
                       self_comforting, persistence, checking_adults, out_of_chair):
        if not clip_label:
            return "⚠️ Please select a clip first."
        
        if not task_type or not annotator_name:
            return "⚠️ Please fill in Task Type and Annotator Name."
        
        clip_info = None
        for clip in self.current_clips:
            if clip['label'] == clip_label:
                clip_info = clip
                break
        
        if not clip_info:
            return "⚠️ Clip not found."
        
        annotation = {
            'ParticipantID': self.participant_id,
            'TaskType': task_type,
            'AnnotatorName': annotator_name,
            'ClipNumber': clip_info['number'],
            'StartTime': self.format_time(clip_info['start']),
            'EndTime': self.format_time(clip_info['end']),
            'NC': 1 if has_no_emotion else 0,
            'FocusedOnStimulus': focused_stimulus,
            'Distraction': distraction,
            'InteractionWithFamily': interaction_family,
            'InteractionWithExperimenter': interaction_experimenter,
            'SelfComforting': self_comforting,
            'Persistence': persistence,
            'CheckingInWithAdults': checking_adults,
            'OutOfChair': out_of_chair,
            'PresenceOfOthersInRoom': self.presence_of_others,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        existing_idx = None
        for idx, ann in enumerate(self.annotations):
            if (ann['ParticipantID'] == annotation['ParticipantID'] and 
                ann['ClipNumber'] == annotation['ClipNumber'] and
                ann['TaskType'] == annotation['TaskType'] and
                ann['AnnotatorName'] == annotation['AnnotatorName']):
                existing_idx = idx
                break
        
        if existing_idx is not None:
            self.annotations[existing_idx] = annotation
            return f"✅ **Annotation updated!** (Clip {clip_info['number']}/{len(self.current_clips)})"
        else:
            self.annotations.append(annotation)
            return f"✅ **Annotation saved!** ({len(self.annotations)}/{len(self.current_clips)} clips annotated)"
    
    def export_csv(self):
        if not self.annotations:
            return None, "⚠️ No annotations to export. Please annotate some clips first."
        
        try:
            df = pd.DataFrame(self.annotations)
            df = df.sort_values(['ParticipantID', 'TaskType', 'ClipNumber'])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"annotations_{self.participant_id}_{timestamp}.csv"
            output_path = os.path.join(self.base_dir, filename)
            
            df.to_csv(output_path, index=False)
            
            return output_path, f"✅ **Exported successfully!** ({len(self.annotations)} annotations saved to {filename})"
        
        except Exception as e:
            return None, f"❌ Export failed: {str(e)}"

annotator = VideoEmotionAnnotator()

def on_video_upload(video):
    if not video:
        return "Please upload a video.", 0, 0, 1
    
    video_path = video.name if hasattr(video, 'name') else video
    print(f"Video uploaded: {video_path}")
    
    duration, fps, frame_count = annotator.get_video_info(video_path)
    
    if duration == 0:
        return "⚠️ Could not read video file. Please try a different format.", 0, 0, 1
    
    total_minutes = int(duration // 60)
    remaining_seconds = int(duration % 60)
    
    return (
        f"✅ **Video uploaded successfully!**\n\n📹 Duration: {total_minutes}:{remaining_seconds:02d} ({duration:.1f}s)\n🎞️ Frame rate: {fps:.2f} fps",
        0,
        0,
        min(4, max(1, total_minutes))
    )

css = """
#process_btn, #save_btn, #export_btn {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    font-weight: 600;
}
.step-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 20px 0 10px 0;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
}
"""

with gr.Blocks(css=css, title="Video Emotion Annotator v2") as demo:
    gr.HTML("<div style='text-align: center; padding: 20px;'><h1>🎬 Video Emotion Annotation Tool</h1><p>Process videos, annotate emotions, and export results</p></div>")
    
    gr.HTML("<div style='max-width: 1400px; margin: 0 auto;'>")
    
    gr.HTML("<div class='step-header'>⚙️ Step 1: Upload & Configure</div>")
    
    with gr.Row():
        with gr.Column(scale=3):
            
            with gr.Row():
                video_input = gr.Video(
                    label="Upload Video", 
                    sources=["upload"],
                    include_audio=True
                )
                participant_id = gr.Textbox(label="Participant ID", placeholder="Enter participant identifier")
            
            with gr.Row():
                task_type = gr.Dropdown(label="Task Type", choices=["Traditional", "Digital"], value="Traditional")
                annotator_name = gr.Textbox(label="Annotator Name", placeholder="Enter annotator name")
            
            gr.Markdown("**Presence of Others in the Room (0-1 scale, coded once per session)**")
            presence_of_others = gr.Slider(minimum=0, maximum=1, step=1, label="Presence of Others in the Room", value=0)
            
            upload_status = gr.Markdown("", visible=True)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Start Time**")
                    with gr.Row():
                        start_minutes = gr.Number(label="Minutes", value=0, minimum=0, precision=0)
                        start_seconds = gr.Number(label="Seconds", value=0, minimum=0, maximum=59, precision=0)
                with gr.Column(scale=1):
                    interval = gr.Number(label="Clip Interval (seconds)", value=5, minimum=1)
                    process_minutes = gr.Number(label="Process Duration (minutes)", value=1, minimum=1)
            
            start_time = gr.Number(value=0, visible=False)
            
            process_btn = gr.Button("🎬 Process Video", elem_id="process_btn", size="lg")
            process_status = gr.Markdown("")
        
        with gr.Column(scale=2):
            gr.HTML("<div class='step-header'>📊 Progress Overview</div>")
            gr.Markdown("Upload your video and set parameters to begin annotation.")
    
    gr.HTML("<div class='step-header'>🎯 Step 2: Annotate Clips</div>")
    
    with gr.Row():
        with gr.Column(scale=2):
            clip_selector = gr.Dropdown(label="Select Clip", choices=[], interactive=True)
            clip_video = gr.Video(label="Current Clip", height=300, autoplay=False)
        
        with gr.Column(scale=1):
            has_emotion = gr.Checkbox(label="NC (No Clear Emotion)", value=False)
            
            gr.Markdown("**Behavioral scales (per clip):**")
            focused_stimulus = gr.Slider(0, 1, step=1, label="Focused on Stimulus (0-1)", value=0)
            distraction = gr.Slider(0, 1, step=1, label="Distraction (0-1)", value=0)
            interaction_family = gr.Slider(0, 1, step=1, label="Interaction with Family Member(s) (0-1)", value=0)
            interaction_experimenter = gr.Slider(0, 1, step=1, label="Interaction with Experimenter (0-1)", value=0)
            self_comforting = gr.Slider(0, 1, step=1, label="Self-comforting (0-1)", value=0)
            persistence = gr.Slider(0, 2, step=1, label="Persistence (0-2)", value=0)
            checking_adults = gr.Slider(0, 1, step=1, label="Checking in with Adults in the Room (0-1)", value=0)
            out_of_chair = gr.Slider(0, 1, step=1, label="Out of the Chair (0-1)", value=0)
            
            save_btn = gr.Button("💾 Save Annotation", elem_id="save_btn")
            annotation_status = gr.Markdown("")
        
    gr.HTML("<div class='step-header'>📤 Step 3: Export Results</div>")
    
    with gr.Row():
        export_btn = gr.Button("📊 Export CSV", elem_id="export_btn", size="lg")
        export_file = gr.File(label="Download Results", visible=False)
        export_status = gr.Markdown("")
    
    gr.HTML("</div>")
    
    def calculate_start_time(minutes, seconds):
        return (minutes or 0) * 60 + (seconds or 0)
    
    def set_presence_of_others(val):
        annotator.presence_of_others = int(val) if val is not None else 0
    
    presence_of_others.change(set_presence_of_others, inputs=[presence_of_others])
    
    video_input.upload(on_video_upload, inputs=[video_input], outputs=[upload_status, start_minutes, start_seconds, process_minutes])
    
    start_minutes.change(calculate_start_time, inputs=[start_minutes, start_seconds], outputs=[start_time])
    start_seconds.change(calculate_start_time, inputs=[start_minutes, start_seconds], outputs=[start_time])
    
    process_btn.click(
        annotator.process_video,
        inputs=[video_input, participant_id, start_time, interval, process_minutes],
        outputs=[process_status, clip_selector],
        show_progress="full",
        api_name="process_video"
    )
    
    def load_clip_and_reset(selected_clip):
        if not selected_clip:
            return (None, False, 0, 0, 0, 0, 0, 0, 0, 0)
        
        video_path = annotator.load_clip(selected_clip)
        
        if video_path and os.path.exists(video_path):
            print(f"Clip ready in temp dir: {video_path}")
            time.sleep(0.5)
            return (video_path, False, 0, 0, 0, 0, 0, 0, 0, 0)
        else:
            print(f"Clip load failed: {video_path} (exists: {os.path.exists(video_path) if video_path else 'None'})")
            return (None, False, 0, 0, 0, 0, 0, 0, 0, 0)
    
    clip_selector.change(
        load_clip_and_reset,
        inputs=[clip_selector],
        outputs=[clip_video, has_emotion, focused_stimulus, distraction, interaction_family,
                 interaction_experimenter, self_comforting, persistence, checking_adults, out_of_chair],
        show_progress="hidden"
    )
    
    save_btn.click(
        annotator.save_annotation,
        inputs=[clip_selector, has_emotion, task_type, annotator_name,
                focused_stimulus, distraction, interaction_family, interaction_experimenter,
                self_comforting, persistence, checking_adults, out_of_chair],
        outputs=[annotation_status]
    )
    
    def handle_export():
        file, status = annotator.export_csv()
        if file:
            return gr.update(value=file, visible=True), status
        return gr.update(visible=False), status
    
    export_btn.click(handle_export, outputs=[export_file, export_status])

if __name__ == "__main__":
    port = 7861
    
    print("\n" + "="*60)
    print("🎬 VIDEO EMOTION ANNOTATION TOOL (v2)")
    print("="*60)
    print(f"Local URL: http://localhost:{port}")
    print("="*60 + "\n")
    
    demo.queue(max_size=20, default_concurrency_limit=3, api_open=False)
    
    demo.launch(
        share=False,
        server_port=port,
        inbrowser=True,
        show_error=True,
        quiet=False
    )
