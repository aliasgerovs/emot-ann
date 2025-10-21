import gradio as gr
import os
import pandas as pd
from datetime import datetime
import shutil
import time
import gc
from contextlib import contextmanager
from moviepy.editor import VideoFileClip

class VideoEmotionAnnotator:
    def __init__(self):
        # Use current working directory for everything
        self.base_dir = os.path.abspath(os.getcwd())
        self.clips_dir = os.path.join(self.base_dir, 'clips')
        self.working_dir = os.path.join(self.base_dir, 'working')
        
        # Create subdirectories for organization
        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.working_dir, exist_ok=True)
        
        print(f"Current working directory: {self.base_dir}")
        print(f"Clips will be saved to: {self.clips_dir}")
        print(f"Temporary files in: {self.working_dir}")
        
        self.annotations = []
        self.current_clips = []
        self.participant_id = ""
        self.uploaded_video_path = None
        
    def cleanup_directory(self, directory):
        """Clean up files in a directory"""
        try:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    try:
                        file_path = os.path.join(directory, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Could not remove {file_path}: {e}")
        except Exception as e:
            print(f"Error cleaning directory {directory}: {e}")
    
    @contextmanager
    def video_clip_context(self, path):
        """Context manager for MoviePy VideoFileClip"""
        clip = None
        try:
            print(f"Opening video: {path}")
            clip = VideoFileClip(path)
            yield clip
        finally:
            if clip is not None:
                clip.close()
            time.sleep(0.1)
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def save_uploaded_video(self, gradio_video_path):
        """Save uploaded video directly to current working directory"""
        try:
            if not gradio_video_path or not os.path.exists(gradio_video_path):
                print(f"Video path invalid or does not exist: {gradio_video_path}")
                return None
            
            # Generate unique filename in current directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_ext = os.path.splitext(gradio_video_path)[1] or '.mp4'
            new_filename = f"uploaded_video_{timestamp}{original_ext}"
            
            # Save directly to current working directory (not subdirectory)
            destination = os.path.join(self.base_dir, new_filename)
            
            print(f"Copying video from: {gradio_video_path}")
            print(f"Saving to current directory: {destination}")
            
            # Copy file
            shutil.copy2(gradio_video_path, destination)
            
            if os.path.exists(destination):
                size = os.path.getsize(destination)
                print(f"✓ Video saved: {new_filename} ({size} bytes)")
                return destination
            else:
                print("✗ Failed to save video")
                return None
                
        except Exception as e:
            print(f"Error saving uploaded video: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_video_info(self, video_path):
        """Get video information"""
        if not video_path or not os.path.exists(video_path):
            return 0, 0, 0
        
        try:
            with self.video_clip_context(video_path) as clip:
                duration = clip.duration
                fps = clip.fps
                frame_count = int(duration * fps) if fps > 0 else 0
                return duration, fps, frame_count
        except Exception as e:
            print(f"Error getting video info: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0, 0

    def process_video(self, video_file, participant_id, start_time, interval, process_minutes, progress=gr.Progress()):
        if not video_file or not participant_id:
            return "Please upload a video and provide a participant ID.", gr.update(choices=[])
        
        progress(0, desc="Starting video processing...")
        
        # Get the video path from Gradio
        gradio_video_path = video_file if isinstance(video_file, str) else (video_file.name if hasattr(video_file, 'name') else None)
        
        if not gradio_video_path:
            return "Invalid video file.", gr.update(choices=[])
        
        print(f"\n{'='*60}")
        print(f"Processing video: {os.path.basename(gradio_video_path)}")
        print(f"{'='*60}")
        
        self.participant_id = participant_id
        
        progress(0.05, desc="Saving uploaded video to current directory...")
        
        # Save the uploaded video to current directory
        self.uploaded_video_path = self.save_uploaded_video(gradio_video_path)
        if not self.uploaded_video_path:
            return "Failed to save uploaded video. Please try again.", gr.update(choices=[])
        
        progress(0.1, desc="Cleaning up old clips...")
        
        # Clean up old clips
        for clip in self.current_clips:
            try:
                if os.path.exists(clip['path']):
                    os.remove(clip['path'])
            except Exception as e:
                print(f"Could not remove old clip: {e}")
        
        self.current_clips = []
        
        progress(0.15, desc="Reading video information...")
        
        # Get video info
        try:
            with self.video_clip_context(self.uploaded_video_path) as clip:
                fps = clip.fps
                duration = clip.duration
                print(f"Duration: {duration:.2f}s | FPS: {fps:.2f}")
        except Exception as e:
            print(f"Error reading video: {e}")
            import traceback
            traceback.print_exc()
            return "Failed to read video file. Please try again.", gr.update(choices=[])
        
        if duration == 0:
            return "Failed to read video duration. Please try again.", gr.update(choices=[])
        
        progress(0.2, desc="Calculating clip parameters...")
        
        # Calculate end time
        requested_seconds = (process_minutes or 0) * 60
        if requested_seconds <= 0:
            requested_seconds = 240
        end_time = min(start_time + requested_seconds, duration)
        
        if end_time <= start_time:
            return "Start time is too close to the end of the video. Please choose an earlier start time.", gr.update(choices=[])
        
        time_range_msg = f"Processing from {self.format_time(start_time)} to {self.format_time(end_time)} (duration: {self.format_time(end_time - start_time)})"
        self.annotations = []
        
        progress(0.25, desc="Starting clip extraction...")
        
        # Calculate total clips
        current_time = start_time
        total_clips = 0
        while current_time < end_time:
            total_clips += 1
            current_time += interval
        
        print(f"Will create {total_clips} clips with {interval}s interval")
        
        # Extract clips
        try:
            with self.video_clip_context(self.uploaded_video_path) as video:
                clip_num = 1
                current_time = start_time
                current_clip_idx = 0
                
                while current_time < end_time:
                    clip_end_time = min(current_time + interval, end_time)
                    
                    progress(
                        0.25 + (0.7 * current_clip_idx / total_clips),
                        desc=f"Extracting clip {clip_num}/{total_clips}"
                    )
                    
                    # Create unique filename in clips subdirectory
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    clip_filename = f"clip_{self.participant_id}_{clip_num:03d}_{timestamp}.mp4"
                    output_path = os.path.join(self.clips_dir, clip_filename)
                    temp_audio = os.path.join(self.working_dir, f"temp_audio_{timestamp}.m4a")
                    
                    print(f"Clip {clip_num}/{total_clips}: {self.format_time(current_time)}-{self.format_time(clip_end_time)}", end=" ")
                    
                    try:
                        # Extract subclip
                        subclip = video.subclip(current_time, clip_end_time)
                        
                        # Write to file
                        subclip.write_videofile(
                            output_path,
                            codec='libx264',
                            audio_codec='aac',
                            temp_audiofile=temp_audio,
                            remove_temp=True,
                            logger=None,
                            verbose=False,
                            threads=2,
                            preset='ultrafast',
                            ffmpeg_params=['-movflags', '+faststart']
                        )
                        
                        subclip.close()
                        del subclip
                        
                        # Verify the clip was created
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                            clip_label = f"Clip {clip_num:03d} ({self.format_time(current_time)}-{self.format_time(clip_end_time)})"
                            self.current_clips.append({
                                'label': clip_label,
                                'path': output_path,
                                'start': current_time,
                                'end': clip_end_time,
                                'number': clip_num
                            })
                            print("✓")
                        else:
                            print("✗ (not created or empty)")
                    
                    except Exception as e:
                        print(f"✗ Error: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    clip_num += 1
                    current_clip_idx += 1
                    current_time += interval
                    
                    gc.collect()
        
        except Exception as e:
            print(f"Error during clip extraction: {e}")
            import traceback
            traceback.print_exc()
            return f"Failed to process video: {str(e)}", gr.update(choices=[])
        
        progress(0.95, desc="Finalizing...")
        
        if not self.current_clips:
            return "No clips were successfully created. Please try again.", gr.update(choices=[])
        
        clip_choices = [clip['label'] for clip in self.current_clips]
        
        progress(1.0, desc="Complete!")
        
        print(f"{'='*60}")
        print(f"✓ Successfully created {len(self.current_clips)} clips")
        print(f"{'='*60}\n")
        
        success_msg = f"""✅ **Video processed successfully!**

{time_range_msg}

📊 **Results:**
- Created {len(self.current_clips)} clips
- Clip interval: {interval} seconds
- Ready for annotation

**Files saved to:**
- Uploaded video: `{os.path.basename(self.uploaded_video_path)}`
- Clips folder: `clips/`

Select a clip below to begin annotating."""
        
        return success_msg, gr.update(choices=clip_choices, value=clip_choices[0] if clip_choices else None)

    def load_clip(self, clip_label):
        if not clip_label:
            return None
        
        for clip in self.current_clips:
            if clip['label'] == clip_label:
                if os.path.exists(clip['path']):
                    return clip['path']
        
        return None
    
    def save_annotation(self, clip_label, has_no_emotion, sadness, anger, pleasure, task_type, annotator_name):
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
            'Sadness': 0 if has_no_emotion else sadness,
            'Anger': 0 if has_no_emotion else anger,
            'Pleasure': 0 if has_no_emotion else pleasure,
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
            
            # Save to current working directory
            output_path = os.path.join(self.base_dir, filename)
            
            df.to_csv(output_path, index=False)
            
            return output_path, f"✅ **Exported successfully!** ({len(self.annotations)} annotations saved to `{filename}`)"
        
        except Exception as e:
            return None, f"❌ Export failed: {str(e)}"

# Initialize annotator
annotator = VideoEmotionAnnotator()

def on_video_upload(video):
    if not video:
        return "Please upload a video.", 0, 0, 1
    
    # Save video immediately to current directory
    video_path = video if isinstance(video, str) else (video.name if hasattr(video, 'name') else None)
    
    if not video_path:
        return "Invalid video file.", 0, 0, 1
    
    saved_path = annotator.save_uploaded_video(video_path)
    
    if not saved_path:
        return "⚠️ Could not save video file. Please try again.", 0, 0, 1
    
    duration, fps, frame_count = annotator.get_video_info(saved_path)
    
    if duration == 0:
        return "⚠️ Could not read video file. Please try a different format.", 0, 0, 1
    
    total_minutes = int(duration // 60)
    remaining_seconds = int(duration % 60)
    
    return (
        f"✅ **Video uploaded and saved!**\n\n📹 Duration: {total_minutes}:{remaining_seconds:02d} ({duration:.1f}s)\n🎞️ Frame rate: {fps:.2f} fps\n📁 Saved as: `{os.path.basename(saved_path)}`",
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

with gr.Blocks(css=css, title="Video Emotion Annotator") as demo:
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
            clip_video = gr.Video(label="Current Clip", height=300, autoplay=False, show_download_button=True)
        
        with gr.Column(scale=1):
            has_emotion = gr.Checkbox(label="NC (No Clear Emotion)", value=False)
            
            gr.Markdown("**Emotion Intensities (1-3):**")
            sadness_intensity = gr.Slider(minimum=0, maximum=3, step=1, label="😢 Sadness", value=0, visible=True)
            anger_intensity = gr.Slider(minimum=0, maximum=3, step=1, label="😠 Anger", value=0, visible=True)
            pleasure_intensity = gr.Slider(minimum=0, maximum=3, step=1, label="😊 Pleasure", value=0, visible=True)
            
            save_btn = gr.Button("💾 Save Annotation", elem_id="save_btn")
            annotation_status = gr.Markdown("")
        
    gr.HTML("<div class='step-header'>📤 Step 3: Export Results</div>")
    
    with gr.Row():
        export_btn = gr.Button("📊 Export CSV", elem_id="export_btn", size="lg")
        export_file = gr.File(label="Download Results", visible=False)
        export_status = gr.Markdown("")
    
    gr.HTML("</div>")
    
    def update_emotion_fields(nc_checked):
        return (
            gr.update(visible=not nc_checked),
            gr.update(visible=not nc_checked),
            gr.update(visible=not nc_checked)
        )
    
    def calculate_start_time(minutes, seconds):
        return (minutes or 0) * 60 + (seconds or 0)
    
    has_emotion.change(update_emotion_fields, inputs=[has_emotion], outputs=[sadness_intensity, anger_intensity, pleasure_intensity])
    
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
    
    def load_clip_and_reset_filters(selected_clip):
        if not selected_clip:
            return (None, False, 0, 0, 0, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))
        
        video_path = annotator.load_clip(selected_clip)
        
        if video_path and os.path.exists(video_path):
            return (
                video_path,
                False,
                0,
                0,
                0,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True)
            )
        else:
            return (None, False, 0, 0, 0, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))
    
    clip_selector.change(
        load_clip_and_reset_filters,
        inputs=[clip_selector],
        outputs=[clip_video, has_emotion, sadness_intensity, anger_intensity, pleasure_intensity, 
                sadness_intensity, anger_intensity, pleasure_intensity],
        show_progress="hidden"
    )
    
    save_btn.click(
        annotator.save_annotation,
        inputs=[clip_selector, has_emotion, sadness_intensity, anger_intensity, pleasure_intensity, task_type, annotator_name],
        outputs=[annotation_status]
    )
    
    def handle_export():
        file, status = annotator.export_csv()
        if file:
            return gr.update(value=file, visible=True), status
        return gr.update(visible=False), status
    
    export_btn.click(handle_export, outputs=[export_file, export_status])

if __name__ == "__main__":
    import socket
    
    # Get absolute paths
    clips_abs = os.path.abspath(annotator.clips_dir)
    working_abs = os.path.abspath(annotator.working_dir)
    base_abs = os.path.abspath(annotator.base_dir)
    
    print(f"\n{'='*60}")
    print(f"🎬 VIDEO EMOTION ANNOTATION TOOL")
    print(f"{'='*60}")
    print(f"📁 Current Directory: {base_abs}")
    print(f"📁 Clips Folder: clips/")
    print(f"📁 Working Folder: working/")
    print(f"{'='*60}\n")
    
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    port = 7860
    
    print(f"🌐 Local URL: http://localhost:{port}")
    print(f"🌐 Network URL: http://{local_ip}:{port}")
    print(f"{'='*60}\n")
    
    demo.queue(max_size=20)
    
    try:
        demo.launch(
            share=True,
            allowed_paths=[clips_abs, working_abs, base_abs],
            server_name="0.0.0.0",
            server_port=port,
            show_error=True,
            inbrowser=False
        )
    except Exception as e:
        print(f"Launch failed: {e}")
        import traceback
        traceback.print_exc()