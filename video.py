import gradio as gr
import cv2
import os
import pandas as pd
from datetime import datetime
import tempfile
import shutil
import time
import gc
from contextlib import contextmanager

class VideoEmotionAnnotator:
    def __init__(self):
        # Create clips directory in current working directory for reliability
        self.clips_dir = os.path.join(os.getcwd(), 'annotation_clips')
        self.working_dir = os.path.join(os.getcwd(), 'annotation_working')
        
        # Create directories if they don't exist
        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.working_dir, exist_ok=True)
        
        print(f"Using clips directory: {self.clips_dir}")
        print(f"Using working directory: {self.working_dir}")
        
        self.annotations = []
        self.current_clips = []
        self.participant_id = ""
        self.working_video_path = None
        
    def __del__(self):
        # Don't delete the clips directory - keep it for user access
        # Only clean up working files
        try:
            if hasattr(self, 'working_dir') and os.path.exists(self.working_dir):
                for file in os.listdir(self.working_dir):
                    try:
                        file_path = os.path.join(self.working_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except:
                        pass
        except Exception as e:
            print(f"Error cleaning up working files: {str(e)}")
    
    @contextmanager
    def video_capture(self, path):
        """Context manager for safe video capture handling"""
        cap = None
        try:
            cap = cv2.VideoCapture(path)
            yield cap
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(0.2)
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def copy_video_to_working_dir(self, video_path):
        """Copy the video to a working directory to avoid file locking issues"""
        if not video_path or not os.path.exists(video_path):
            return None
        
        try:
            filename = os.path.basename(video_path)
            timestamp = int(time.time() * 1000)  # millisecond precision
            working_path = os.path.join(self.working_dir, f"working_{timestamp}_{filename}")
            
            # Use shutil with explicit buffer size for better reliability
            with open(video_path, 'rb') as src:
                with open(working_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)  # 1MB buffer
            
            print(f"Copied video to: {working_path}")
            time.sleep(0.3)  # Wait for filesystem
            return working_path
        except Exception as e:
            print(f"Error copying video: {e}")
            return None
    
    def get_video_info(self, video_path):
        if not video_path:
            return 0, 0, 0
        
        working_path = self.copy_video_to_working_dir(video_path)
        if not working_path:
            return 0, 0, 0
        
        try:
            time.sleep(0.3)
            with self.video_capture(working_path) as cap:
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    return duration, fps, frame_count
        except Exception as e:
            print(f"Error getting video info: {e}")
        
        return 0, 0, 0

    def process_video(self, video_file, participant_id, start_time, interval, process_minutes, progress=gr.Progress()):
        if not video_file or not participant_id:
            return "Please upload a video and provide a participant ID.", gr.update(choices=[])
        
        progress(0, desc="Starting video processing...")
        
        # Handle Gradio file object
        if hasattr(video_file, 'name'):
            video_path = video_file.name
        elif isinstance(video_file, str):
            video_path = video_file
        else:
            return "Invalid video file.", gr.update(choices=[])
            
        self.participant_id = participant_id
        
        progress(0.05, desc="Cleaning up old files...")
        
        # Clean up old clips
        for clip in self.current_clips:
            try:
                if os.path.exists(clip['path']):
                    os.remove(clip['path'])
                    time.sleep(0.1)
            except Exception as e:
                print(f"Could not remove old clip: {str(e)}")
        
        # Clean up old working video if exists
        if self.working_video_path and os.path.exists(self.working_video_path):
            try:
                os.remove(self.working_video_path)
                time.sleep(0.3)
            except Exception as e:
                print(f"Could not remove old working video: {str(e)}")
        
        progress(0.1, desc="Copying video to working directory...")
        
        # Copy video to working directory
        self.working_video_path = self.copy_video_to_working_dir(video_path)
        if not self.working_video_path:
            return "Failed to prepare video file. Please try again.", gr.update(choices=[])
        
        time.sleep(0.5)
        
        progress(0.15, desc="Reading video information...")
        
        # Get video info with proper cleanup
        fps = 0
        frame_count = 0
        duration = 0
        
        try:
            with self.video_capture(self.working_video_path) as cap:
                if not cap.isOpened():
                    return "Failed to open video file.", gr.update(choices=[])
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
        except Exception as e:
            print(f"Error reading video: {e}")
            return "Failed to read video file. Please try again.", gr.update(choices=[])
        
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
        
        # Process clips with proper cleanup
        try:
            time.sleep(0.3)
            with self.video_capture(self.working_video_path) as cap:
                if not cap.isOpened():
                    return "Failed to open video file.", gr.update(choices=[])
                    
                fps = cap.get(cv2.CAP_PROP_FPS)
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                interval_frames = int(interval * fps)
                self.current_clips = []
                clip_num = 1
                
                # Calculate total clips for progress
                total_clips = len(range(start_frame, end_frame, interval_frames))
                current_clip_idx = 0
                
                for frame_start in range(start_frame, end_frame, interval_frames):
                    frame_end = min(frame_start + interval_frames, end_frame)
                    
                    # Update progress
                    progress_val = 0.3 + (0.6 * (current_clip_idx / max(total_clips, 1)))
                    progress(progress_val, desc=f"Creating clip {clip_num}/{total_clips}...")
                    current_clip_idx += 1
                    
                    clip_filename = f"{self.participant_id}_clip_{clip_num:03d}.mp4"
                    clip_path = os.path.join(self.clips_dir, clip_filename)
                    
                    # Remove old clip if exists
                    if os.path.exists(clip_path):
                        try:
                            os.remove(clip_path)
                            time.sleep(0.2)
                        except:
                            timestamp = int(time.time() * 1000)
                            clip_path = os.path.join(self.clips_dir, f"{self.participant_id}_clip_{timestamp}_{clip_num:03d}.mp4")
                    
                    success = self.create_clip(cap, frame_start, frame_end, clip_path, fps)
                    
                    if success and os.path.exists(clip_path):
                        time.sleep(0.2)
                        # Verify the clip is readable
                        try:
                            with self.video_capture(clip_path) as test_cap:
                                if test_cap.isOpened():
                                    self.current_clips.append({
                                        'path': clip_path,
                                        'number': clip_num,
                                        'start_time': frame_start/fps,
                                        'end_time': frame_end/fps
                                    })
                                    print(f"Clip {clip_num} created and verified: {clip_path}")
                                    clip_num += 1
                                else:
                                    print(f"Warning: Clip {clip_num} was created but cannot be opened: {clip_path}")
                        except Exception as e:
                            print(f"Error verifying clip {clip_num}: {e}")
                    
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"Error processing video: {e}")
            return f"Error processing video: {str(e)}", gr.update(choices=[])
        
        # Force garbage collection to free memory
        gc.collect()
        
        progress(1.0, desc="Finalizing...")
        time.sleep(0.3)
        
        if not self.current_clips:
            return "No clips were created. Please check your parameters.", gr.update(choices=[])
        
        clip_options = [f"Clip {clip['number']:03d} ({self.format_time(clip['start_time'])} - {self.format_time(clip['end_time'])})" 
                       for clip in self.current_clips]
        return f"{time_range_msg}\n\nCreated {len(self.current_clips)} clips successfully!", gr.update(choices=clip_options, value=clip_options[0] if clip_options else None)
    
    def create_clip(self, cap, start_frame, end_frame, output_path, fps):
        """Create a clip using an already-opened video capture object"""
        out = None
        try:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Try different codecs
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                output_path_avi = output_path.replace('.mp4', '.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path_avi, fourcc, fps, (width, height))
                if out.isOpened():
                    output_path = output_path_avi
            
            if not out.isOpened():
                print(f"Error: Could not create VideoWriter for {output_path}")
                return False
            
            # Set position to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_written = 0
            current_frame = start_frame
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_written += 1
                current_frame += 1
            
            print(f"Successfully wrote {frames_written} frames to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error creating clip: {str(e)}")
            return False
        finally:
            if out is not None:
                out.release()
            cv2.waitKey(1)
            time.sleep(0.2)
    
    def load_clip(self, selected_clip):
        if not selected_clip or not self.current_clips:
            return None
        
        try:
            clip_num_str = selected_clip.split()[1]
            clip_num = int(clip_num_str)
            
            for clip in self.current_clips:
                if clip['number'] == clip_num:
                    clip_path = clip['path']
                    if os.path.exists(clip_path):
                        print(f"Loading clip: {clip_path}")
                        return os.path.abspath(clip_path)
                    else:
                        print(f"Clip not found: {clip_path}")
                        return None
        except (ValueError, IndexError) as e:
            print(f"Error parsing clip selection: {e}")
        return None
    
    def save_annotation(self, selected_clip, has_emotion, sadness_intensity, anger_intensity, pleasure_intensity, task_type, annotator_name):
        if not selected_clip:
            return "Please select a clip first."
        
        try:
            clip_num = None
            clip_data = None
            
            clip_num_str = selected_clip.split()[1]
            clip_num = int(clip_num_str)
            clip_data = next((clip for clip in self.current_clips if clip['number'] == clip_num), None)
            
            if not clip_data:
                return "Clip not found."
        except (ValueError, IndexError):
            return "Invalid clip selection."
        
        self.annotations = [ann for ann in self.annotations if ann['clip_number'] != clip_num]
        
        annotation = {
            'participant_id': self.participant_id,
            'task_type': task_type,
            'annotator_name': annotator_name,
            'clip_number': clip_num,
            'start_time': clip_data['start_time'],
            'end_time': clip_data['end_time'],
            'duration': clip_data['end_time'] - clip_data['start_time'],
            'no_clear_emotion': has_emotion,
            'sadness_intensity': sadness_intensity if not has_emotion else 0,
            'anger_intensity': anger_intensity if not has_emotion else 0,
            'pleasure_intensity': pleasure_intensity if not has_emotion else 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.annotations.append(annotation)
        
        return f"Annotation saved for Clip {clip_num:03d}. Total annotated: {len(self.annotations)}/{len(self.current_clips)}"
    
    def export_csv(self):
        if not self.annotations:
            return None, "No annotations to export."
        
        df = pd.DataFrame(self.annotations)
        df = df.sort_values('clip_number')
        
        filename = f"{self.participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        
        return filename, f"Exported {len(self.annotations)} annotations to {filename}"

annotator = VideoEmotionAnnotator()

css = """
#main_container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

#title {
    text-align: center;
    color: #2c3e50;
    margin-bottom: 30px;
    font-weight: 300;
}

.step-header {
    background: linear-gradient(90deg, #3498db, #2980b9);
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin: 20px 0 15px 0;
    font-weight: 500;
}

#process_btn, #save_btn, #export_btn {
    background: linear-gradient(90deg, #27ae60, #229954) !important;
    border: none !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    border-radius: 6px !important;
}

#process_btn:hover, #save_btn:hover, #export_btn:hover {
    background: linear-gradient(90deg, #229954, #1e8449) !important;
}

.video-preview {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
"""

def on_video_upload(video_file):
    if not video_file:
        return (gr.update(visible=False), gr.update(maximum=999), gr.update(), gr.update(minimum=1, maximum=1, value=1, visible=True))
    
    # Handle Gradio file object
    if hasattr(video_file, 'name'):
        video_path = video_file.name
    elif isinstance(video_file, str):
        video_path = video_file
    else:
        return (gr.update(visible=False), gr.update(maximum=999), gr.update(), gr.update(minimum=1, maximum=1, value=1, visible=True))
    
    # Copy to working directory and get info
    working_path = annotator.copy_video_to_working_dir(video_path)
    if not working_path:
        return (gr.update(visible=False), gr.update(maximum=999), gr.update(), gr.update(minimum=1, maximum=1, value=1, visible=True))
    
    time.sleep(0.5)
    
    try:
        with annotator.video_capture(working_path) as cap:
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                if duration > 0:
                    max_minutes = int(duration // 60)
                    return (
                        gr.update(visible=True),
                        gr.update(maximum=max_minutes),
                        gr.update(),
                        gr.update(minimum=1, maximum=max(1, int(duration // 60)), value=1, visible=True)
                    )
    except Exception as e:
        print(f"Error in video upload: {e}")
    
    time.sleep(0.3)
    
    return (
        gr.update(visible=False),
        gr.update(maximum=999),
        gr.update(),
        gr.update(minimum=1, maximum=1, value=1, visible=True)
    )

with gr.Blocks(css=css, title="Video Emotion Annotation Tool") as demo:
    gr.HTML("<div id='main_container'>")
    
    gr.Markdown("# 🎬 Video Emotion Annotation Tool", elem_id="title")
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<div class='step-header'>📁 Step 1: Upload & Configure</div>")
            
            with gr.Row():
                video_input = gr.Video(label="Upload Video", sources=["upload"])
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
            clip_video = gr.Video(label="Current Clip", height=300, autoplay=False)
        
        with gr.Column(scale=1):
            has_emotion = gr.Checkbox(
                label="NC (No Clear Emotion)",
                value=False
            )
            
            gr.Markdown("**Emotion Intensities (1-3):**")
            sadness_intensity = gr.Slider(
                minimum=0,
                maximum=3,
                step=1,
                label="😢 Sadness",
                value=0,
                visible=True
            )
            anger_intensity = gr.Slider(
                minimum=0,
                maximum=3,
                step=1,
                label="😠 Anger",
                value=0,
                visible=True
            )
            pleasure_intensity = gr.Slider(
                minimum=0,
                maximum=3,
                step=1,
                label="😊 Pleasure",
                value=0,
                visible=True
            )
            
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
    
    video_input.upload(
        on_video_upload,
        inputs=[video_input],
        outputs=[upload_status, start_minutes, start_seconds, process_minutes]
    )
    
    start_minutes.change(
        calculate_start_time,
        inputs=[start_minutes, start_seconds],
        outputs=[start_time]
    )
    
    start_seconds.change(
        calculate_start_time,
        inputs=[start_minutes, start_seconds],
        outputs=[start_time]
    )
    
    process_btn.click(
        annotator.process_video,
        inputs=[video_input, participant_id, start_time, interval, process_minutes],
        outputs=[process_status, clip_selector],
        show_progress="full",  # Show progress bar
        api_name="process_video"  # Enable API access
    )
    
    def load_clip_and_reset_filters(selected_clip):
        if not selected_clip:
            return (None, False, 0, 0, 0, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))
        
        video_path = annotator.load_clip(selected_clip)
        print(f"Clip path to load: {video_path}")
        
        if video_path and os.path.exists(video_path):
            # Copy to a temp location that Gradio can access
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=tempfile.gettempdir())
            temp_path = temp_file.name
            temp_file.close()
            
            try:
                time.sleep(0.2)
                shutil.copy2(video_path, temp_path)
                print(f"Copied clip to temp location: {temp_path}")
                time.sleep(0.2)
                return (
                    temp_path,
                    False,
                    0,
                    0,
                    0,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True)
                )
            except Exception as e:
                print(f"Error copying clip: {e}")
                return (None, False, 0, 0, 0, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))
        else:
            print(f"Could not load clip, path exists: {os.path.exists(video_path) if video_path else 'No path'}")
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
    
    export_btn.click(
        handle_export,
        outputs=[export_file, export_status]
    )

if __name__ == "__main__":
    # Ensure clips directory exists before launch
    os.makedirs(annotator.clips_dir, exist_ok=True)
    os.makedirs(annotator.working_dir, exist_ok=True)
    
    # Launch with configurations to prevent connection issues
    demo.queue(
        max_size=10,  # Limit queue size
        default_concurrency_limit=2  # Limit concurrent users
    )
    
    demo.launch(
        share=True,
        allowed_paths=[annotator.clips_dir, annotator.working_dir],
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,  # Explicit port
        max_file_size=500 * 1024 * 1024,  # 500MB max file size
        show_error=True,  # Show detailed errors
        max_threads=10  # Limit threads
    )