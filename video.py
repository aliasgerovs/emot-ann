import gradio as gr
import cv2
import os
import pandas as pd
from datetime import datetime
import tempfile
import shutil

class VideoEmotionAnnotator:
    def __del__(self):
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                for file in os.listdir(self.temp_dir):
                    file_path = os.path.join(self.temp_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        except Exception as e:
            print(f"Error cleaning up temp files: {str(e)}")
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_video_info(self, video_file):
        if not video_file:
            return 0, 0, 0
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration, fps, frame_count

    def process_video(self, video_file, participant_id, start_time, interval, process_minutes):
        if not video_file or not participant_id:
            return "Please upload a video and provide a participant ID.", gr.update(choices=[])
        self.participant_id = participant_id
        
        for clip in self.current_clips:
            try:
                if os.path.exists(clip['path']):
                    os.remove(clip['path'])
            except Exception as e:
                print(f"Could not remove old clip {clip['path']}: {str(e)}")
        
        duration, fps, frame_count = self.get_video_info(video_file)
        requested_seconds = (process_minutes or 0) * 60
        if requested_seconds <= 0:
            requested_seconds = 240
        end_time = min(start_time + requested_seconds, duration)
        
        if end_time <= start_time:
            return "Start time is too close to the end of the video. Please choose an earlier start time.", gr.update(choices=[])
        
        time_range_msg = f"Processing from {self.format_time(start_time)} to {self.format_time(end_time)} (duration: {self.format_time(end_time - start_time)})"
        self.annotations = []
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        interval_frames = int(interval * fps)
        self.current_clips = []
        clip_num = 1
        
        for frame_start in range(start_frame, end_frame, interval_frames):
            frame_end = min(frame_start + interval_frames, end_frame)
            
            clip_path = os.path.join(self.temp_dir, f"clip_{clip_num:03d}.mp4")
            self.create_clip(video_file, frame_start/fps, frame_end/fps, clip_path)
            
            if os.path.exists(clip_path):
                self.current_clips.append({
                    'path': clip_path,
                    'number': clip_num,
                    'start_time': frame_start/fps,
                    'end_time': frame_end/fps
                })
                clip_num += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not self.current_clips:
            return "No clips were created. Please check your parameters.", gr.update(choices=[])
        
        clip_options = [f"Clip {clip['number']:03d} ({self.format_time(clip['start_time'])} - {self.format_time(clip['end_time'])})" 
                       for clip in self.current_clips]
        return f"{time_range_msg}\n\nCreated {len(self.current_clips)} clips successfully!", gr.update(choices=clip_options, value=clip_options[0] if clip_options else None)
    
    def create_clip(self, input_video, start_time, end_time, output_path):
        cap = None
        out = None
        try:
            cap = cv2.VideoCapture(input_video)
            if not cap.isOpened():
                print(f"Error: Could not open video {input_video}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                output_path_avi = output_path.replace('.mp4', '.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path_avi, fourcc, fps, (width, height))
                if out.isOpened():
                    output_path = output_path_avi
            
            if not out.isOpened():
                print(f"Error: Could not create VideoWriter for {output_path}")
                if cap:
                    cap.release()
                return
            
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            
            current_time = start_time
            frames_written = 0
            while current_time < end_time:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_written += 1
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            
            print(f"Successfully wrote {frames_written} frames to {output_path}")
            
        except Exception as e:
            print(f"Error creating clip: {str(e)}")
        finally:
            if out is not None:
                out.release()
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    
    def load_clip(self, selected_clip):
        if not selected_clip or not self.current_clips:
            return None
        
        try:
            clip_num_str = selected_clip.split()[1]
            clip_num = int(clip_num_str)
            
            for clip in self.current_clips:
                if clip['number'] == clip_num:
                    return clip['path']
        except (ValueError, IndexError):
            pass
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
            'has_emotion': has_emotion,
            'sadness_intensity': 0 if has_emotion else sadness_intensity,
            'anger_intensity': 0 if has_emotion else anger_intensity,
            'pleasure_intensity': 0 if has_emotion else pleasure_intensity,
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

.status-success {
    color: #27ae60;
    font-weight: 500;
}

.status-info {
    color: #3498db;
    font-weight: 500;
}

.time-range-slider {
    margin: 10px 0;
}

.video-preview {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.time-display {
    font-family: monospace;
    font-weight: bold;
    color: #2c3e50;
}
"""

with gr.Blocks(css=css, title="Video Emotion Annotation Tool") as demo:
    gr.HTML("<div id='main_container'>")
    
    gr.Markdown("# 🎬 Video Emotion Annotation Tool", elem_id="title")
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.HTML("<div class='step-header'>📁 Step 1: Upload & Configure</div>")
            
            with gr.Row():
                video_input = gr.File(label="Upload Video", file_types=[".mp4", ".avi", ".mov", ".mkv"])
                participant_id = gr.Textbox(label="Participant ID", placeholder="Enter participant identifier")
            
            with gr.Row():
                task_type = gr.Dropdown(label="Task Type", choices=["Traditional", "Digital"], value="Traditional")
                annotator_name = gr.Textbox(label="Annotator Name", placeholder="Enter annotator name")
            
            video_player = gr.Video(label="Video Preview", height=300, visible=False, elem_classes="video-preview")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Start Time**")
                    with gr.Row():
                        start_minutes = gr.Number(label="Minutes", value=0, minimum=0, precision=0)
                        start_seconds = gr.Number(label="Seconds", value=0, minimum=0, maximum=59, precision=0)
                with gr.Column(scale=1):
                    interval = gr.Number(label="Clip Interval (seconds)", value=5, minimum=1)
                    process_minutes = gr.Number(label="Process Duration (minutes)", value=1, minimum=1)
            
            # Hidden input that stores the actual start time in seconds
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
            clip_video = gr.Video(label="Current Clip", height=300)
        
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
    
    def on_video_upload(video_file):
        if video_file:
            duration, fps, frame_count = annotator.get_video_info(video_file)
            max_minutes = int(duration // 60)
            return (
                gr.update(value=video_file, visible=True),
                gr.update(maximum=max_minutes),
                gr.update(),
                gr.update(minimum=1, maximum=max(1, int(duration // 60)), value=1, visible=True)
            )
        return (
            gr.update(visible=False),
            gr.update(maximum=999),
            gr.update(),
            gr.update(minimum=1, maximum=1, value=1, visible=True)
        )
    
    def calculate_start_time(minutes, seconds):
        return (minutes or 0) * 60 + (seconds or 0)
    
    has_emotion.change(update_emotion_fields, inputs=[has_emotion], outputs=[sadness_intensity, anger_intensity, pleasure_intensity])
    
    video_input.change(
        on_video_upload,
        inputs=[video_input],
        outputs=[video_player, start_minutes, start_seconds, process_minutes]
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
        outputs=[process_status, clip_selector]
    )
    
    def load_clip_and_reset_filters(selected_clip):
        video_path = annotator.load_clip(selected_clip)
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
    
    clip_selector.change(
        load_clip_and_reset_filters,
        inputs=[clip_selector],
        outputs=[clip_video, has_emotion, sadness_intensity, anger_intensity, pleasure_intensity, 
                sadness_intensity, anger_intensity, pleasure_intensity]
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
    os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), 'gradio_temp')
    if not os.path.exists(os.environ['GRADIO_TEMP_DIR']):
        os.makedirs(os.environ['GRADIO_TEMP_DIR'])
    demo.launch(share=True)