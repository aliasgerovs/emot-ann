import streamlit as st
import os
import pandas as pd
from datetime import datetime
import shutil
import time
import gc
from contextlib import contextmanager
from moviepy import VideoFileClip

class VideoEmotionAnnotator:
    def __init__(self):
        # Use absolute paths in current working directory
        self.base_dir = os.path.abspath(os.getcwd())
        self.clips_dir = os.path.join(self.base_dir, 'annotation_clips')
        self.working_dir = os.path.join(self.base_dir, 'annotation_working')
        self.uploads_dir = os.path.join(self.base_dir, 'annotation_uploads')
        
        # Create all directories
        os.makedirs(self.clips_dir, exist_ok=True)
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.uploads_dir, exist_ok=True)
        
        print(f"Using clips directory: {self.clips_dir}")
        print(f"Using working directory: {self.working_dir}")
        print(f"Using uploads directory: {self.uploads_dir}")
        
        self.annotations = []
        self.current_clips = []
        self.participant_id = ""
        self.working_video_path = None
        
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
        except Exception as e:
            print(f"Error cleaning up working files: {str(e)}")
    
    @contextmanager
    def video_clip_context(self, path):
        """Context manager for MoviePy VideoFileClip"""
        clip = None
        try:
            clip = VideoFileClip(path)
            yield clip
        finally:
            if clip is not None:
                clip.close()
            time.sleep(0.2)
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def copy_video_to_working_dir(self, video_path):
        """Copy video from temp to our working directory"""
        if not video_path or not os.path.exists(video_path):
            print(f"Video path does not exist: {video_path}")
            return None
        
        try:
            # Get just the filename
            filename = os.path.basename(video_path)
            # Add timestamp to avoid conflicts
            timestamp = int(time.time() * 1000)
            # Save to our uploads directory first
            upload_path = os.path.join(self.uploads_dir, f"upload_{timestamp}_{filename}")
            
            print(f"Copying from: {video_path}")
            print(f"Copying to: {upload_path}")
            
            # Copy file in chunks
            with open(video_path, 'rb') as src:
                with open(upload_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
            
            if not os.path.exists(upload_path):
                print("Error: Upload file was not created")
                return None
            
            file_size = os.path.getsize(upload_path)
            print(f"Uploaded file size: {file_size} bytes")
            
            # Now copy to working directory
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
    
    def get_video_info(self, video_path):
        if not video_path:
            return 0, 0, 0
        
        working_path = self.copy_video_to_working_dir(video_path)
        if not working_path:
            return 0, 0, 0
        
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

    def process_video(self, video_file, participant_id, start_time, interval, process_minutes):
        if not video_file or not participant_id:
            return "Please upload a video and provide a participant ID.", []
        
        st.write("Starting video processing...")
        
        # Get the video path from Streamlit
        if isinstance(video_file, str):
            video_path = video_file
        else:
            video_path = video_file.name
        
        print(f"Received video path: {video_path}")
        
        if not os.path.exists(video_path):
            return f"Video file not found: {video_path}", []
            
        self.participant_id = participant_id
        
        st.write("Cleaning up old files...")
        
        # Clean up old clips
        for clip in self.current_clips:
            try:
                if os.path.exists(clip['path']):
                    os.remove(clip['path'])
                    time.sleep(0.1)
            except Exception as e:
                print(f"Could not remove old clip: {str(e)}")
        
        # Clean up old working video
        if self.working_video_path and os.path.exists(self.working_video_path):
            try:
                os.remove(self.working_video_path)
                time.sleep(0.3)
            except Exception as e:
                print(f"Could not remove old working video: {str(e)}")
        
        st.write("Copying video to working directory...")
        
        # Copy video to our controlled directory
        self.working_video_path = self.copy_video_to_working_dir(video_path)
        if not self.working_video_path:
            return "Failed to prepare video file. Please try again.", []
        
        time.sleep(0.5)
        
        st.write("Reading video information...")
        
        fps = 0
        duration = 0
        
        try:
            with self.video_clip_context(self.working_video_path) as clip:
                fps = clip.fps
                duration = clip.duration
                print(f"Video info - Duration: {duration}s, FPS: {fps}")
        except Exception as e:
            print(f"Error reading video: {e}")
            import traceback
            traceback.print_exc()
            return "Failed to read video file. Please try again.", []
        
        time.sleep(0.3)
        
        if duration == 0:
            return "Failed to read video file. Please try again.", []
        
        st.write("Calculating clip parameters...")
        
        requested_seconds = (process_minutes or 0) * 60
        if requested_seconds <= 0:
            requested_seconds = 240
        end_time = min(start_time + requested_seconds, duration)
        
        if end_time <= start_time:
            return "Start time is too close to the end of the video. Please choose an earlier start time.", []
        
        time_range_msg = f"Processing from {self.format_time(start_time)} to {self.format_time(end_time)} (duration: {self.format_time(end_time - start_time)})"
        self.annotations = []
        
        st.write("Starting clip extraction...")
        
        try:
            time.sleep(0.3)
            with self.video_clip_context(self.working_video_path) as video:
                fps = video.fps
                self.current_clips = []
                clip_num = 1
                
                # Calculate total number of clips
                current_time = start_time
                total_clips = 0
                while current_time < end_time:
                    total_clips += 1
                    current_time += interval
                
                print(f"Will create {total_clips} clips")
                
                current_clip_idx = 0
                current_time = start_time
                
                with st.spinner(f"Extracting clips..."):
                    while current_time < end_time:
                        clip_end_time = min(current_time + interval, end_time)
                        
                        st.write(f"Extracting clip {clip_num}/{total_clips} ({self.format_time(current_time)} - {self.format_time(clip_end_time)})")
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        clip_filename = f"clip_{self.participant_id}_{timestamp}_{clip_num:03d}.mp4"
                        output_path = os.path.join(self.clips_dir, clip_filename)
                        
                        print(f"Creating clip {clip_num} at {output_path}")
                        
                        try:
                            # Extract subclip using MoviePy
                            subclip = video.subclip(current_time, clip_end_time)
                            
                            # Use our working directory for temp files
                            temp_audiofile = os.path.join(self.working_dir, f'temp_audio_{timestamp}_{clip_num}.m4a')
                            
                            subclip.write_videofile(
                                output_path,
                                codec='libx264',
                                audio_codec='aac',
                                temp_audiofile=temp_audiofile,
                                remove_temp=True,
                                logger=None,
                                verbose=False,
                                threads=4,
                                preset='ultrafast'
                            )
                            subclip.close()
                            
                            time.sleep(0.2)
                            
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                clip_label = f"Clip {clip_num:03d} ({self.format_time(current_time)} - {self.format_time(clip_end_time)})"
                                self.current_clips.append({
                                    'label': clip_label,
                                    'path': output_path,
                                    'start': current_time,
                                    'end': clip_end_time,
                                    'number': clip_num
                                })
                                print(f"Successfully saved clip to: {output_path}")
                            else:
                                print(f"Warning: Clip file not created or is empty: {output_path}")
                        
                        except Exception as e:
                            print(f"Error creating clip {clip_num}: {str(e)}")
                            import traceback
                            traceback.print_exc()
                        
                        clip_num += 1
                        current_clip_idx += 1
                        current_time += interval
                        
                        # Clean up memory
                        gc.collect()
        
        except Exception as e:
            print(f"Error during clip extraction: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Failed to process video: {str(e)}", []
        
        st.write("Finalizing...")
        time.sleep(0.5)
        
        if not self.current_clips:
            return "No clips were successfully created. Please try again.", []
        
        clip_choices = [clip['label'] for clip in self.current_clips]
        
        st.write("Complete!")
        
        success_msg = f"""✅ **Video processed successfully!**

{time_range_msg}

📊 **Results:**
- Created {len(self.current_clips)} clips
- Clip interval: {interval} seconds
- Ready for annotation

Select a clip below to begin annotating."""
        
        return success_msg, clip_choices

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

# Streamlit CSS
st.markdown("""
    <style>
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
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    .video-container {
        max-height: 300px;
        overflow: auto;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<div style='text-align: center; padding: 20px;'><h1>🎬 Video Emotion Annotation Tool</h1><p>Process videos, annotate emotions, and export results</p></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    
    # Step 1: Upload & Configure
    st.markdown("<div class='step-header'>⚙️ Step 1: Upload & Configure</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        video_input = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], accept_multiple_files=False)
        participant_id = st.text_input("Participant ID", placeholder="Enter participant identifier")
        
        col_task, col_annotator = st.columns(2)
        with col_task:
            task_type = st.selectbox("Task Type", ["Traditional", "Digital"], index=0)
        with col_annotator:
            annotator_name = st.text_input("Annotator Name", placeholder="Enter annotator name")
        
        upload_status = st.empty()
        
        col_start1, col_start2 = st.columns(2)
        with col_start1:
            st.markdown("**Start Time**")
            start_minutes = st.number_input("Minutes", min_value=0, value=0, step=1)
        with col_start2:
            st.markdown("<br>", unsafe_allow_html=True)
            start_seconds = st.number_input("Seconds", min_value=0, max_value=59, value=0, step=1)
        
        col_interval, col_duration = st.columns(2)
        with col_interval:
            interval = st.number_input("Clip Interval (seconds)", min_value=1, value=5, step=1)
        with col_duration:
            process_minutes = st.number_input("Process Duration (minutes)", min_value=1, value=1, step=1)
        
        start_time = (start_minutes or 0) * 60 + (start_seconds or 0)
        
        if st.button("🎬 Process Video"):
            if video_input is not None:
                upload_status.markdown(*on_video_upload(video_input))
                with st.spinner("Processing video..."):
                    process_status_text, clip_choices = annotator.process_video(video_input, participant_id, start_time, interval, process_minutes)
                    st.session_state['clip_choices'] = clip_choices
                    st.session_state['process_status'] = process_status_text
            else:
                upload_status.markdown("Please upload a video file.")
        
        if 'process_status' in st.session_state:
            st.markdown(st.session_state['process_status'])
    
    with col2:
        st.markdown("<div class='step-header'>📊 Progress Overview</div>", unsafe_allow_html=True)
        st.markdown("Upload your video and set parameters to begin annotation.")
    
    # Step 2: Annotate Clips
    st.markdown("<div class='step-header'>🎯 Step 2: Annotate Clips</div>", unsafe_allow_html=True)
    
    col_clip, col_annotate = st.columns([2, 1])
    
    with col_clip:
        clip_choices = st.session_state.get('clip_choices', [])
        clip_selector = st.selectbox("Select Clip", clip_choices)
        
        if clip_selector:
            clip_path = annotator.load_clip(clip_selector)
            if clip_path and os.path.exists(clip_path):
                st.session_state['current_clip_path'] = clip_path
                st.video(clip_path)
            else:
                st.markdown("⚠️ Could not load clip.")
    
    with col_annotate:
        has_emotion = st.checkbox("NC (No Clear Emotion)", value=False)
        
        st.markdown("**Emotion Intensities (1-3):**")
        sadness_intensity = st.slider("😢 Sadness", min_value=0, max_value=3, value=0, step=1, disabled=has_emotion)
        anger_intensity = st.slider("😠 Anger", min_value=0, max_value=3, value=0, step=1, disabled=has_emotion)
        pleasure_intensity = st.slider("😊 Pleasure", min_value=0, max_value=3, value=0, step=1, disabled=has_emotion)
        
        if st.button("💾 Save Annotation"):
            if clip_selector:
                annotation_status = annotator.save_annotation(
                    clip_selector, has_emotion, sadness_intensity, anger_intensity, pleasure_intensity, task_type, annotator_name
                )
                st.markdown(annotation_status)
            else:
                st.markdown("⚠️ Please select a clip first.")
    
    # Step 3: Export Results
    st.markdown("<div class='step-header'>📤 Step 3: Export Results</div>", unsafe_allow_html=True)
    
    if st.button("📊 Export CSV"):
        file, status = annotator.export_csv()
        st.markdown(status)
        if file:
            with open(file, "rb") as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name=os.path.basename(file),
                    mime="text/csv"
                )
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Set Streamlit temp directory to our controlled location
    streamlit_temp_dir = os.path.join(annotator.base_dir, 'streamlit_temp')
    os.environ['STREAMLIT_UPLOAD_DIR'] = streamlit_temp_dir
    os.makedirs(streamlit_temp_dir, exist_ok=True)
    
    # Get absolute paths
    clips_abs = os.path.abspath(annotator.clips_dir)
    working_abs = os.path.abspath(annotator.working_dir)
    uploads_abs = os.path.abspath(annotator.uploads_dir)
    streamlit_temp_abs = os.path.abspath(streamlit_temp_dir)
    cwd_abs = os.path.abspath(os.getcwd())
    
    print(f"\nCurrent working directory: {cwd_abs}")
    print(f"Clips directory: {clips_abs}")
    print(f"Working directory: {working_abs}")
    print(f"Uploads directory: {uploads_abs}")
    print(f"Streamlit temp directory: {streamlit_temp_abs}")
    
    main()