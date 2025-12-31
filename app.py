from flask import Flask, render_template, request, Response, jsonify, send_from_directory
from utils import process_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
import threading
import queue
import logging
import traceback
import json
from flask import make_response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output_videos'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables for streaming
frame_queue = queue.Queue(maxsize=10)          # main annotated camera frames
proj_frame_queue = queue.Queue(maxsize=10)     # projection / position map frames
processing_active = False
processing_lock = threading.Lock()

# possession shared state (updated by processing thread, read by /possession)
poss_data = {'club1': 0.5, 'club2': 0.5, 'timestamp': 0.0}
poss_lock = threading.Lock()

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_processor(club1_name, club1_color, club1_gk_color,
                        club2_name, club2_color, club2_gk_color,
                        obj_conf=0.5, ball_conf=0.05, field_conf=0.3, kp_conf=0.7):
    """Initialize all models and processors"""

    # 1. Load the object detection model
    obj_tracker = ObjectTracker(
        model_path='models/weights/object-detection.pt',
        conf=obj_conf,
        ball_conf=ball_conf
    )

    # 2. Load the keypoints detection model
    kp_tracker = KeypointsTracker(
        model_path='models/weights/keypoints-detection.pt',
        conf=field_conf,
        kp_conf=kp_conf,
    )

    # 3. Assign clubs
    club1 = Club(club1_name, club1_color, club1_gk_color)
    club2 = Club(club2_name, club2_color, club2_gk_color)
    club_assigner = ClubAssigner(club1, club2)

    # 4. Initialize the BallToPlayerAssigner
    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    # 5. Define the keypoints for a top-down view
    top_down_keypoints = np.array([
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],
        [32, 122], [32, 229],
        [64, 176],
        [96, 57], [96, 122], [96, 229], [96, 293],
        [263, 0], [263, 122], [263, 229], [263, 351],
        [431, 57], [431, 122], [431, 229], [431, 293],
        [463, 176],
        [495, 122], [495, 229],
        [527, 0], [527, 57], [527, 122], [527, 229], [527, 293], [527, 351],
        [210, 176], [317, 176]
    ])

    # 6. Initialize the video processor
    processor = FootballVideoProcessor(
        obj_tracker,
        kp_tracker,
        club_assigner,
        ball_player_assigner,
        top_down_keypoints,
        field_img_path='videos/field_2d_v2.png',
        save_tracks_dir=app.config['OUTPUT_FOLDER'],
        draw_frame_num=True
    )

    return processor

def _make_placeholder_jpeg(text: str = "No frame", width: int = 640, height: int = 360) -> bytes:
    """
    Create a simple black image with white text and return JPEG bytes.
    Used to push a visible frame to the queues on error so the UI shows something.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    try:
        _, buf = cv2.imencode('.jpg', img)
        return buf.tobytes()
    except Exception:
        return b''

def process_video_stream(video_path, processor):
    """Process video and stream frames in real-time"""
    global processing_active, poss_data

    cap = cv2.VideoCapture(video_path)

    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0  # Default to 30 FPS if unable to detect

    try:
        while cap.isOpened() and processing_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame through the processor with proper FPS
            try:
                cam_frame, proj_frame = processor.process_frame(frame, fps)

                # Update latest possession numbers from the processor's ball assigner (if available)
                try:
                    possessions = processor.ball_to_player_assigner.get_ball_possessions()
                    if possessions and len(possessions) > 0:
                        latest = possessions[-1]
                        # expected pair (club1_frac, club2_frac)
                        with poss_lock:
                            poss_data['club1'] = float(latest[0])
                            poss_data['club2'] = float(latest[1])
                            poss_data['timestamp'] = cv2.getTickCount() / cv2.getTickFrequency()
                except Exception:
                    # ignore if getter not available or errors
                    pass

                # Rotate projection to vertical orientation to save horizontal space on UI
                try:
                    proj_frame = cv2.rotate(proj_frame, cv2.ROTATE_90_CLOCKWISE)
                except Exception:
                    pass

                # Encode camera frame
                try:
                    _, cam_buffer = cv2.imencode('.jpg', cam_frame)
                    cam_bytes = cam_buffer.tobytes()
                except Exception:
                    cam_bytes = _make_placeholder_jpeg("Camera encode error")

                # Encode projection frame
                try:
                    _, proj_buffer = cv2.imencode('.jpg', proj_frame)
                    proj_bytes = proj_buffer.tobytes()
                except Exception:
                    proj_bytes = _make_placeholder_jpeg("Projection encode error")

                # Put camera frame in queue (drop old frames if queue is full)
                try:
                    if cam_bytes:
                        frame_queue.put(cam_bytes, block=False)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put(cam_bytes, block=False)
                    except:
                        pass

                # Put projection frame in its queue (drop old frames if queue is full)
                try:
                    if proj_bytes:
                        proj_frame_queue.put(proj_bytes, block=False)
                except queue.Full:
                    try:
                        proj_frame_queue.get_nowait()
                        proj_frame_queue.put(proj_bytes, block=False)
                    except:
                        pass

            except Exception as e:
                # Log the full traceback, mark processing as stopped and push placeholder frames
                logging.error("Exception in process_video_stream: %s", e)
                traceback.print_exc()
                with processing_lock:
                    processing_active = False

                # push visible placeholders so client sees an error frame instead of blank
                err_cam = _make_placeholder_jpeg("Processing error", 640, 360)
                err_proj = _make_placeholder_jpeg("Processing error", 640, 360)
                try:
                    frame_queue.put(err_cam, block=False)
                except:
                    pass
                try:
                    proj_frame_queue.put(err_proj, block=False)
                except:
                    pass
                break

    finally:
        cap.release()
        processing_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing"""
    global processing_active

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Get club configuration from form
    club1_name = request.form.get('club1_name', 'Team 1')
    club1_color = tuple(map(int, request.form.get('club1_color', '199,207,198').split(',')))
    club1_gk_color = tuple(map(int, request.form.get('club1_gk_color', '11,136,194').split(',')))

    club2_name = request.form.get('club2_name', 'Team 2')
    club2_color = tuple(map(int, request.form.get('club2_color', '108,38,34').split(',')))
    club2_gk_color = tuple(map(int, request.form.get('club2_gk_color', '211,207,47').split(',')))

    # Get confidence configuration from form
    obj_conf = float(request.form.get('obj_conf', 0.5))
    ball_conf = float(request.form.get('ball_conf', 0.05))
    field_conf = float(request.form.get('field_conf', 0.3))
    kp_conf = float(request.form.get('kp_conf', 0.7))

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Initialize processor
    try:
        processor = initialize_processor(
            club1_name, club1_color, club1_gk_color,
            club2_name, club2_color, club2_gk_color,
            obj_conf, ball_conf, field_conf, kp_conf
        )

        # Start processing in background thread
        with processing_lock:
            processing_active = True

        thread = threading.Thread(
            target=process_video_stream,
            args=(filepath, processor),
            daemon=True
        )
        thread.start()

        return jsonify({'success': True, 'message': 'Processing started'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route (annotated camera frames)"""
    def generate():
        while processing_active or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except queue.Empty:
                continue

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/projection_feed')
def projection_feed():
    """Projection / position map streaming route"""
    def generate():
        while processing_active or not proj_frame_queue.empty():
            try:
                frame = proj_frame_queue.get(timeout=1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except queue.Empty:
                continue

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/possession')
def possession():
    """Return latest possession percentages as JSON (club1, club2)."""
    with poss_lock:
        data = {'club1': round(float(poss_data.get('club1', 0.5)), 3),
                'club2': round(float(poss_data.get('club2', 0.5)), 3),
                'timestamp': float(poss_data.get('timestamp', 0.0))}
    return jsonify(data)

@app.route('/stop', methods=['POST'])
def stop_processing():
    """Stop video processing"""
    global processing_active
    with processing_lock:
        processing_active = False
    return jsonify({'success': True, 'message': 'Processing stopped'})

@app.route('/status')
def status():
    """Get processing status"""
    return jsonify({'processing': processing_active})

@app.route('/outputs/<filename>')
def download_output(filename):
    """Download processed video"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
