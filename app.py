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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output_videos'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables for streaming
frame_queue = queue.Queue(maxsize=10)
processing_active = False
processing_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_processor(club1_name, club1_color, club1_gk_color,
                        club2_name, club2_color, club2_gk_color):
    """Initialize all models and processors"""

    # 1. Load the object detection model
    obj_tracker = ObjectTracker(
        model_path='models/weights/object-detection.pt',
        conf=.5,
        ball_conf=.05
    )

    # 2. Load the keypoints detection model
    kp_tracker = KeypointsTracker(
        model_path='models/weights/keypoints-detection.pt',
        conf=.3,
        kp_conf=.7,
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

def process_video_stream(video_path, processor):
    """Process video and stream frames in real-time"""
    global processing_active

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
            processed_frame = processor.process_frame(frame, fps)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            # Put frame in queue (drop old frames if queue is full)
            try:
                frame_queue.put(frame_bytes, block=False)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put(frame_bytes, block=False)
                except:
                    pass

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

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Initialize processor
    try:
        processor = initialize_processor(
            club1_name, club1_color, club1_gk_color,
            club2_name, club2_color, club2_gk_color
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
    """Video streaming route"""
    def generate():
        while processing_active or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=1)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except queue.Empty:
                continue

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
