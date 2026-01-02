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

# possession (seconds) store for temporary initialization and accumulation
poss_seconds = {'club1': 0.0, 'club2': 0.0}
poss_seconds_lock = threading.Lock()

# store the currently configured team colors (R,G,B tuples)
club_colors = {'club1': (199, 207, 198), 'club2': (108, 38, 34)}
club_colors_lock = threading.Lock()

# projection visualization mode shared state: 'heatmap' | 'voronoi' | 'both'
projection_mode = 'heatmap'
projection_lock = threading.Lock()

logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def rgb_tuple_to_hex(t: tuple) -> str:
    """Convert (R,G,B) tuple to hex string for CSS (assumes 0-255 ints)."""
    try:
        r, g, b = int(t[0]), int(t[1]), int(t[2])
        return f'#{r:02x}{g:02x}{b:02x}'
    except Exception:
        return '#ffffff'

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
    global processing_active, poss_data, projection_mode, club_colors, poss_seconds

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

            # read latest projection_mode and set on processor so annotate respects it
            try:
                with projection_lock:
                    processor.projection_mode = projection_mode
            except Exception:
                processor.projection_mode = 'heatmap'

            # Process frame through the processor with proper FPS
            try:
                cam_frame, proj_frame = processor.process_frame(frame, fps)

                # Update latest possession numbers from the processor's ball assigner (if available)
                try:
                    possessions = processor.ball_to_player_assigner.get_ball_possessions()
                    if possessions and len(possessions) > 0:
                        latest = possessions[-1]
                        # expected pair (club1_frac, club2_frac)
                        raw_c1 = float(latest[0]) if len(latest) > 0 else 0.5
                        raw_c2 = float(latest[1]) if len(latest) > 1 else 0.5

                        # clamp and normalize so sum == 1.0 (fractions for this frame)
                        raw_c1 = max(0.0, min(1.0, raw_c1))
                        raw_c2 = max(0.0, min(1.0, raw_c2))
                        s = raw_c1 + raw_c2
                        if s <= 1e-6:
                            frac1, frac2 = 0.5, 0.5
                        else:
                            frac1, frac2 = raw_c1 / s, raw_c2 / s

                        # accumulate possession seconds (delta = 1/fps)
                        delta = 1.0 / max(fps, 1e-6)
                        with poss_seconds_lock:
                            poss_seconds['club1'] = poss_seconds.get('club1', 0.0) + frac1 * delta
                            poss_seconds['club2'] = poss_seconds.get('club2', 0.0) + frac2 * delta

                        # update poss_data snapshots for backward compatibility (normalize seconds -> fractions)
                        with poss_lock, poss_seconds_lock:
                            ssec = poss_seconds.get('club1', 0.0) + poss_seconds.get('club2', 0.0)
                            if ssec <= 1e-6:
                                poss_data['club1'] = 0.5
                                poss_data['club2'] = 0.5
                            else:
                                poss_data['club1'] = poss_seconds['club1'] / ssec
                                poss_data['club2'] = poss_seconds['club2'] / ssec
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
    global processing_active, club_colors, poss_seconds

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

        # store chosen club colors for the web UI
        with club_colors_lock:
            club_colors['club1'] = club1_color
            club_colors['club2'] = club2_color

        # Temporary initialization: give each team 30 seconds possession at load
        with poss_seconds_lock:
            poss_seconds['club1'] = 30.0
            poss_seconds['club2'] = 30.0
        # sync poss_data snapshot for UI
        with poss_lock, poss_seconds_lock:
            ssec = poss_seconds['club1'] + poss_seconds['club2']
            if ssec <= 1e-6:
                poss_data['club1'] = 0.5
                poss_data['club2'] = 0.5
            else:
                poss_data['club1'] = poss_seconds['club1'] / ssec
                poss_data['club2'] = poss_seconds['club2'] / ssec
            poss_data['timestamp'] = cv2.getTickCount() / cv2.getTickFrequency()

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
    """Return latest possession percentages as JSON (club1, club2) and colors."""
    # compute percentages from accumulated seconds (preferred), fall back to poss_data
    with poss_seconds_lock:
        s1 = float(poss_seconds.get('club1', 0.0))
        s2 = float(poss_seconds.get('club2', 0.0))
    ssum = s1 + s2
    if ssum > 1e-6:
        c1n = s1 / ssum
        c2n = s2 / ssum
        timestamp = cv2.getTickCount() / cv2.getTickFrequency()
    else:
        with poss_lock:
            c1 = float(poss_data.get('club1', 0.5))
            c2 = float(poss_data.get('club2', 0.5))
        s = max(1e-6, c1 + c2)
        c1n = c1 / s
        c2n = c2 / s
        timestamp = float(poss_data.get('timestamp', 0.0))

    with club_colors_lock:
        col1 = club_colors.get('club1', (199, 207, 198))
        col2 = club_colors.get('club2', (108, 38, 34))
        col1_hex = rgb_tuple_to_hex(col1)
        col2_hex = rgb_tuple_to_hex(col2)

    return jsonify({
        'club1': round(c1n, 3),
        'club2': round(c2n, 3),
        'timestamp': timestamp,
        'club1_color': col1_hex,
        'club2_color': col2_hex,
        'club1_percent': int(round(c1n * 100)),
        'club2_percent': int(round(c2n * 100)),
        'club1_seconds': round(s1, 2),
        'club2_seconds': round(s2, 2)
    })

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

@app.route('/projection_mode', methods=['GET', 'POST'])
def projection_mode_api():
    """GET current projection mode or set it via POST (JSON or form)"""
    global projection_mode
    if request.method == 'POST':
        # try JSON first, then form field
        data_mode = None
        try:
            data = request.get_json(silent=True)
            if isinstance(data, dict) and 'mode' in data:
                data_mode = data['mode']
        except Exception:
            data_mode = None
        if data_mode is None:
            data_mode = request.form.get('projection_mode', None)
        if data_mode is None:
            return jsonify({'error': 'no mode supplied'}), 400
        data_mode = str(data_mode).lower()
        if data_mode not in ('heatmap', 'voronoi', 'both'):
            return jsonify({'error': 'invalid mode'}), 400
        with projection_lock:
            projection_mode = data_mode
        return jsonify({'success': True, 'mode': projection_mode})
    else:
        with projection_lock:
            return jsonify({'mode': projection_mode})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
