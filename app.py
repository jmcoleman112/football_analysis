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

# draw_keypoints shared state (enable/disable drawing keypoints on camera frame)
draw_keypoints = True
draw_keypoints_lock = threading.Lock()

# Processing performance / optimization state
# processing_mode: 'quality' | 'balanced' | 'speed'
processing_mode = 'quality'
processing_mode_lock = threading.Lock()

# input resolution scale applied before detection/tracking (0.25 .. 1.0)
input_scale = 1.0
input_scale_lock = threading.Lock()

# Global for current processor to enable exporting logs
current_processor = None

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
                        obj_conf=0.5, ball_conf=0.05, field_conf=0.3, kp_conf=0.7,
                        draw_kp: bool = True):
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
        draw_frame_num=True,
        draw_heatmap=False,
        draw_keypoints=draw_kp
    )

    # apply initial optimization settings (will be overriden live via endpoints)
    processor.input_scale = float(1.0)
    processor.detection_interval = 1

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
    global processing_active, poss_data, projection_mode, club_colors, poss_seconds, draw_keypoints
    global processing_mode, input_scale

    cap = cv2.VideoCapture(video_path)

    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0  # Default to 30 FPS if unable to detect

    frame_idx = 0

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

            # read latest draw_keypoints flag so annotate respects it
            try:
                with draw_keypoints_lock:
                    processor.draw_keypoints = draw_keypoints
            except Exception:
                processor.draw_keypoints = True

            # read current processing_mode and input_scale
            try:
                with processing_mode_lock:
                    pmode = processing_mode
            except Exception:
                pmode = 'quality'
            try:
                with input_scale_lock:
                    scale = float(input_scale)
            except Exception:
                scale = 1.0

            # map mode -> detection interval
            if pmode == 'quality':
                interval = 1
            elif pmode == 'balanced':
                interval = 3   # detect every 3rd frame (tuneable)
            elif pmode == 'speed':
                interval = 5
            else:
                interval = 1

            # prepare frame for processor (apply input scaling if requested)
            h, w = frame.shape[:2]
            proc_frame = frame
            did_resize = False
            if scale > 0 and scale < 0.999:
                try:
                    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                    proc_frame = cv2.resize(frame, (nw, nh))
                    did_resize = True
                except Exception:
                    proc_frame = frame
                    did_resize = False

            try:
                # choose detection vs. tracking-only path
                if (frame_idx % interval) == 0:
                    # run full detection+process
                    cam_frame, proj_frame = processor.process_frame(proc_frame, fps)
                else:
                    # run tracking-only process (no detector)
                    cam_frame, proj_frame = processor.process_without_detection(proc_frame, fps)
            except Exception:
                # fallback to safe call
                try:
                    cam_frame, proj_frame = processor.process_frame(proc_frame, fps)
                except Exception:
                    cam_frame = _make_placeholder_jpeg("Processing error")
                    proj_frame = _make_placeholder_jpeg("Processing error")

            frame_idx += 1

            # If we processed on a scaled frame, resize outputs back to original size for UI
            if did_resize:
                try:
                    cam_frame = cv2.resize(cam_frame, (w, h))
                except Exception:
                    pass
                try:
                    proj_frame = cv2.resize(proj_frame, (w, h))
                except Exception:
                    pass

            # Update latest possession numbers from the processor's ball assigner (if available)
            try:
                possessions = processor.ball_to_player_assigner.get_ball_possessions()
                if possessions and len(possessions) > 0:
                    latest = possessions[-1]
                    raw_c1 = float(latest[0]) if len(latest) > 0 else 0.5
                    raw_c2 = float(latest[1]) if len(latest) > 1 else 0.5
                    raw_c1 = max(0.0, min(1.0, raw_c1))
                    raw_c2 = max(0.0, min(1.0, raw_c2))
                    s = raw_c1 + raw_c2
                    if s <= 1e-6:
                        frac1, frac2 = 0.5, 0.5
                    else:
                        frac1, frac2 = raw_c1 / s, raw_c2 / s

                    delta = 1.0 / max(fps, 1e-6)
                    with poss_seconds_lock:
                        poss_seconds['club1'] = poss_seconds.get('club1', 0.0) + frac1 * delta
                        poss_seconds['club2'] = poss_seconds.get('club2', 0.0) + frac2 * delta

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

    finally:
        cap.release()
        processing_active = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing"""
    global processing_active, club_colors, poss_seconds, current_processor, processing_mode, input_scale

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

    # get initial optimization settings from form (if provided)
    try:
        fm_mode = request.form.get('processing_mode', None)
        if fm_mode:
            fm_mode = fm_mode.lower()
            if fm_mode in ('quality', 'balanced', 'speed'):
                with processing_mode_lock:
                    processing_mode = fm_mode
    except Exception:
        pass

    try:
        fm_scale = request.form.get('input_scale', None)
        if fm_scale:
            sf = float(fm_scale)
            with input_scale_lock:
                input_scale = max(0.2, min(1.0, sf))
    except Exception:
        pass

    # Initialize processor
    try:
        processor = initialize_processor(
            club1_name, club1_color, club1_gk_color,
            club2_name, club2_color, club2_gk_color,
            obj_conf, ball_conf, field_conf, kp_conf,
            draw_kp=draw_keypoints
        )

        # store chosen club colors for the web UI
        with club_colors_lock:
            club_colors['club1'] = club1_color
            club_colors['club2'] = club2_color

        # set global current processor so we can export logs later
        current_processor = processor

        # apply initial values to processor
        with input_scale_lock:
            try:
                processor.input_scale = float(input_scale)
            except Exception:
                processor.input_scale = 1.0
        with processing_mode_lock:
            # detection_interval kept in processor for potential use; main loop decides interval
            if processing_mode == 'quality':
                processor.detection_interval = 1
            elif processing_mode == 'balanced':
                processor.detection_interval = 3
            else:
                processor.detection_interval = 5

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

# New endpoint to export timings CSV
@app.route('/export_log', methods=['GET'])
def export_log():
    """Download timings CSV (timings.csv) from the current processor output folder."""
    global current_processor
    if current_processor is None:
        return jsonify({'error': 'No processor initialized'}), 404

    timings_filename = 'timings.csv'
    timings_path = os.path.join(app.config['OUTPUT_FOLDER'], timings_filename)
    if not os.path.exists(timings_path):
        return jsonify({'error': 'timings file not found', 'path': timings_path}), 404

    # send file as attachment for download
    return send_from_directory(app.config['OUTPUT_FOLDER'], timings_filename, as_attachment=True)

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

@app.route('/draw_keypoints', methods=['GET', 'POST'])
def draw_keypoints_api():
    """GET/POST to control whether keypoints are drawn on camera frames."""
    global draw_keypoints
    if request.method == 'POST':
        data_draw = None
        try:
            data = request.get_json(silent=True)
            if isinstance(data, dict) and 'draw' in data:
                data_draw = data['draw']
        except Exception:
            data_draw = None
        if data_draw is None:
            # fall back to form field
            val = request.form.get('draw_keypoints', None)
            if val is None:
                return jsonify({'error': 'no draw value supplied'}), 400
            # checkbox yields 'on' when checked
            data_draw = str(val).lower() in ('1', 'true', 'on', 'yes')
        else:
            # coerce JSON boolean-like values
            data_draw = bool(data_draw)

        with draw_keypoints_lock:
            draw_keypoints = data_draw
        return jsonify({'success': True, 'draw_keypoints': draw_keypoints})
    else:
        with draw_keypoints_lock:
            return jsonify({'draw_keypoints': draw_keypoints})

# New endpoints: processing_mode and input_scale
@app.route('/processing_mode', methods=['GET', 'POST'])
def processing_mode_api():
    """GET/POST to control processing mode (quality | balanced | speed)"""
    global processing_mode
    if request.method == 'POST':
        mode = None
        try:
            data = request.get_json(silent=True)
            if isinstance(data, dict) and 'mode' in data:
                mode = data['mode']
        except Exception:
            mode = None
        if mode is None:
            mode = request.form.get('processing_mode', None)
        if mode is None:
            return jsonify({'error': 'no mode supplied'}), 400
        mode = str(mode).lower()
        if mode not in ('quality', 'balanced', 'speed'):
            return jsonify({'error': 'invalid mode'}), 400
        with processing_mode_lock:
            processing_mode = mode
        return jsonify({'success': True, 'processing_mode': processing_mode})
    else:
        with processing_mode_lock:
            return jsonify({'processing_mode': processing_mode})

@app.route('/input_scale', methods=['GET', 'POST'])
def input_scale_api():
    """GET/POST to control input resolution scale applied before detection/tracking (0.25 .. 1.0)"""
    global input_scale
    if request.method == 'POST':
        val = None
        try:
            data = request.get_json(silent=True)
            if isinstance(data, dict) and 'scale' in data:
                val = float(data['scale'])
        except Exception:
            val = None
        if val is None:
            s = request.form.get('input_scale', None)
            if s is None:
                return jsonify({'error': 'no scale supplied'}), 400
            try:
                val = float(s)
            except Exception:
                return jsonify({'error': 'invalid scale'}, 400)
        val = max(0.2, min(1.0, float(val)))
        with input_scale_lock:
            input_scale = val
        return jsonify({'success': True, 'input_scale': input_scale})
    else:
        with input_scale_lock:
            return jsonify({'input_scale': input_scale})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
