# Football Analysis Web App

This web application allows you to upload football match videos and process them in real-time with object detection, player tracking, and analysis.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload a video file and configure team settings
4. Click "Start Analysis" to begin real-time processing
5. Watch the processed video stream in your browser

## Features

- **Video Upload**: Upload football match videos (MP4, AVI, MOV, MKV)
- **Team Configuration**: Customize team names and jersey colors
- **Real-time Processing**: Stream processed frames as they're analyzed
- **Player Tracking**: Track players, ball, and goalkeepers
- **Speed Estimation**: Calculate player speeds
- **Ball Possession**: Track which team has possession
- **Perspective Transformation**: View player positions on a 2D field map

## Configuration

### Team Settings

You can customize the following for each team:
- **Team Name**: Display name for the team
- **Jersey Color**: RGB values for player jerseys (e.g., `199,207,198`)
- **Goalkeeper Color**: RGB values for goalkeeper jerseys (e.g., `11,136,194`)

### Advanced Settings

Modify `app.py` to adjust:
- `MAX_CONTENT_LENGTH`: Maximum upload file size (default: 500MB)
- `batch_size`: Number of frames to process at once (in `process_video_stream`)
- Model confidence thresholds in `initialize_processor()`

## Important Note: Real-time Frame Processing

The current implementation uses the batch processing method from `FootballVideoProcessor`. For **true frame-by-frame real-time processing**, you need to add a `process_frame()` method to the `FootballVideoProcessor` class.

### Adding Real-time Frame Processing

Add this method to `annotation/football_video_processor.py`:

```python
def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
    """
    Process a single frame in real-time.

    Args:
        frame (np.ndarray): Single video frame
        frame_number (int): Current frame number

    Returns:
        np.ndarray: Annotated frame
    """
    # Detect objects and keypoints
    obj_detection = self.obj_tracker.detect([frame])[0]
    kp_detection = self.kp_tracker.detect([frame])[0]

    # Track
    obj_tracks = self.obj_tracker.track(obj_detection)
    kp_tracks = self.kp_tracker.track(kp_detection)

    # Process tracks (assign clubs, map positions, etc.)
    obj_tracks = self._process_tracks(frame, obj_tracks, kp_tracks)

    # Annotate frame
    annotated_frame = self.annotate(frame, obj_tracks, kp_tracks)

    return annotated_frame

def _process_tracks(self, frame: np.ndarray, obj_tracks: dict, kp_tracks: dict) -> dict:
    """Helper method to process tracks for a single frame"""
    # Assign clubs to players
    obj_tracks = self.club_assigner.assign(frame, obj_tracks)

    # Estimate speeds
    obj_tracks = self.speed_estimator.estimate(obj_tracks, self.cur_fps if hasattr(self, 'cur_fps') else 30.0)

    # Map positions
    obj_tracks = self.obj_mapper.map(obj_tracks, kp_tracks)

    # Assign ball to player
    obj_tracks = self.ball_to_player_assigner.assign(obj_tracks)

    return obj_tracks
```

## Folder Structure

```
Football-Analysis/
├── app.py                  # Flask web application
├── main.py                 # Original CLI script
├── templates/
│   └── index.html         # Web interface
├── uploads/               # Uploaded videos (created automatically)
├── output_videos/         # Processed videos (created automatically)
├── models/
│   └── weights/          # Model weights
├── videos/
│   └── field_2d_v2.png   # Field visualization image
└── requirements.txt       # Python dependencies
```

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload video and start processing
- `GET /video_feed` - Video stream endpoint (MJPEG)
- `POST /stop` - Stop current processing
- `GET /status` - Check processing status
- `GET /outputs/<filename>` - Download processed videos

## Troubleshooting

### Video not streaming
- Ensure models are loaded correctly in `models/weights/`
- Check console for error messages
- Verify video file format is supported

### Slow processing
- Reduce video resolution before uploading
- Adjust batch_size in `process_video_stream()`
- Lower model confidence thresholds

### Out of memory
- Reduce `MAX_CONTENT_LENGTH`
- Process smaller video files
- Reduce batch_size

## Performance Tips

1. **GPU Acceleration**: Ensure PyTorch is using CUDA for faster processing
2. **Smaller Videos**: Process shorter clips or lower resolutions for faster results
3. **Batch Size**: Adjust based on your system's memory (lower = less memory, slower processing)
4. **Queue Size**: Modify `frame_queue.Queue(maxsize=10)` to buffer more/fewer frames

## Security Notes

- The app accepts video uploads up to 500MB
- Files are saved to the `uploads/` directory
- Consider adding authentication for production use
- Add file validation and sanitization for public deployments
