# Fire Detection System

This repository contains a Flask-based web application for real-time fire detection using a YOLOv8 model. The application processes video input from either a file or an ESP32-CAM stream and provides a live video feed with fire detection bounding boxes.

## Features

- **Real-Time Fire Detection**: Detects fire in video streams using a pre-trained YOLOv8 model.
- **Video Input Options**: Supports video files and live streams from ESP32-CAM.
- **Live Output Stream**: Provides a live video feed with detection results via a web interface.
- **Optimized Performance**: Processes every 2nd frame to improve FPS.

## Requirements

The following dependencies are required to run the application:

- Python 3.9 or higher
- Flask
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLO
- Supervision
- Roboflow

Install the dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## File Structure

```
backend/
├── fire_detection_inference.py  # Main application script
├── requirements.txt             # Python dependencies
├── models/                      # Directory for YOLO model files
│   ├── v8n.pt                   # YOLOv8n model
│   ├── v8n50epoch.pt            # YOLOv8n model trained for 50 epochs
├── .gitignore                   # Git ignore file
```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Place the YOLO Model**:
   Ensure the YOLO model file (`v8n50epoch.pt`) is located in the `models/` directory.

3. **Run the Application**:
   Start the Flask server:
   ```bash
   python fire_detection_inference.py
   ```

4. **Access the Web Interface**:
   Open your browser and navigate to:
   ```
   http://<your-ip>:5000/
   ```

5. **Specify Input Source**:
   - Enter the path to a video file or the URL of an ESP32-CAM stream.
   - Click "Start Detection" to begin processing.

6. **View the Output Stream**:
   - Click the "View Output Stream" link on the web interface.
   - The processed video feed with fire detection will be displayed.

7. **Stop Detection**:
   - Navigate to `/stop` to stop the detection process.

## Notes

- The application uses the `v8n50epoch.pt` model by default. You can update the model path in `fire_detection_inference.py` if needed.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLOv8 model.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [OpenCV](https://opencv.org/) for video processing.