# Fire Detection System

This repository contains a Flask-based web application for real-time fire detection using a YOLOv8 model. The application processes video input from either a file or an ESP32-CAM stream and provides a live video feed with fire detection bounding boxes. It also integrates sensor data for enhanced fire detection confidence.

## Features

- **Real-Time Fire Detection**: Detects fire in video streams using a pre-trained YOLOv8 model.
- **Smoke Detection**: Detects smoke when using the `50epochv11x.pt` model.
- **Sensor Integration**: Fetches live sensor data from Firebase and calculates fire confidence using a trained machine learning model.
- **Confidence Calculation**: Displays fire confidence (from video) and sensor confidence (from sensor data) in the live feed.
- **Video Input Options**: Supports video files and live streams from ESP32-CAM.
- **Live Output Stream**: Provides a live video feed with detection results via a web interface.
- **Optimized Performance**: Processes every 2nd frame to improve FPS.
- **JSON API**: Exposes a `/status` endpoint to fetch fire detection and sensor confidence data.

## Requirements

The following dependencies are required to run the application:

- Python 3.9 or higher
- Flask
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLO
- Firebase Admin SDK
- Joblib
- Pandas
- Python Dotenv

Install the dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## File Structure

```
backend/
├── fire_detection_inference.py  # Main application script
├── requirements.txt             # Python dependencies
├── models/                      # Directory for YOLO model files and sensor model
│   ├── v8n.pt                   # YOLOv8n model
│   ├── v8n50epoch.pt            # YOLOv8n model trained for 50 epochs
│   ├── sensor_fire_model.pkl    # Trained machine learning model for sensor data
├── .env                         # Environment variables (e.g., Firebase key path)
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

3. **Add Sensor Model**:
   Place the trained sensor model file (`sensor_fire_model.pkl`) in the `models/` directory.

4. **Set Up Firebase**:
   - Download your Firebase Admin SDK key as a JSON file.
   - Place the file in the project directory and set its path in the `.env` file:
     ```
     FIREBASE_KEY_PATH=firebase_key.json
     ```

5. **Run the Application**:
   Start the Flask server:
   ```bash
   python fire_detection_inference.py
   ```

6. **Access the Web Interface**:
   Open your browser and navigate to:
   ```
   http://<your-ip>:5000/
   ```

7. **Specify Input Source**:
   - Enter the path to a video file or the URL of an ESP32-CAM stream.
   - Click "Start Detection" to begin processing.

8. **View the Output Stream**:
   - Click the "View Output Stream" link on the web interface.
   - The processed video feed with fire detection, smoke detection, and confidence percentages will be displayed.

9. **Fetch JSON Status**:
   - Access the `/status` endpoint to get fire confidence, sensor confidence, and other details:
     ```
     http://<your-ip>:5000/status
     ```

10. **Stop Detection**:
    - Navigate to `/stop` to stop the detection process.

## Notes

- The application uses the `v8n50epoch.pt` model by default. You can update the model path in `fire_detection_inference.py` if needed.
- Ensure the Firebase Realtime Database structure matches the expected format for sensor data.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLOv8 model.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
- [OpenCV](https://opencv.org/) for video processing.
- [Firebase](https://firebase.google.com/) for sensor data integration.
- [Scikit-learn](https://scikit-learn.org/) for the sensor confidence model.
- [Environmental Sensor Data](https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k) for the sensor telemetry dataset.