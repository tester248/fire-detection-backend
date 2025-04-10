from flask import Flask, Response, request, render_template_string, jsonify
import cv2
import os
from ultralytics import YOLO
from threading import Thread
import numpy as np
import time  # Add this import for measuring time
from dotenv import load_dotenv
import joblib
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from flask_cors import CORS  # Add this import
from twilio_alerts import check_thresholds_and_alert  # Import the alert utility

# Load environment variables from .env file
load_dotenv()

# Get the Firebase key path from the environment variable
firebase_key_path = os.getenv("FIREBASE_KEY_PATH")

# Initialize Firebase Admin SDK
if firebase_key_path:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://fire-detection-system-29797-default-rtdb.asia-southeast1.firebasedatabase.app/'  # Replace with your Firebase Realtime Database URL
    })
else:
    raise ValueError("FIREBASE_KEY_PATH environment variable not set.")
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

# Global variables
output_frame = None
video_thread = None
stop_thread = False
model = None
# Load the trained sensor model
sensor_model = joblib.load('models/sensor_fire_model.pkl')
current_model_path = 'models/best.pt'

# Global variables for JSON response
fire_confidence = 0.0
smoke_detected_status = False

def load_model(model_path):
    global model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

def detect_fire(frame):
    global current_model_path, fire_confidence, smoke_detected_status
    try:
        results = model.predict(frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        fire_detected = False
        smoke_detected_status = False
        total_confidence = 0
        fire_count = 0

        for *box, conf, cls in detections:
            if int(cls) == 0:  # Assuming '0' is the class ID for fire
                if 'best.pt' in current_model_path:  # Flip annotations for best.pt
                    smoke_detected_status = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Smoke: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    fire_detected = True
                    fire_count += 1
                    total_confidence += conf
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Fire: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif int(cls) == 1:  # Assuming '1' is the class ID for smoke
                if 'best.pt' in current_model_path:  # Flip annotations for best.pt
                    fire_detected = True
                    fire_count += 1
                    total_confidence += conf
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Fire: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    smoke_detected_status = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"Smoke: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Calculate average confidence for fire detection
        fire_confidence = (total_confidence / fire_count) * 100 if fire_count > 0 else 0.0

        return frame, fire_detected, smoke_detected_status
    except Exception as e:
        print(f"Error in detect_fire: {e}")
        fire_confidence = 0.0
        smoke_detected_status = False
        return frame, False, False

def process_video(input_source):
    global output_frame, stop_thread
    print(f"Attempting to open video source: {input_source}")
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {input_source}")
        return

    frame_skip = 2
    frame_count = 0
    prev_time = time.time()  # Initialize time for FPS calculation

    print("Video capture started")
    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from video source. Stopping thread.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        start_time = time.time()  # Start time for processing
        frame = cv2.resize(frame, (640, 360))
        processed_frame, fire_detected, smoke_detected = detect_fire(frame)
        end_time = time.time()  # End time for processing

        # Calculate real-time FPS
        fps = 1 / (end_time - prev_time)
        prev_time = end_time

        # Fetch live sensor data from Firebase
        sensor_data = fetch_sensor_data()
        sensor_confidence = 0.0
        if sensor_data:
            sensor_confidence = calculate_sensor_confidence(sensor_data)

        # Add FPS counter, fire confidence, and sensor confidence to the frame
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Fire Confidence: {fire_confidence:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_frame, f"Sensor Confidence: {sensor_confidence:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        output_frame = processed_frame

    cap.release()
    print("Video capture released")

def generate_feed():
    global output_frame
    # Fallback image if no frame is available
    fallback_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(fallback_frame, "No Video Feed Available", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    while True:
        frame_to_send = output_frame if output_frame is not None else fallback_frame
        try:
            _, buffer = cv2.imencode('.jpg', frame_to_send)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_feed: {e}")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', fallback_frame)[1].tobytes() + b'\r\n')

@app.route('/')
def index():
    model_files = [f for f in os.listdir('models') if f.endswith('.pt')]
    return render_template_string('''
        <!doctype html>
        <title>Fire Detection</title>
        <h1>Fire Detection with YOLO</h1>
        <form id="startForm" method="post" onsubmit="startDetection(event)">
            <label for="input_source">Input Source (Video file path or ESP32-CAM URL):</label><br>
            <input type="text" id="input_source" name="input_source" required><br><br>
            <label for="model_selector">Select Model:</label><br>
            <select id="model_selector" name="model_selector" onchange="changeModel(this.value)">
                {% for model in models %}
                <option value="{{ model }}" {% if model == current_model %}selected{% endif %}>{{ model }}</option>
                {% endfor %}
            </select><br><br>
            <button type="submit">Start Detection</button>
            <button type="button" id="stopButton" onclick="stopDetection()" style="background-color: #f44336; display: none;">Stop Detection</button>
        </form>
        <div id="streamContainer" style="display: none;">
            <h2>Live Output Stream:</h2>
            <img id="videoFeed" src="" style="width: 640px; height: 360px;" alt="Video Stream">
            <p>Direct Stream URL: <a href="/video_feed" target="_blank">/video_feed</a></p>
        </div>
        <script>
            function startDetection(event) {
                event.preventDefault();
                const form = document.getElementById('startForm');
                const formData = new FormData(form);
                
                fetch('/start', {
                    method: 'POST',
                    body: formData
                }).then(response => {
                    if (response.ok) {
                        const streamContainer = document.getElementById('streamContainer');
                        const videoFeed = document.getElementById('videoFeed');
                        const stopButton = document.getElementById('stopButton');
                        videoFeed.src = '/video_feed';
                        streamContainer.style.display = 'block';
                        stopButton.style.display = 'inline-block';
                    } else {
                        alert('Error starting detection');
                    }
                }).catch(error => console.error('Error:', error));
            }

            function stopDetection() {
                fetch('/stop')
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        // Reset the UI
                        document.getElementById('streamContainer').style.display = 'none';
                        document.getElementById('stopButton').style.display = 'none';
                        document.getElementById('videoFeed').src = '';
                    })
                    .catch(error => console.error('Error:', error));
            }

            function changeModel(modelPath) {
                fetch('/change_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_path: modelPath })
                }).then(response => response.json())
                  .then(data => alert(data.message))
                  .catch(error => console.error('Error:', error));
            }
        </script>
    ''', models=model_files, current_model=os.path.basename(current_model_path))

@app.route('/start', methods=['POST'])
def start_detection():
    global video_thread, stop_thread
    input_source = request.form['input_source']
    print(f"Starting detection with input source: {input_source}")

    if video_thread and video_thread.is_alive():
        stop_thread = True
        video_thread.join()

    stop_thread = False
    video_thread = Thread(target=process_video, args=(input_source,))
    video_thread.daemon = True
    video_thread.start()

    return '', 200  # Return a simple success response

@app.route('/video_feed')
def video_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_model', methods=['POST'])
def change_model():
    global current_model_path
    data = request.get_json()
    model_path = data.get('model_path')

    if model_path and os.path.exists(os.path.join('models', model_path)):
        current_model_path = os.path.join('models', model_path)
        load_model(current_model_path)
        return jsonify({"message": f"Model changed to {model_path}"})
    else:
        return jsonify({"message": "Invalid model path"}), 400

@app.route('/stop')
def stop_detection():
    global stop_thread, video_thread
    stop_thread = True
    if video_thread and video_thread.is_alive():
        video_thread.join()
    return jsonify({"message": "Fire detection stopped!"})

def fetch_sensor_data():
    """
    Fetch the latest sensor readings from Firebase Realtime Database.
    :return: Dictionary with sensor readings
    """
    try:
        # Replace 'sensor_data/latest' with the correct path in your Firebase database
        ref = db.reference('sensors')  # Adjust the path as per your Firebase structure
        sensor_data = ref.get()

        #print("Fetched sensor data from Firebase:")
        #print(sensor_data)
        # Ensure the data is in the expected format
        if sensor_data:
            return {
                "co": sensor_data.get("co", 0.0),
                "humidity": sensor_data.get("humidity", 0.0),
                "lpg": sensor_data.get("lpg", 0.0),
                "smoke": sensor_data.get("smoke", 0.0),
                "temp": sensor_data.get("temperature", 0.0)
            }
        else:
            print("No sensor data found in Firebase.")
            return None
    except Exception as e:
        print(f"Error fetching sensor data from Firebase: {e}")
        return None




def calculate_sensor_confidence(sensor_data):
    """
    Calculate fire confidence based on sensor readings using the trained model.
    :param sensor_data: Dictionary with keys 'co', 'humidity', 'lpg', 'smoke', 'temp'
    :return: Confidence percentage (0-100)
    """
    # Create a DataFrame with the same column names as the training data
    feature_names = ['co', 'humidity', 'lpg', 'smoke', 'temp']  # Ensure these match the training data
    features = pd.DataFrame([[
        sensor_data['co'],
        sensor_data['humidity'],
        sensor_data['lpg'],
        sensor_data['smoke'],
        sensor_data['temp']
    ]], columns=feature_names)

    # Predict fire confidence
    confidence = sensor_model.predict_proba(features)[0][1] * 100  # Probability of fire
    return confidence


@app.route('/status', methods=['GET'])
def get_status():
    global fire_confidence, smoke_detected_status

    try:
        # Fetch live sensor data from Firebase
        sensor_data = fetch_sensor_data()

        if sensor_data:
            # Calculate sensor confidence using the trained model
            sensor_confidence = calculate_sensor_confidence(sensor_data)

            # Adjusted confidence (weighted average)
            adjusted_confidence = (fire_confidence * 0.7) + (sensor_confidence * 0.3)
            
            # Check thresholds and send alerts if necessary
            location = request.args.get('location', 'Lab 607')
            check_thresholds_and_alert(
                fire_confidence=fire_confidence,
                sensor_confidence=sensor_confidence,
                adjusted_confidence=adjusted_confidence,
                location=location
            )

            return jsonify({
                "fire_confidence": float(fire_confidence),  # Convert to Python float
                "sensor_confidence": float(sensor_confidence),  # Convert to Python float
                "adjusted_confidence": float(adjusted_confidence),  # Convert to Python float
                "smoke_detected": smoke_detected_status,
                "output_frame_available": output_frame is not None,
                "sensor_readings": sensor_data  # Include all real-time sensor readings
            })
        else:
            return jsonify({
                "fire_confidence": float(fire_confidence),  # Convert to Python float
                "sensor_confidence": 0.0,
                "adjusted_confidence": float(fire_confidence * 0.7),  # Convert to Python float
                "smoke_detected": smoke_detected_status,
                "output_frame_available": output_frame is not None,
                "error": "No sensor data available."
            }), 200
    except Exception as e:
        print(f"Error in /status endpoint: {e}")
        return jsonify({
            "error": "An error occurred while fetching the status.",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    load_model(current_model_path)
    app.run(host='0.0.0.0', port=5000, debug=False)