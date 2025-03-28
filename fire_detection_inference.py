from flask import Flask, Response, request, render_template_string
import cv2
import os
import platform
from ultralytics import YOLO
from threading import Thread

# Load the YOLOv8n model
model = YOLO('models/v8n50epoch.pt')  # Update with your model path

app = Flask(__name__)
output_frame = None  # Global variable to store the latest processed frame
video_thread = None  # Thread for video processing
stop_thread = False  # Flag to stop the video processing thread


def detect_fire(frame):
    """
    Detect fire in the given frame using the YOLOv8n model.
    """
    results = model.predict(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
    fire_detected = False

    for *box, conf, cls in detections:
        if int(cls) == 0:  # Assuming '0' is the class ID for fire
            fire_detected = True
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Fire: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame, fire_detected


def process_video(input_source):
    """
    Process video from input source and perform fire detection.
    """
    global output_frame, stop_thread

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    frame_skip = 2  # Process every 2nd frame to improve FPS
    frame_count = 0

    while not stop_thread:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip this frame

        frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
        processed_frame, _ = detect_fire(frame)
        output_frame = processed_frame

    cap.release()


def generate_feed():
    """
    Generator function to stream the processed frames.
    """
    global output_frame
    while True:
        if output_frame is not None:
            _, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """
    Render the main page with options to specify input source and mode.
    """
    return render_template_string('''
        <!doctype html>
        <title>Fire Detection</title>
        <h1>Fire Detection with YOLO</h1>
        <form action="/start" method="post">
            <label for="input_source">Input Source (Video file path or ESP32-CAM URL):</label><br>
            <input type="text" id="input_source" name="input_source" required><br><br>
            <label for="show_realtime">Show Realtime Detection:</label>
            <input type="checkbox" id="show_realtime" name="show_realtime"><br><br>
            <button type="submit">Start Detection</button>
        </form>
        <h2>View Output Stream:</h2>
        <a href="/video_feed" target="_blank">Click here to view the output stream</a>
    ''')


@app.route('/start', methods=['POST'])
def start_detection():
    """
    Start the fire detection process based on user input.
    """
    global video_thread, stop_thread

    input_source = request.form['input_source']

    if video_thread and video_thread.is_alive():
        stop_thread = True
        video_thread.join()

    stop_thread = False
    video_thread = Thread(target=process_video, args=(input_source,))
    video_thread.daemon = True
    video_thread.start()

    return "Fire detection started! <a href='/'>Go back</a>"


@app.route('/video_feed')
def video_feed():
    """
    Route to stream the video feed for the frontend dashboard.
    """
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop')
def stop_detection():
    """
    Stop the fire detection process.
    """
    global stop_thread, video_thread

    stop_thread = True
    if video_thread and video_thread.is_alive():
        video_thread.join()

    return "Fire detection stopped! <a href='/'>Go back</a>"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
