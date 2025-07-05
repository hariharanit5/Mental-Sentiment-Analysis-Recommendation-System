from flask import Flask, render_template, Response, jsonify, send_file
import cv2
from ultralytics import YOLO
import threading
import time
import os

app = Flask(__name__)

# Define the path to store images
IMAGE_PATH = "C:/Users/ADMIN/OneDrive/Desktop/mp/Yolov8/last_frame.jpg"

# Define the directory where songs are stored
SONG_DIR = "C:/Users/ADMIN/OneDrive/Desktop/mp/Yolov8/"

# Define the path for emotion-based songs
SONG_PATHS = {
    "sad": "sad.mp3",
    "angry": "angry.mp3",
}

# Load the trained YOLO model
model = YOLO('Yolov8/best.pt')

# Load OpenCV face detector
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = None
camera_running = False
lock = threading.Lock()
last_frame = None
last_emotion = ""

# Emotion-based quotes
emotion_quotes = {
    "angry": "Anger doesn't solve anything. It builds nothing, but it can destroy everything.<br> Take Breathing Excercise For 5 Minutes",
    "sad": "Tough times never last, but tough people do.<br> Take Breathing Excercise For 5 Minutes",
    "neutral": "\"Choose Happy, Be Happy.\" - This slogan emphasizes the idea that happiness is a choice you can actively make, encouraging you to actively pursue a joyful state of mind.",
    "happy": "Happiness is not something ready-made. It comes from your own actions."
}

def generate_frames():
    global cap, camera_running, last_frame, last_emotion
    
    while True:
        with lock:
            if not camera_running:
                break  # Stop the video feed loop

        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Reinitialize camera
        
        success, frame = cap.read()
        if not success:
            break
        
        last_frame = frame.copy()  # Store the last frame

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = frame[y:y + h, x:x + w]

            try:
                results = model(face_roi)
                if len(results) > 0:
                    result = results[0]
                    probs = result.probs if hasattr(result, 'probs') else None
                    if probs is not None:
                        class_id = int(probs.top1)
                        confidence = float(probs.top1conf)
                        class_name = model.names[class_id]
                        last_emotion = class_name  # Store last detected emotion
                        label = f'{class_name} {confidence:.2f}'
                        cv2.putText(frame, label, (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error processing face ROI: {e}")
                continue

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_camera():
    global cap, camera_running
    with lock:
        if cap is not None:
            cap.release()
            time.sleep(1)  # Small delay to ensure the camera properly resets
        cap = cv2.VideoCapture(0)  # Reinitialize camera
        camera_running = True
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop_camera():
    global cap, camera_running, last_frame
    with lock:
        camera_running = False
        if cap is not None:
            cap.release()
            cap = None  # Reset camera object
        cv2.destroyAllWindows()
        
        # Save the last frame to the specified path
        if last_frame is not None:
            cv2.imwrite(IMAGE_PATH, last_frame)
    return jsonify({'status': 'stopped'})

@app.route('/get_last_frame')
def get_last_frame():
    if os.path.exists(IMAGE_PATH):
        return send_file(IMAGE_PATH, mimetype   ='image/jpeg')
    else:
        return jsonify({'error': 'No saved image found.'}), 404

@app.route('/get_quote')
def get_quote():
    global last_emotion
    quote = emotion_quotes.get(last_emotion, "No emotion detected. Be yourself!")
    song_url = f"/play_song/{last_emotion}" if last_emotion in SONG_PATHS else None
    return jsonify({'emotion': last_emotion, 'quote': quote, 'song_url': song_url})

@app.route('/play_song/<emotion>')
def play_song(emotion):
    if emotion in SONG_PATHS:
        song_path = os.path.join(SONG_DIR, SONG_PATHS[emotion])
        if os.path.exists(song_path):
            return send_file(song_path, mimetype="audio/mpeg")
    return "Song not found", 404

if __name__ == '__main__':
    app.run(debug=True)