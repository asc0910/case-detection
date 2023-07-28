import cv2
import numpy as np
import time
import detect_from_images
from flask import Flask, Response, render_template
from datetime import timedelta
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Path to the video file
video_file = os.environ.get('CAMERA_VIDEO_INPUT')
fps = float(os.environ.get('CAMERA_FPS'))

def generate_frames():
    video = cv2.VideoCapture(video_file)
    while True:
        frame_start_time = time.perf_counter()
        success, frame = video.read()
        if not success:
            break
        else:            
            frame = detect_from_images.process(frame, 147, 72, 3, 3, 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        frame_end_time = time.perf_counter()
        frame_elapsed_time = frame_end_time - frame_start_time
        time.sleep(max(0, 1 / fps - frame_elapsed_time))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
