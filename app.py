import cv2
import numpy as np
import time
import detect_from_images
from flask import Flask, Response, render_template
from datetime import timedelta

app = Flask(__name__)

# Path to the video file
video_file = 'input.mp4'

def generate_frames():
    video = cv2.VideoCapture(video_file)
    started_time = time.time()
    frame_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break
        else:
            start_time = time.perf_counter()
            frame = detect_from_images.process(frame, 147, 72, 3, 7)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f'process-The function took {elapsed_time} seconds to complete.')


            # Add current time to the frame
            height, width, _ = frame.shape
            elapsed_time = time.time() - started_time
            working_time = str(timedelta(seconds=int(elapsed_time)))
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left = (10, height - 20)
            font_scale = 5
            font_color = (255, 255, 255)  # White color in BGR format
            line_type = 3
            frame = cv2.putText(frame, working_time, bottom_left, font, font_scale, font_color, line_type)

            # Encode the frame as JPEG and yield it for streaming
            start_time = time.perf_counter()
            ret, buffer = cv2.imencode('.jpg', frame)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f'imencode-The function took {elapsed_time} seconds to complete.')

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
