from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
from monitor import CrowdMonitor

app = Flask(__name__)
monitor = CrowdMonitor()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    def get_dummy_frame():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Detected", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame
    is_dummy = True
else:
    is_dummy = False

def gen_frames():
    while True:
        if is_dummy:
            frame = get_dummy_frame()
            success = True
            cv2.circle(frame, (int(320 + 50 * np.sin(cv2.getTickCount()/10000000)), 300), 20, (0, 0, 255), -1)
            time.sleep(0.03)
        else:
            success, frame = cap.read()
            
        if not success:
            break
        else:
            processed_frame = monitor.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(monitor.get_stats())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
