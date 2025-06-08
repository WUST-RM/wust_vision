from flask import Flask, render_template, Response
import time
from flask import jsonify
import json
import os

app = Flask(__name__)
shared_frame_path = '/dev/shm/debug_frame.jpg'

def mjpeg_stream():
    while True:
        try:
            with open(shared_frame_path, 'rb') as f:
                jpg_bytes = f.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        except FileNotFoundError:
            time.sleep(0.01)
        time.sleep(0.03)

@app.route('/')
def index():
    import socket
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    ip = get_local_ip()
    port = 5000
    url = f"http://{ip}:{port}"
    return render_template('index.html', server_url=url)


@app.route('/video')
def video_feed():
    return Response(mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/data')
def get_data():
    try:
        with open('/dev/shm/cmd_log.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

 
    import socket
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    ip = get_local_ip()
    port = 5000
    url = f"http://{ip}:{port}"
    print(f"Server running at: {url}")

    app.run(host='0.0.0.0', port=5000, threaded=True)
