from flask import Flask, render_template, Response, jsonify, request
import time, json, socket, os
import yaml
import logging

app = Flask(__name__)

shared_frame_path = '/dev/shm/debug_frame.jpg'

# 动态配置路径，初始值
CONFIG_PATH = '/home/nvidia/wust_vision/config/config_trt.yaml'

# 视频流生成器（MJPEG）
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
    url = f"http://{ip}:5000"
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

# 动态修改配置文件路径接口
@app.route('/config_path', methods=['GET', 'POST'])
def config_path_handler():
    global CONFIG_PATH
    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            new_path = data.get('config_path', '').strip()
            if not new_path:
                return jsonify({'error': 'config_path 不能为空'}), 400
            if not os.path.isfile(new_path):
                return jsonify({'error': f'{new_path} 文件不存在'}), 404
            CONFIG_PATH = new_path
            return jsonify({'message': f'配置路径已更新为: {CONFIG_PATH}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'config_path': CONFIG_PATH})

# 配置数据读取和保存，使用当前 CONFIG_PATH
@app.route('/config', methods=['GET', 'POST'])
def config_handler():
    global CONFIG_PATH
    if not os.path.exists(CONFIG_PATH):
        return jsonify({"error": f"{CONFIG_PATH} 不存在"}), 404

    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
            return jsonify({"message": "配置保存成功"})
        except Exception as e:
            return jsonify({"error": f"保存配置失败: {str(e)}"}), 500
    else:
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

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
    url = f"http://{ip}:5000"
    print(f"✅ Web 调试器已启动: {url}")
    app.run(host='0.0.0.0', port=5000, threaded=True)
