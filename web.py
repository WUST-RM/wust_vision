from flask import Flask, render_template, Response, jsonify, request
import time, json, socket, os
import yaml
import logging

app = Flask(__name__)

# 配置路径和视频帧共享路径
shared_frame_path = '/dev/shm/debug_frame.jpg'
CONFIG_PATH = '/home/old-nuc/wust_vision/config/config_openvino.yaml'

# 视频流生成器（MJPEG）
def mjpeg_stream():
    while True:
        try:
            with open(shared_frame_path, 'rb') as f:
                jpg_bytes = f.read()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        except FileNotFoundError:
            time.sleep(0.01)  # 若帧文件未生成
        time.sleep(0.03)  # 控制帧率 ~30fps

# 首页：渲染带 server_url 的模板
@app.route('/')
def index():
    def get_local_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))  # broadcast dummy
            IP = s.getsockname()[0]
        except:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    ip = get_local_ip()
    url = f"http://{ip}:5000"
    return render_template('index.html', server_url=url)

# 视频流路由
@app.route('/video')
def video_feed():
    return Response(mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 实时波形数据：读取 JSON
@app.route('/data')
def get_data():
    try:
        with open('/dev/shm/cmd_log.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/aim_log')
def aim_log():
    try:
        with open('/dev/shm/aim_log.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/target_log')
def target_log():
    try:
        with open('/dev/shm/target_log.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 配置数据（YAML）：读取和保存
@app.route('/config', methods=['GET', 'POST'])
def config_handler():
    if not os.path.exists(CONFIG_PATH):
        return jsonify({"error": f"{CONFIG_PATH} 不存在"}), 404

    if request.method == 'POST':
        try:
            data = request.get_json(force=True)
            #app.logger.info("接收到配置数据: %s", json.dumps(data, ensure_ascii=False, indent=2))

            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)

            return jsonify({"message": "配置保存成功"})
        except Exception as e:
            #app.logger.error("保存配置失败: %s", e, exc_info=True)
            return jsonify({"error": f"保存配置失败: {str(e)}"}), 500
    else:
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# 启动服务
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

