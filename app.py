# app.py
import matplotlib

matplotlib.use('Agg')  # 设置非交互式后端，必须在导入pyplot之前
from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import uuid
import shutil
import json
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['SESSIONS_FOLDER'] = 'sessions'  # 新增会话存储文件夹
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# 确保所有必要的目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSIONS_FOLDER'], exist_ok=True)


# 会话管理函数
def save_session(session_id, data):
    """保存会话数据到文件"""
    # 对于包含numpy数组和pandas时间戳的数据，使用pickle保存
    session_file = os.path.join(app.config['SESSIONS_FOLDER'], f"{session_id}.pkl")
    with open(session_file, 'wb') as f:
        pickle.dump(data, f)


def load_session(session_id):
    """从文件加载会话数据"""
    session_file = os.path.join(app.config['SESSIONS_FOLDER'], f"{session_id}.pkl")
    if not os.path.exists(session_file):
        return None

    with open(session_file, 'rb') as f:
        return pickle.load(f)


def delete_session(session_id):
    """删除会话文件"""
    session_file = os.path.join(app.config['SESSIONS_FOLDER'], f"{session_id}.pkl")
    if os.path.exists(session_file):
        os.remove(session_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # 创建会话ID
        session_id = str(uuid.uuid4())

        # 保存文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{file.filename}")
        file.save(file_path)

        # 创建并保存会话信息
        session_data = {
            'image_path': file_path,
            'filename': file.filename
        }
        save_session(session_id, session_data)

        # 读取图像并转换为base64以在前端显示
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400

        height, width = img.shape[:2]

        # 将图像转换为base64
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'session_id': session_id,
            'image': img_base64,
            'width': width,
            'height': height
        })


@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    session_id = data.get('session_id')

    # 加载会话数据
    session_data = load_session(session_id)
    if session_data is None:
        return jsonify({'error': 'Invalid session'}), 400

    try:
        # 获取参数
        p0 = (data.get('p0_x'), data.get('p0_y'))
        p1 = (data.get('p1_x'), data.get('p1_y'))
        p2 = (data.get('p2_x'), data.get('p2_y'))
        y_min = data.get('y_min')
        y_max = data.get('y_max')
        t_start = pd.Timestamp(data.get('t_start'))
        t_end = pd.Timestamp(data.get('t_end'))

        # 更新会话数据
        session_data.update({
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'y_min': y_min,
            'y_max': y_max,
            't_start': t_start,
            't_end': t_end
        })

        # 处理图像
        image_path = session_data['image_path']
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return jsonify({'error': 'Failed to read image'}), 400

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # 绿色曲线提取
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 只保留绘图区
        mask[:p1[1], :] = 0  # 上边栏
        mask[p0[1]:, :] = 0  # 下边栏
        mask[:, :p0[0]] = 0  # 左边栏
        mask[:, p2[0]:] = 0  # 右边栏

        # 形态学清噪
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 像素坐标 → 数据坐标
        xs, ys = [], []
        for x in range(p0[0], p2[0]):
            y_idx = np.where(mask[:, x] > 0)[0]
            if len(y_idx) == 0:
                continue
            y_pix = int(np.mean(y_idx))

            # Y 轴映射
            frac_y = (y_pix - p1[1]) / (p0[1] - p1[1])
            y_val = y_max - frac_y * (y_max - y_min)

            # X 轴映射
            frac_x = (x - p0[0]) / (p2[0] - p0[0])
            t_val = t_start + frac_x * (t_end - t_start)

            xs.append(t_val)
            ys.append(y_val)

        # 更新结果数据
        session_data['xs'] = xs
        session_data['ys'] = ys

        # 保存更新后的会话数据
        save_session(session_id, session_data)

        # 生成预览图
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.plot(xs, ys, 'g-')
        ax.set_title('Curve Preview', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Power (KW)', fontsize=12)
        ax.grid(True)
        plt.tight_layout()

        # 将图表转换为base64
        canvas = FigureCanvas(fig)
        img_io = io.BytesIO()
        canvas.print_png(img_io)
        img_io.seek(0)
        plot_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({
            'status': 'success',
            'points': len(xs),
            'preview': plot_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download', methods=['GET'])
def download_excel():
    session_id = request.args.get('session_id')

    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400

    # 加载会话数据
    session_data = load_session(session_id)
    if session_data is None:
        return jsonify({'error': f'Invalid session: {session_id}'}), 400

    try:
        # 检查必要的数据是否存在
        if 'xs' not in session_data or 'ys' not in session_data:
            return jsonify({'error': 'Session data is incomplete'}), 400

        xs = session_data['xs']
        ys = session_data['ys']

        # 创建Excel文件
        df_raw = pd.DataFrame({'time': xs, 'power': ys})
        excel_df = pd.DataFrame()

        # 处理时间列，拆分为日期和时间
        def split_datetime(datetime_val):
            dt = pd.to_datetime(datetime_val)
            return pd.Series([dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M')])

        # 应用拆分函数
        excel_df[['日期', '时间']] = df_raw['time'].apply(split_datetime)

        # 处理电力数据，保留三位小数
        excel_df['电力(KW)'] = df_raw['power'].round(3)

        # 保存Excel文件
        output_file = os.path.join(app.config['RESULTS_FOLDER'], f'拟合数据_{session_id}.xlsx')
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            excel_df.to_excel(writer, index=False, sheet_name='电力数据')

        # 确保文件存在
        if not os.path.exists(output_file):
            return jsonify({'error': 'Failed to create Excel file'}), 500

        return send_file(output_file, as_attachment=True, download_name='拟合数据.xlsx')

    except Exception as e:
        print(f"Error during download: {str(e)}")  # 服务器端日志记录
        return jsonify({'error': str(e)}), 500


@app.route('/cleanup', methods=['POST'])
def cleanup():
    session_id = request.json.get('session_id')

    # 加载会话数据
    session_data = load_session(session_id)
    if session_data:
        # 删除会话文件
        try:
            if 'image_path' in session_data and os.path.exists(session_data['image_path']):
                os.remove(session_data['image_path'])

            excel_path = os.path.join(app.config['RESULTS_FOLDER'], f'拟合数据_{session_id}.xlsx')
            if os.path.exists(excel_path):
                os.remove(excel_path)

            # 删除会话数据文件
            delete_session(session_id)
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    return jsonify({'status': 'success'})


@app.route('/debug/sessions', methods=['GET'])
def debug_sessions():
    # 获取所有会话文件
    session_files = [f for f in os.listdir(app.config['SESSIONS_FOLDER']) if f.endswith('.pkl')]
    sessions_info = {}

    for session_file in session_files:
        session_id = session_file[:-4]  # 移除 .pkl 后缀
        try:
            session_data = load_session(session_id)
            sessions_info[session_id] = {
                'has_data': 'xs' in session_data and 'ys' in session_data,
                'filename': session_data.get('filename', 'unknown')
            }
        except:
            sessions_info[session_id] = {'error': 'Failed to load session data'}

    return jsonify({
        'session_count': len(session_files),
        'sessions': sessions_info
    })


# 定时清理过期会话的函数
def cleanup_expired_sessions():
    # 遍历会话目录
    now = datetime.now()
    session_files = [f for f in os.listdir(app.config['SESSIONS_FOLDER']) if f.endswith('.pkl')]

    for session_file in session_files:
        file_path = os.path.join(app.config['SESSIONS_FOLDER'], session_file)
        # 检查文件修改时间
        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        # 24小时过期
        if (now - mtime).total_seconds() > 86400:
            session_id = session_file[:-4]
            try:
                session_data = load_session(session_id)
                if session_data and 'image_path' in session_data:
                    if os.path.exists(session_data['image_path']):
                        os.remove(session_data['image_path'])

                excel_path = os.path.join(app.config['RESULTS_FOLDER'], f'拟合数据_{session_id}.xlsx')
                if os.path.exists(excel_path):
                    os.remove(excel_path)

                # 删除会话数据文件
                os.remove(file_path)
            except:
                pass


if __name__ == '__main__':
    app.run(debug=True)