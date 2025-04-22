# app.py
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，必须在导入pyplot之前

# 设置中文字体支持
from matplotlib import font_manager as fm
import platform

# 根据不同操作系统设置合适的中文字体
system = platform.system()
if system == 'Windows':
    # Windows 常见中文字体选择
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
    for font_name in chinese_fonts:
        try:
            # 尝试设置字体
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            break
        except:
            continue
elif system == 'Darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False
else:  # Linux
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans', 'Droid Sans Fallback']
    matplotlib.rcParams['axes.unicode_minus'] = False

# 然后再导入其他模块
from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import io
import base64
import matplotlib.pyplot as plt  # 现在导入pyplot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import uuid
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# 确保上传和结果目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# 存储用户上传的图像和处理参数
sessions = {}


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

        # 存储会话信息
        sessions[session_id] = {
            'image_path': file_path,
            'filename': file.filename
        }

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

    if session_id not in sessions:
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

        # 存储参数
        sessions[session_id].update({
            'p0': p0,
            'p1': p1,
            'p2': p2,
            'y_min': y_min,
            'y_max': y_max,
            't_start': t_start,
            't_end': t_end
        })

        # 处理图像
        image_path = sessions[session_id]['image_path']
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

        # 存储结果数据
        sessions[session_id]['xs'] = xs
        sessions[session_id]['ys'] = ys

        # 生成预览图
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.plot(xs, ys, 'g-')
        ax.set_title('提取曲线预览', fontsize=14)  # 中文标题
        ax.set_xlabel('时间', fontsize=12)  # 中文标签
        ax.set_ylabel('功率 (KW)', fontsize=12)  # 中文标签
        ax.grid(True)
        plt.tight_layout()  # 调整布局以确保中文标题不被剪切

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

    if session_id not in sessions:
        return jsonify({'error': f'Invalid session: {session_id}'}), 400

    try:
        # 获取会话数据
        session_data = sessions[session_id]

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

    if session_id in sessions:
        # 删除会话文件
        try:
            os.remove(sessions[session_id]['image_path'])
            excel_path = os.path.join(app.config['RESULTS_FOLDER'], f'拟合数据_{session_id}.xlsx')
            if os.path.exists(excel_path):
                os.remove(excel_path)
        except:
            pass

        # 删除会话数据
        del sessions[session_id]

    return jsonify({'status': 'success'})


# 定时清理过期会话的函数（在实际部署中可以使用定时任务）
def cleanup_expired_sessions():
    # 简单示例，实际应用可能需要更复杂的过期逻辑
    for session_id in list(sessions.keys()):
        try:
            os.remove(sessions[session_id]['image_path'])
            excel_path = os.path.join(app.config['RESULTS_FOLDER'], f'拟合数据_{session_id}.xlsx')
            if os.path.exists(excel_path):
                os.remove(excel_path)
            del sessions[session_id]
        except:
            pass

@app.route('/debug/sessions', methods=['GET'])
def debug_sessions():
    # 仅用于调试，生产环境应移除
    return jsonify({
        'session_count': len(sessions),
        'sessions': {k: {'has_data': 'xs' in v and 'ys' in v} for k, v in sessions.items()}
    })

if __name__ == '__main__':
    app.run(debug=True)