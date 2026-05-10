"""
农作物病虫害智能检测系统 - 后端
基于 Flask + YOLO11 检测（害虫）+ YOLO11 分类（病害）
"""
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import os
import json
import uuid
import cv2

# ========== 初始化 ==========
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ========== 加载两个模型 ==========
print('=' * 60)
print('🔄 正在加载害虫检测模型...')
pest_model = YOLO('best.pt')
print(f'✅ 害虫模型加载完成，共 {len(pest_model.names)} 类')

print('\n🔄 正在加载病害分类模型...')
disease_model = YOLO('disease_best.pt')
print(f'✅ 病害模型加载完成，共 {len(disease_model.names)} 类')

# ========== 加载知识库 ==========
print('\n🔄 正在加载知识库...')
with open('pest_info.json', 'r', encoding='utf-8') as f:
    PEST_INFO = json.load(f)
print(f'✅ 害虫知识库: {len(PEST_INFO)} 条')

with open('disease_info.json', 'r', encoding='utf-8') as f:
    DISEASE_INFO = json.load(f)
print(f'✅ 病害知识库: {len(DISEASE_INFO)} 条')
print('=' * 60)


# ========== 路由 ==========
@app.route('/')
def index():
    """返回前端页面"""
    return send_file('index.html')


@app.route('/results/<filename>')
def get_result_image(filename):
    """让前端能访问 results/ 里的带框结果图"""
    return send_from_directory(RESULTS_FOLDER, filename)


@app.route('/upload', methods=['POST'])
def upload():
    """核心接口：接收图片 → 检测害虫 + 分类病害 → 返回综合结果"""
    # ----- 1. 检查文件 -----
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': '没有收到图片'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': '文件名为空'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
        return jsonify({'success': False, 'error': '不支持的图片格式'}), 400

    # ----- 2. 保存上传的图片 -----
    unique_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(upload_path)

    # ========================================
    # 任务 1：害虫检测（YOLO 检测）
    # ========================================
    pest_result = run_pest_detection(upload_path, unique_name)

    # ========================================
    # 任务 2：病害分类（YOLO 分类）
    # ========================================
    disease_result = run_disease_classification(upload_path)

    # ----- 打印日志 -----
    print(f'\n📷 {unique_name}')
    print(f'   🐛 害虫: 检出 {pest_result["total_count"]} 只 / {pest_result["pest_types"]} 种')
    print(f'   🍃 病害: Top1 = {disease_result["top1"]["chinese_name"]} ({disease_result["top1"]["confidence"]}%)')

    # ----- 返回综合 JSON -----
    return jsonify({
        'success': True,
        'pest': pest_result,
        'disease': disease_result
    })


def run_pest_detection(image_path, unique_name):
    """跑害虫检测，返回结构化结果"""
    try:
        results = pest_model(image_path, conf=0.25)
        result = results[0]
    except Exception as e:
        return {'error': f'害虫检测失败: {str(e)}', 'total_count': 0, 'pest_types': 0,
                'detections': [], 'pest_summary': [], 'result_image': None}

    # 保存带框结果图
    plotted_img = result.plot()
    result_path = os.path.join(RESULTS_FOLDER, unique_name)
    cv2.imwrite(result_path, plotted_img)

    # 整理检测结果
    detections = []
    pest_counts = {}

    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = pest_model.names[cls_id]
        conf = float(box.conf[0])
        info = PEST_INFO.get(cls_name, {})

        detections.append({
            'class_name': cls_name,
            'chinese_name': info.get('chinese_name', cls_name),
            'confidence': round(conf * 100, 2)
        })
        pest_counts[cls_name] = pest_counts.get(cls_name, 0) + 1

    # 每种害虫的摘要信息
    pest_summary = []
    for cls_name, count in pest_counts.items():
        info = PEST_INFO.get(cls_name, {})
        pest_summary.append({
            'class_name': cls_name,
            'chinese_name': info.get('chinese_name', cls_name),
            'count': count,
            'intro': info.get('intro', '暂无简介'),
            'harm': info.get('harm', '暂无危害描述'),
            'crops_affected': info.get('crops_affected', []),
            'treatment': info.get('treatment', '暂无防治建议')
        })
    pest_summary.sort(key=lambda x: x['count'], reverse=True)

    return {
        'total_count': len(detections),
        'pest_types': len(pest_summary),
        'result_image': f'/results/{unique_name}',
        'detections': detections,
        'pest_summary': pest_summary
    }


def run_disease_classification(image_path):
    """跑病害分类，返回 Top3 结果"""
    try:
        results = disease_model(image_path)
        result = results[0]
    except Exception as e:
        return {'error': f'病害分类失败: {str(e)}', 'top1': None, 'top3': []}

    # 取 Top3
    probs = result.probs
    top3_idx = probs.top5[:3]  # 直接取前 3
    top3_conf = probs.top5conf[:3].tolist()

    top3_list = []
    for i in range(3):
        cls_id = int(top3_idx[i])
        cls_name = disease_model.names[cls_id]
        conf = float(top3_conf[i]) * 100
        info = DISEASE_INFO.get(cls_name, {})

        top3_list.append({
            'class_name': cls_name,
            'chinese_name': info.get('chinese_name', cls_name),
            'crop': info.get('crop', '未知'),
            'confidence': round(conf, 2),
            'cause': info.get('cause', '未知'),
            'intro': info.get('intro', '暂无简介'),
            'symptoms': info.get('symptoms', '暂无症状描述'),
            'crops_affected': info.get('crops_affected', []),
            'treatment': info.get('treatment', '暂无防治建议')
        })

    return {
        'top1': top3_list[0],   # 最可能的病害（前端主要展示）
        'top3': top3_list       # 完整 Top3（前端可选展示）
    }


# ========== 启动 ==========
if __name__ == '__main__':
    print('\n🚀 服务启动中...')
    print('📍 访问地址: http://localhost:5000')
    print('=' * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)