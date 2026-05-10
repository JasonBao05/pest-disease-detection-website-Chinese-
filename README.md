# pest-disease-detection-website-Chinese-
A YOLO11-based intelligent crop pest detection and disease diagnosis system with Flask web deployment.

# 🌾 农作物病虫害智能监测系统

基于 **YOLO11 深度学习模型** 的农作物病虫害智能监测系统，实现：

- 🐛 **害虫检测（目标检测）**
- 🍃 **病害诊断（图像分类）**
- 🌐 **前后端一体化 Web 部署**
- 📊 **结果可视化展示**
- 📚 **病虫害知识库查询**

系统面向精准农业场景，可辅助农户快速识别病虫害种类并提供防治建议。

---

## 🚀 在线体验

**项目访问地址：**

👉 https://pestdetectbjc.vip.cpolar.cn/

---

## ✨ 功能特点

### 1. 害虫检测（YOLO11 Detection）

支持对农作物图像中的害虫进行目标检测：

- 自动定位虫体位置
- 输出检测框
- 显示数量统计
- 展示虫害介绍
- 给出防治建议

支持害虫类别：**38 类**

---

### 2. 病害诊断（YOLO11 Classification）

支持作物叶片病害智能分类：

- 自动识别病害类型
- 输出分类置信度
- Top3 候选结果展示
- 病害知识详情
- 防治建议

支持病害类别：**71 类**

---

### 3. 前后端交互式 Web 系统

支持：

- 图片上传
- 拖拽上传
- 实时检测
- 结果展示
- 响应式界面

---

## 🖥️ 系统界面

### 首页

支持选择：

- 害虫检测
- 病害诊断

上传图片即可自动分析。

---

### 病害诊断结果页

展示：

- 最终诊断结果
- 置信度
- Top3 候选结果
- 病害介绍
- 症状说明
- 防治建议

---

### 害虫检测结果页

展示：

- 检测框
- 害虫数量
- 害虫简介
- 危害说明
- 易感作物
- 防治建议

---

# 📁 项目结构

```bash
crop-pest-detection/
│
├── uploads/                  # 用户上传图片临时存储目录
├── results/                  # 检测结果保存目录
│
├── pest_best.pt              # 害虫检测模型权重（YOLO11 Detection）
├── disease_best.pt           # 病害分类模型权重（YOLO11 Classification）
│
├── pest_info.json            # 害虫知识库
├── disease_info.json         # 病害知识库
│
├── data_disease.yaml         # 病害类别配置文件
│
├── server.py                 # Flask 后端主程序
├── index.html                # 前端页面
│
└── README.md
