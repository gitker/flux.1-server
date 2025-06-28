from flask import Flask, request, render_template_string, send_file, jsonify, redirect, url_for
import torch
from diffusers import FluxKontextPipeline, FluxPipeline
from diffusers.utils import load_image
from PIL import Image
import io
import os
import uuid
from werkzeug.utils import secure_filename
import random

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# 创建必要的文件夹
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 初始化模型
print("正在加载FLUX模型...")

# 图像编辑模型
print("加载图像编辑模型...")
edit_pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
edit_pipe.to("mps")

# 文生图模型
print("加载文生图模型...")
text_to_image_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
text_to_image_pipe.to("mps")  

print("所有模型加载完成")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# HTML模板作为字符串
INDEX_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FLUX AI 图片工具</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .mode-selector {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 8px;
        }
        .mode-btn {
            flex: 1;
            padding: 12px 24px;
            background-color: transparent;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            max-width: 200px;
        }
        .mode-btn.active {
            background-color: #007bff;
            color: white;
        }
        .mode-btn:hover:not(.active) {
            background-color: #e9ecef;
        }
        .mode-content {
            display: none;
        }
        .mode-content.active {
            display: block;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input[type="file"], input[type="text"], input[type="number"], textarea {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        textarea {
            resize: vertical;
            min-height: 80px;
        }
        .input-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 18px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover:not(:disabled) {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .preview {
            margin-top: 20px;
            text-align: center;
        }
        .preview img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .loading {
            margin-top: 20px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        .progress-container {
            margin: 20px 0;
        }
        .progress-bar {
            width: 100%;
            height: 25px;
            background-color: #e9ecef;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #007bff, #0056b3);
            width: 0%;
            transition: width 0.3s ease;
            border-radius: 12px;
            position: relative;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        .progress-text {
            text-align: center;
            margin-top: 10px;
            font-weight: bold;
            color: #495057;
        }
        .loading-title {
            text-align: center;
            font-size: 18px;
            color: #007bff;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .loading-subtitle {
            text-align: center;
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 20px;
        }
        .processing-steps {
            margin-top: 15px;
            font-size: 14px;
            color: #6c757d;
        }
        .step {
            margin: 5px 0;
            padding-left: 20px;
            position: relative;
        }
        .step.active {
            color: #007bff;
            font-weight: bold;
        }
        .step.completed {
            color: #28a745;
        }
        .step::before {
            content: '○';
            position: absolute;
            left: 0;
            top: 0;
        }
        .step.active::before {
            content: '●';
            color: #007bff;
        }
        .step.completed::before {
            content: '✓';
            color: #28a745;
        }
        .error {
            color: #dc3545;
            text-align: center;
            margin-top: 10px;
        }
        .small-text {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .edit-mode {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .edit-mode h3 {
            margin-top: 0;
            color: #856404;
        }
        .feature-description {
            background-color: #e7f3ff;
            border: 1px solid #bee5eb;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .feature-description h4 {
            margin-top: 0;
            color: #0c5460;
        }
        @media (max-width: 768px) {
            .input-row {
                grid-template-columns: 1fr;
            }
            .mode-selector {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 FLUX AI 图片工具</h1>
        
        <div class="mode-selector">
            <button class="mode-btn active" data-mode="text-to-image">🖼️ 文字生成图片</button>
            <button class="mode-btn" data-mode="image-edit">✏️ 图片编辑</button>
        </div>

        <!-- 文生图模式 -->
        <div id="text-to-image" class="mode-content active">
            <div class="feature-description">
                <h4>🎯 文字生成图片</h4>
                <p>使用AI根据文字描述生成全新的图片。支持详细的提示词描述，生成高质量的1024x1024图片。</p>
            </div>
            
            <form id="textToImageForm" enctype="multipart/form-data">
                <input type="hidden" name="mode" value="text-to-image">
                
                <div class="form-group">
                    <label for="text_prompt">图片描述 (提示词)：</label>
                    <textarea id="text_prompt" name="prompt" placeholder="详细描述你想要生成的图片，例如：A beautiful sunset over a mountain lake, with pink and orange clouds, photorealistic, high quality" required>A cat holding a sign that says hello world</textarea>
                    <div class="small-text">详细描述图片内容、风格、光线、色彩等，提示词越详细效果越好</div>
                </div>
                
                <div class="input-row">
                    <div class="form-group">
                        <label for="text_guidance_scale">引导强度 (1.0-10.0)：</label>
                        <input type="number" id="text_guidance_scale" name="guidance_scale" value="3.5" min="1.0" max="10.0" step="0.1">
                        <div class="small-text">数值越高，越严格按照提示词生成</div>
                    </div>
                    
                    <div class="form-group">
                        <label for="num_inference_steps">推理步数 (20-100)：</label>
                        <input type="number" id="num_inference_steps" name="num_inference_steps" value="50" min="20" max="100" step="1">
                        <div class="small-text">步数越多质量越高，但耗时更长</div>
                    </div>
                </div>
                
                <button type="submit" id="textToImageBtn">🚀 生成图片</button>
            </form>
        </div>

        <!-- 图片编辑模式 -->
        <div id="image-edit" class="mode-content">
            {% if edit_mode %}
            <div class="edit-mode">
                <h3>🔄 继续编辑模式</h3>
                <p>正在编辑图片：<strong>{{ original_filename }}</strong></p>
                <p>上次提示词：<em>{{ last_prompt }}</em></p>
            </div>
            {% else %}
            <div class="feature-description">
                <h4>✏️ 图片编辑</h4>
                <p>上传一张图片，使用AI根据提示词对图片进行智能编辑和修改。</p>
            </div>
            {% endif %}
            
            <form id="imageEditForm" enctype="multipart/form-data">
                <input type="hidden" name="mode" value="image-edit">
                
                {% if not edit_mode %}
                <div class="form-group">
                    <label for="file">选择图片：</label>
                    <input type="file" id="file" name="file" accept="image/*" required>
                    <div class="small-text">支持格式：PNG, JPG, JPEG, GIF, BMP, WebP (最大16MB)</div>
                </div>
                {% else %}
                <input type="hidden" name="original_image" value="{{ original_image }}">
                <div class="form-group">
                    <label>当前编辑图片：</label>
                    <div class="preview">
                        <img src="/upload/{{ original_image }}" alt="原图" style="max-height: 200px;">
                    </div>
                </div>
                {% endif %}
                
                <div class="form-group">
                    <label for="edit_prompt">编辑提示词：</label>
                    <input type="text" id="edit_prompt" name="prompt" value="{{ last_prompt or 'Add a hat to the cat' }}" placeholder="描述你想要的图片效果...">
                    <div class="small-text">例如：Add sunglasses, Change background to beach, Make it cartoon style</div>
                </div>
                
                <div class="form-group">
                    <label for="edit_guidance_scale">引导强度 (1.0-5.0)：</label>
                    <input type="number" id="edit_guidance_scale" name="guidance_scale" value="2.5" min="1.0" max="5.0" step="0.1">
                    <div class="small-text">数值越高，AI越严格按照提示词处理图片</div>
                </div>
                
                <button type="submit" id="imageEditBtn">🚀 {{ '重新处理' if edit_mode else '开始处理' }}</button>
            </form>
            
            {% if not edit_mode %}
            <div id="preview" class="preview" style="display:none;">
                <h3>预览图片：</h3>
                <img id="previewImg" src="" alt="预览">
            </div>
            {% endif %}
        </div>
        
        <div id="loading" class="loading" style="display:none;">
            <div class="loading-title">⚡ 正在处理您的请求</div>
            <div class="loading-subtitle">这可能需要几分钟时间，请耐心等待...</div>
            
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="progress-text" id="progressText">0%</div>
            </div>
            
            <div class="processing-steps">
                <div class="step" id="step1">📤 准备工作完成</div>
                <div class="step" id="step2">🔍 分析输入内容</div>
                <div class="step" id="step3">🧠 AI模型推理中</div>
                <div class="step" id="step4">🎨 生成图片</div>
                <div class="step" id="step5">💾 保存结果</div>
            </div>
        </div>
        
        <div id="error" class="error" style="display:none;"></div>
    </div>

    <script>
        // 模式切换
        const modeBtns = document.querySelectorAll('.mode-btn');
        const modeContents = document.querySelectorAll('.mode-content');
        
        modeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const mode = btn.dataset.mode;
                
                // 更新按钮状态
                modeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // 更新内容显示
                modeContents.forEach(content => {
                    content.classList.remove('active');
                });
                document.getElementById(mode).classList.add('active');
            });
        });

        // 通用元素
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        // 图片编辑相关元素
        const fileInput = document.getElementById('file');
        const preview = document.getElementById('preview');
        const previewImg = document.getElementById('previewImg');

        let progressInterval;
        const steps = ['step1', 'step2', 'step3', 'step4', 'step5'];

        // 文件选择预览（仅在图片编辑模式且非编辑模式下）
        if (fileInput) {
            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        previewImg.src = e.target.result;
                        preview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                } else {
                    preview.style.display = 'none';
                }
            });
        }

        // 更新进度条
        function updateProgress(percent, stepIndex) {
            progressFill.style.width = percent + '%';
            progressText.textContent = Math.round(percent) + '%';
            
            // 更新步骤状态
            steps.forEach((stepId, index) => {
                const stepElement = document.getElementById(stepId);
                if (stepElement) {
                    if (index < stepIndex) {
                        stepElement.className = 'step completed';
                    } else if (index === stepIndex) {
                        stepElement.className = 'step active';
                    } else {
                        stepElement.className = 'step';
                    }
                }
            });
        }

        // 模拟进度更新
        function simulateProgress() {
            let progress = 0;
            let stepIndex = 0;
            
            progressInterval = setInterval(() => {
                if (stepIndex === 0 && progress < 10) {
                    progress += 1;
                } else if (stepIndex === 1 && progress < 25) {
                    progress += 0.5;
                } else if (stepIndex === 2 && progress < 80) {
                    progress += 0.3;
                } else if (stepIndex === 3 && progress < 95) {
                    progress += 1;
                } else if (stepIndex === 4 && progress < 100) {
                    progress += 2;
                } else if (stepIndex < steps.length - 1) {
                    stepIndex++;
                }
                
                updateProgress(progress, stepIndex);
                
                if (progress >= 100) {
                    clearInterval(progressInterval);
                    updateProgress(100, steps.length);
                }
            }, 100);
        }

        // 重置进度
        function resetProgress() {
            if (progressInterval) {
                clearInterval(progressInterval);
            }
            updateProgress(0, -1);
        }

        // 处理表单提交
        function handleFormSubmit(form, submitBtn, originalBtnText) {
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                
                const formData = new FormData(form);
                const mode = formData.get('mode');
                
                // 验证
                if (mode === 'image-edit') {
                    const file = fileInput ? fileInput.files[0] : null;
                    const originalImage = document.querySelector('input[name="original_image"]');
                    
                    if (!file && !originalImage) {
                        showError('请选择一张图片');
                        return;
                    }
                } else if (mode === 'text-to-image') {
                    const prompt = formData.get('prompt');
                    if (!prompt || prompt.trim() === '') {
                        showError('请输入图片描述');
                        return;
                    }
                }
                
                // 显示加载状态
                loading.style.display = 'block';
                error.style.display = 'none';
                submitBtn.disabled = true;
                submitBtn.textContent = '处理中...';
                
                // 开始模拟进度
                resetProgress();
                simulateProgress();
                
                fetch('/process', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // 确保进度条到达100%
                    clearInterval(progressInterval);
                    updateProgress(100, steps.length);
                    
                    setTimeout(() => {
                        loading.style.display = 'none';
                        submitBtn.disabled = false;
                        submitBtn.textContent = originalBtnText;
                        
                        if (data.success) {
                            // 构建URL参数
                            let url = `/result/${data.output_image}?mode=${data.mode}&prompt=${encodeURIComponent(data.prompt)}`;
                            if (data.original_image) {
                                url += `&original=${data.original_image}`;
                            }
                            window.location.href = url;
                        } else {
                            showError(data.error || '处理失败');
                            resetProgress();
                        }
                    }, 500);
                })
                .catch(err => {
                    clearInterval(progressInterval);
                    loading.style.display = 'none';
                    submitBtn.disabled = false;
                    submitBtn.textContent = originalBtnText;
                    showError('网络错误：' + err.message);
                    resetProgress();
                });
            });
        }

        // 绑定表单事件
        const textToImageForm = document.getElementById('textToImageForm');
        const textToImageBtn = document.getElementById('textToImageBtn');
        handleFormSubmit(textToImageForm, textToImageBtn, '🚀 生成图片');

        const imageEditForm = document.getElementById('imageEditForm');
        const imageEditBtn = document.getElementById('imageEditBtn');
        const originalImageInput = document.querySelector('input[name="original_image"]');
        const originalBtnText = originalImageInput ? '🚀 重新处理' : '🚀 开始处理';
        handleFormSubmit(imageEditForm, imageEditBtn, originalBtnText);

        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }
    </script>
</body>
</html>
'''

RESULT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>处理结果 - FLUX AI 图片工具</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .result-content {
            margin: 30px 0;
        }
        .text-to-image-result {
            text-align: center;
        }
        .comparison-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        .image-section {
            text-align: center;
        }
        .image-section h3 {
            margin-bottom: 15px;
            color: #555;
            font-size: 18px;
        }
        .image-section img, .text-to-image-result img {
            max-width: 100%;
            max-height: 500px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .image-section img:hover, .text-to-image-result img:hover {
            transform: scale(1.02);
        }
        .prompt-info {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #007bff;
        }
        .prompt-info h4 {
            margin: 0 0 10px 0;
            color: #007bff;
        }
        .prompt-text {
            font-style: italic;
            color: #555;
        }
        .mode-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .mode-badge.text-to-image {
            background-color: #d4edda;
            color: #155724;
        }
        .mode-badge.image-edit {
            background-color: #cce5ff;
            color: #004085;
        }
        .actions {
            text-align: center;
            margin-top: 30px;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
        }
        .btn {
            display: inline-block;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            border: none;
        }
        .btn-primary {
            background-color: #007bff;
            color: white;
        }
        .btn-primary:hover {
            background-color: #0056b3;
        }
        .btn-success {
            background-color: #28a745;
            color: white;
        }
        .btn-success:hover {
            background-color: #1e7e34;
        }
        .btn-warning {
            background-color: #ffc107;
            color: #212529;
        }
        .btn-warning:hover {
            background-color: #e0a800;
        }
        .btn-secondary {
            background-color: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background-color: #545b62;
        }
        .btn-info {
            background-color: #17a2b8;
            color: white;
        }
        .btn-info:hover {
            background-color: #138496;
        }
        .success-message {
            text-align: center;
            color: #28a745;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .completion-animation {
            text-align: center;
            margin-bottom: 20px;
        }
        .checkmark {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: #28a745;
            margin: 0 auto 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: checkmark-scale 0.6s ease-in-out;
        }
        .checkmark::before {
            content: '✓';
            color: white;
            font-size: 30px;
            font-weight: bold;
        }
        @keyframes checkmark-scale {
            0% { transform: scale(0); }
            50% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        @media (max-width: 768px) {
            .comparison-container {
                grid-template-columns: 1fr;
                gap: 20px;
            }
            .actions {
                flex-direction: column;
                align-items: center;
            }
            .btn {
                width: 200px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✨ {{ '图片生成完成' if mode == 'text-to-image' else '图片处理完成' }}！</h1>
        
        <div class="completion-animation">
            <div class="checkmark"></div>
        </div>
        
        <div class="mode-badge {{ mode }}">
            {{ '🖼️ 文字生成图片' if mode == 'text-to-image' else '✏️ 图片编辑' }}
        </div>
        
        <div class="success-message">
            🎉 {{ '您的图片已经成功生成' if mode == 'text-to-image' else '您的图片已经成功处理完成' }}
        </div>

        {% if prompt %}
        <div class="prompt-info">
            <h4>🎯 使用的提示词</h4>
            <div class="prompt-text">"{{ prompt }}"</div>
        </div>
        {% endif %}
        
        <div class="result-content">
            {% if mode == 'text-to-image' %}
            <!-- 文生图结果 -->
            <div class="text-to-image-result">
                <h3>🎨 生成的图片</h3>
                <img src="/output/{{ filename }}" alt="生成的图片" id="resultImg">
            </div>
            {% else %}
            <!-- 图片编辑结果 -->
            <div class="comparison-container">
                <div class="image-section">
                    <h3>📷 原始图片</h3>
                    <img src="/upload/{{ original_filename }}" alt="原始图片" id="originalImg">
                </div>
                <div class="image-section">
                    <h3>🎨 处理后图片</h3>
                    <img src="/output/{{ filename }}" alt="处理后的图片" id="resultImg">
                </div>
            </div>
            {% endif %}
        </div>
        
        <div class="actions">
            <a href="/output/{{ filename }}" download class="btn btn-success">💾 下载{{ '生成的' if mode == 'text-to-image' else '处理后' }}图片</a>
            
            {% if mode == 'image-edit' and original_filename %}
            <a href="/upload/{{ original_filename }}" download class="btn btn-secondary">📥 下载原始图片</a>
            <a href="/edit/{{ original_filename }}?prompt={{ prompt | urlencode }}" class="btn btn-warning">🔄 继续编辑原图</a>
            {% endif %}
            
            {% if mode == 'text-to-image' %}
            <a href="/?mode=text-to-image&prompt={{ prompt | urlencode }}" class="btn btn-info">🔄 重新生成</a>
            {% endif %}
            
            <a href="/" class="btn btn-primary">🆕 {{ '生成新图片' if mode == 'text-to-image' else '处理新图片' }}</a>
        </div>
    </div>

    <script>
        // 确保图片加载完成后显示
        const resultImg = document.getElementById('resultImg');
        const originalImg = document.getElementById('originalImg');
        
        resultImg.onload = function() {
            console.log('结果图片加载成功');
        };
        resultImg.onerror = function() {
            console.error('结果图片加载失败');
            alert('图片加载失败，请刷新页面重试');
        };
        
        if (originalImg) {
            originalImg.onload = function() {
                console.log('原图加载成功');
            };
            originalImg.onerror = function() {
                console.error('原图加载失败');
            };
        }
    </script>
</body>
</html>
'''

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_edit(input_image_path, prompt="Add a hat to the cat", guidance_scale=2.5):
    """图片编辑处理函数"""
    try:
        input_image = load_image(input_image_path)
        processed_image = edit_pipe(
            image=input_image,
            prompt=prompt,
            guidance_scale=guidance_scale
        ).images[0]
        return processed_image
    except Exception as e:
        print(f"图片编辑出错: {e}")
        return None

def generate_text_to_image(prompt, guidance_scale=3.5, num_inference_steps=50):
    """文生图处理函数"""
    try:
        image = text_to_image_pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(random.randint(1, 10000))
        ).images[0]
        return image
    except Exception as e:
        print(f"文生图出错: {e}")
        return None

@app.route('/')
def index():
    mode = request.args.get('mode', 'text-to-image')
    prompt = request.args.get('prompt', '')
    return render_template_string(INDEX_TEMPLATE, edit_mode=False, mode=mode, prompt=prompt)

@app.route('/edit/<original_filename>')
def edit_image(original_filename):
    last_prompt = request.args.get('prompt', 'Add a hat to the cat')
    return render_template_string(INDEX_TEMPLATE, 
                                edit_mode=True, 
                                original_image=original_filename,
                                original_filename=original_filename.replace('_', ' ').replace('.jpg', '').replace('.png', ''),
                                last_prompt=last_prompt)

@app.route('/process', methods=['POST'])
def process_request():
    mode = request.form.get('mode', 'text-to-image')
    prompt = request.form.get('prompt', '')
    guidance_scale = float(request.form.get('guidance_scale', 3.5))
    
    if mode == 'text-to-image':
        # 文生图处理
        num_inference_steps = int(request.form.get('num_inference_steps', 50))
        
        try:
            print(f"正在生成图片，提示词: {prompt}")
            generated_image = generate_text_to_image(prompt, guidance_scale, num_inference_steps)
            
            if generated_image is None:
                return jsonify({'error': '图片生成失败'}), 500
            
            # 保存生成的图片
            output_filename = f"generated_{uuid.uuid4()}.png"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            generated_image.save(output_path)
            
            return jsonify({
                'success': True,
                'output_image': output_filename,
                'mode': 'text-to-image',
                'prompt': prompt,
                'message': '图片生成完成'
            })
            
        except Exception as e:
            return jsonify({'error': f'生成图片时出错: {str(e)}'}), 500
    
    elif mode == 'image-edit':
        # 图片编辑处理
        original_image = request.form.get('original_image')
        
        # 如果是继续编辑模式
        if original_image:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_image)
            if not os.path.exists(filepath):
                return jsonify({'error': '原始图片不存在'}), 400
            
            try:
                print(f"正在重新处理图片: {filepath}")
                print(f"提示词: {prompt}")
                processed_image = process_image_edit(filepath, prompt, guidance_scale)
                
                if processed_image is None:
                    return jsonify({'error': '图片处理失败'}), 500
                
                # 保存处理后的图片
                base_name = os.path.splitext(original_image)[0]
                ext = os.path.splitext(original_image)[1]
                output_filename = f"processed_{uuid.uuid4()}_{base_name}{ext}"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                processed_image.save(output_path)
                
                return jsonify({
                    'success': True,
                    'output_image': output_filename,
                    'original_image': original_image,
                    'mode': 'image-edit',
                    'prompt': prompt,
                    'message': '图片处理完成'
                })
                
            except Exception as e:
                return jsonify({'error': f'处理图片时出错: {str(e)}'}), 500
        
        # 新上传模式
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if file and allowed_file(file.filename):
            # 保存上传的文件
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            try:
                print(f"正在处理图片: {filepath}")
                print(f"提示词: {prompt}")
                processed_image = process_image_edit(filepath, prompt, guidance_scale)
                
                if processed_image is None:
                    return jsonify({'error': '图片处理失败'}), 500
                
                # 保存处理后的图片
                output_filename = f"processed_{unique_filename}"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                processed_image.save(output_path)
                
                return jsonify({
                    'success': True,
                    'output_image': output_filename,
                    'original_image': unique_filename,
                    'mode': 'image-edit',
                    'prompt': prompt,
                    'message': '图片处理完成'
                })
                
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': f'处理图片时出错: {str(e)}'}), 500
        
        return jsonify({'error': '不支持的文件格式'}), 400
    
    return jsonify({'error': '无效的处理模式'}), 400

@app.route('/result/<filename>')
def show_result(filename):
    mode = request.args.get('mode', 'image-edit')
    original_filename = request.args.get('original')
    prompt = request.args.get('prompt', '')
    return render_template_string(RESULT_TEMPLATE, 
                                filename=filename, 
                                original_filename=original_filename,
                                mode=mode,
                                prompt=prompt)

@app.route('/output/<filename>')
def output_file(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

@app.route('/upload/<filename>')
def upload_file_serve(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5120)