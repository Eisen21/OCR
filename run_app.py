import os
from datetime import timedelta
from werkzeug.utils import secure_filename
from detect.code.run_detect import run_detect
from recognize.code.run_recognize import processFunction
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
import recognize.code.models

# 设置允许upload的文件格式
ALLOWED_EXTENSIONS = {'png', 'PNG', 'jpg', 'JPG', 'bmp'}


# 判断图片格式
def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# 实例化app对象
app = Flask(__name__)


# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# 设置默认端口下跳转到主页面
@app.route('/')
def index():
    return redirect(url_for('upload'))


# 在主页面下定义图片上传
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # 检查文件upload的类型
        if not (f and allow_file(f.filename)):
            return jsonify({"error": 404, "msg": "请检查上传的图片类型,仅限于png,jpg,bmp"})
        file_name = secure_filename(f.filename)
        # 设置保存upload文件所在路径
        base_path = os.path.dirname(__file__)
        upload_path = os.path.join(base_path, 'static/image_upload', file_name)
        f.save(upload_path)

        return render_template('upload_ok.html', img_name=file_name)
    return render_template('upload.html')


# 对上传图片进行检测
@app.route('/detect/<img_name>', methods=['POST', 'GET'])
def detect(img_name):
    if request.method == 'POST':
        base_path = os.path.dirname(__file__)
        img_path = os.path.join(base_path, 'static/image_upload', img_name)
        print(img_path)
        # 模型路径
        model_path = "D:/liandongyoushi/Project/Coding/OCR/detect/model/ResNet_00blue_augyr.pb"
        # 加载模型进行预测,并返回预测结果(1、分割的32图片;2、带box的图片)
        re_img_name = run_detect(img_path, model_path)
        print(re_img_name)
        return render_template('detect.html', img_name=re_img_name)
    return render_template('upload_ok.html', img_name=img_name)


# 对检测结果进行识别
@app.route('/recognize/<img_name>', methods=['POST', 'GET'])
def recognize(img_name):
    if request.method == 'POST':
        img_name_ = img_name.split('.')[0] + '/'
        base_path = os.path.dirname(__file__)
        img_path = os.path.join(base_path, 'static/image_segment', img_name_)
        # img_path = 'D:/liandongyoushi/Project/Coding/OCR/static/image_segment/02_original_detect/'
        re_recognize = processFunction(img_path)
        print(re_recognize)
    #     return render_template('recognize.html')
    # return render_template('detect.html', img_name=img_name)


if __name__ == '__main__':
    app.run()
