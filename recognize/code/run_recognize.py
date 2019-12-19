# !/usr/bin/python
# -*- coding: UTF-8 -*-
import os
from PIL import Image
import recognize.code.models as model


def recoLineImage(image_path):
    img_src = Image.open(image_path)
    # 调整比例
    # img_w = img_src.size[0]
    # img_h = img_src.size[1]
    # img_w1 = img_w*1.2
    # img1 = img_src.resize((round(img_w1), img_h))

    #增强对比度
    # contrast = 2.5
    # img_contrast = ImageEnhance.Contrast(img_src).enhance(contrast)
    # img_gray = img_contrast.convert("L")
    # img_contrast.show()
    # img_gray.show()

    #锐化
    # enh_sharped = 1.5
    # img_sharped = ImageEnhance.Sharpness(img_contrast).enhance(enh_sharped)
    # img_gray = img_sharped.convert("L")
    # img_sharped.show()
    # img_gray.show()

    img = img_src.convert("L")
    predict_text = model.predict(img)

    return predict_text


def processFunction(line_image_dir):
    images_list = os.listdir(line_image_dir)
    print(images_list)
    results = {}
    for image in images_list:
        image_path = line_image_dir + image
        reco_text = recoLineImage(image_path)
        results[image.split('.')[0]] = reco_text
    return results


if __name__ == "__main__":
    line_image_dir = 'D:/liandongyoushi/Project/Coding/OCR/static/image_segment/02_original_detect/'
    results_list = processFunction(line_image_dir)
    print(results_list)
