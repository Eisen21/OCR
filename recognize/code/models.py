# -*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
import recognize.code.keys_5990 as keys


def decode(pred):
    characters = keys.alphabet[:]
    characters = characters[1:] + u'卍'
    nclass = len(characters)
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and ((not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)


def predict(img, basemodel):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    width = int(width / scale)

    img = img.resize([width, 32], Image.ANTIALIAS)

    '''
    img_array = np.array(img.convert('1'))
    boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
    if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
        img = ImageOps.invert(img)
    '''
    img = np.array(img).astype(np.float32) / 255.0 - 0.5

    X = img.reshape([1, 32, width, 1])

    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]

    out = decode(y_pred)

    return out

