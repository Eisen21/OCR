# 检查标注结果
import os
import re
import cv2
import time
import shutil
import shapely
import numpy as np
import tensorflow as tf
import detect.code.lanms as lanms
from shapely.geometry import Polygon, MultiPoint
from detect.code.icdar import restore_rectangle

max_side_len = 1024


def crop_target(box, image):
    # 上下左右都多出三个像素较好
    bndbox_left = min([box[0, 0], box[3, 0]]) - 3
    bndbox_top = min(box[0, 1], box[1, 1]) - 3
    bndbox_right = max(box[1, 0], box[2, 0]) + 3
    bndbox_bottom = max(box[3, 1], box[2, 1]) + 3
    target = image[bndbox_top:bndbox_bottom, bndbox_left:bndbox_right,:]
    height, width, channels = target.shape

    new_height = 32
    new_width = int(new_height * width / height)
    target_resize = cv2.resize(target, (new_width, new_height))

    return target_resize


def resize_image(im, max_side_len=512,square=False):
    """
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    """
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    if square:
        img_new = np.zeros(shape=(max_side_len, max_side_len, 3), dtype="uint8")
        img_new[:int(resize_h), :int(resize_w)] = im[:, :]

        im = img_new

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    """
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    """
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)  # score_map_thresh
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    # nms part
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return None

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def read_boxes(box_path):
    boxes = []
    labels = []
    if os.path.exists(box_path):
        with open(box_path, 'rb') as f:
            for line in f.readlines():
                data = str(line.decode('UTF-8')).split(',')
                if len(data) < 9:
                    continue
                # 过滤掉非数字字符
                for index in range(8):
                    data[index] = re.sub('\D', '', data[index])
                box = np.array(data[0:8]).astype(int)
                boxes.append(box.reshape((4, 2)))
                labels.append(data[8].rstrip('\r\n'))
    else:
        print(box_path, '不存在！')
    boxes = np.array(boxes)
    return boxes, labels


def run_detect(image_path, pb_path):
    image_name = os.path.basename(image_path).split(".")[0]
    image_name = "{}_detect".format(image_name)
    # 修改相对路径
    output_path = os.path.join("D:/OCR_zzx/static/image_segment", image_name)
    output_path_box = "D:/OCR_zzx/static/image_upload"
    save_path_upload = "{}/{}.{}".format(str(output_path_box), image_name, "jpg")
    # 判断文件路径是否存在,存在则删除
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        input_tensor_name = 'input_images:0'
        f_tensor_name = 'feature_fusion/Conv_7/Sigmoid:0'
        g_tensor_name = 'feature_fusion/concat_6:0'
        input_images = sess.graph.get_tensor_by_name(input_tensor_name)
        f_score = sess.graph.get_tensor_by_name(f_tensor_name)
        f_geometry = sess.graph.get_tensor_by_name(g_tensor_name)

        im_fn = image_path
        label_path = im_fn.rstrip('.jpg') + '.txt'
        im = cv2.imread(im_fn)[:, :, ::-1]

        im_resized, (ratio_h, ratio_w) = resize_image(im,max_side_len=1024,square=True)

        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})

        boxes = detect(score_map=score, geo_map=geometry)

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        for index in range(len(boxes)):
            box = boxes[index]
            # to avoid submitting errors
            box = sort_poly(box.astype(np.int32))
            target = crop_target(box, im[:, :, ::-1])
            save_path = "{}\{}.{}".format(str(output_path), index, "jpg")
            cv2.imwrite(save_path, target)

        for box in boxes:
            # to avoid submitting errors
            box = sort_poly(box.astype(np.int32))
            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=1)

        cv2.imwrite(save_path_upload, im[:, :, ::-1])
        time.sleep(1)
        return "{}.{}".format(image_name, "jpg")

if __name__ == '__main__':
    # 修改相对路径
    pb_path = "D:/OCR_zzx/detect/model/MobileNet_1219.pb"
    image_path = "D:/OCR_zzx/static/image_upload/02_1.jpg"
    run_detect(image_path, pb_path)
