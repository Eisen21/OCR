import tensorflow as tf
import cv2
import numpy as np
import os
import shutil
import shapely
from shapely.geometry import Polygon, MultiPoint

from detect.code.icdar import restore_rectangle
import detect.code.lanms as lanms

class Detector:

    def __resize_image__(self, im, max_side_len=512, square=False):
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

    def __detect__(self, score_map, geo_map, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
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
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
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

    def __sort_poly__(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def __crop_target__(self, box, image):
        # 上下左右都多出三个像素较好
        bndbox_left = min([box[0, 0], box[3, 0]]) - 3
        bndbox_top = min(box[0, 1], box[1, 1]) - 3
        bndbox_right = max(box[1, 0], box[2, 0]) + 3
        bndbox_bottom = max(box[3, 1], box[2, 1]) + 3
        target = image[bndbox_top:bndbox_bottom, bndbox_left:bndbox_right, :]
        height, width, channels = target.shape

        new_height = 32
        new_width = int(new_height * width / height)
        target_resize = cv2.resize(target, (new_width, new_height))

        return target_resize

    def __mul_box_IOU__(self, boxes, boxes_label, labels):
        if len(boxes_label) == 0:
            return 0.0

        if boxes_label is not None:
            num_boxes_label = len(boxes_label)

        if len(boxes) > 0:
            iou_matrix = np.zeros(num_boxes_label)
            for i in range(num_boxes_label):
                # 计算iou矩阵
                iou_matrix[i] = self.__IOU__(boxes, boxes_label[i])
                index_i = iou_matrix[:].argmax()
                if iou_matrix[index_i] > 1e-3:
                    boxes_name = labels[index_i]

                else:
                    boxes_name = None

        return boxes_name

    def __IOU__(self, box, box_label):
        """
        # 计算两四边形的IOU
        :param box: 输出检测框顶点坐标[x0, y0, x1, y1, x2, y2, x3, y3]
        :param box_label: 标签检测框顶点坐标[x0, y0, x1, y1, x2, y2, x3, y3]
        :return: IOU = S(quadrangle 交 quadrangle_label) / S(quadrangle 并 quadrangle_label)
        """
        ratio = 0.0

        box_np = np.array(box).reshape(4, 2)
        box_label = list(map(int, box_label))
        box_label_np = np.array(box_label).reshape(4, 2)
        # 四边形对象
        poly = Polygon(box_np).convex_hull
        poly_label = Polygon(box_label_np).convex_hull
        # print("poly:\n", poly)
        # print("poly_label:\n", poly_label)

        # 合并两个box坐标
        union_poly = MultiPoint(np.concatenate((box_np, box_label_np))).convex_hull

        # print("union_poly:\n", union_poly)
        if not poly.intersects(poly_label):  # 如果两四边形不相交
            ratio = 0.0
        else:
            try:
                # 相交面积
                inter_area = poly.intersection(poly_label).area
                # print("相交面积：", inter_area)
                # 并集面积
                union_area = union_poly.area
                # print("并集面积：", union_area)
                if union_area < 1e-3:
                    ratio = 0.0
                else:
                    ratio = float(inter_area) / float(union_area)
            except shapely.geos.TopologicalError:
                print('shapely.geos.TopologicalError occured, iou set to 0')
                ratio = 0.0

        return ratio

    def __load_template__(self, txt):
        f = open(txt, "r", encoding="utf-8")
        lines = f.readlines()
        boxes_list = []
        name_list = []
        for line in lines:
            words = line.strip().split(",")
            boxes_list.append(words[:8])
            name_list.append(words[-1])
        return boxes_list, name_list

    def __init__(self, pb_path, txt):
        with open(pb_path, 'rb') as f:
            serialized = f.read()
        tf.reset_default_graph()
        output_graph_def = tf.GraphDef()
        output_graph_def.ParseFromString(serialized)
        tf.import_graph_def(output_graph_def, name='')
        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)

        input_tensor_name = 'input_images:0'
        f_tensor_name = 'feature_fusion/Conv_7/Sigmoid:0'
        g_tensor_name = 'feature_fusion/concat_6:0'
        self.input_images = self.sess.graph.get_tensor_by_name(input_tensor_name)
        self.f_score = self.sess.graph.get_tensor_by_name(f_tensor_name)
        self.f_geometry = self.sess.graph.get_tensor_by_name(g_tensor_name)

        self.label_boxes, self.label_txt = self.__load_template__(txt)

    def run(self, im, is_write_in_file=False, output_path=None):
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        # im = cv2.imread(im_fn)[:, :, ::-1]
        im = im[:, :, ::-1]
        im_resized, (ratio_h, ratio_w) = self.__resize_image__(im,max_side_len=1024,square=True)
        score, geometry = self.sess.run([self.f_score, self.f_geometry], feed_dict={self.input_images: [im_resized]})
        boxes = self.__detect__(score_map=score, geo_map=geometry)
        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        targets = []
        for index in range(len(boxes)):
            box = boxes[index]
            box_name = self.__mul_box_IOU__(box, self.label_boxes, self.label_txt)
            # to avoid submitting errors
            box = self.__sort_poly__(box.astype(np.int32))
            target = self.__crop_target__(box, im[:, :, ::-1])
            targets.append(target)
            if is_write_in_file == True:
                save_path = "{}\{}.{}".format(str(output_path), box_name, "jpg")
                cv2.imwrite(save_path, target)
        return boxes, targets
