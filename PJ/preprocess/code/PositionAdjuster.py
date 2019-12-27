import numpy as np
import cv2
import os
import shutil
from shapely.geometry import Polygon     #多边形
import itertools
from detect.code.ReferPointDetector import Detector
from recognize.code.Recognizer import Recognizer
from PJ import constant

class ReferencePoint():

    def __init__(self, name, image_array, box=None,  x=None, y=None):
        self.name = name
        self.image_array = image_array
        if box is not None:
            x1 = box[0, 0]
            y1 = box[0, 1]
            x2 = box[1, 0]
            y2 = box[1, 1]
            x3 = box[2, 0]
            y3 = box[2, 1]
            x4 = box[3, 0]
            y4 = box[3, 1]
            self.x = int((x1 + x2 + x3 + x4) / 4)
            self.y = int((y1 + y2 + y3 + y4) / 4)
            self.box = box
        if x is not None and y is not None:
            self.x = x
            self.y = y

    def init_from_txt(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_image_array(self):
        return self.image_array

    def set_image_array(self, image_array):
        self.image_array = image_array

    def get_box(self):
        return self.box

    def set_box(self, box):
        x1 = box[0, 0]
        y1 = box[0, 1]
        x2 = box[1, 0]
        y2 = box[1, 1]
        x3 = box[2, 0]
        y3 = box[2, 1]
        x4 = box[3, 0]
        y4 = box[3, 1]
        self.x = (x1 + x2 + x3 + x4) / 4
        self.y = (y1 + y2 + y3 + y4) / 4
        self.box = box

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    def get_xy(self):
        return np.array([self.x, self.y])

class PositionAdjuster:

    def __load_template__(self, txt):
        f = open(txt, "r", encoding="utf-8")
        lines = f.readlines()
        rp_name_lib = []
        reference_point_list = []
        for line in lines:
            name, x, y = line.split(",")
            reference_point = ReferencePoint(name, None, x=x, y=y)
            reference_point_list.append(reference_point)
            rp_name_lib.append(name)
        return reference_point_list, rp_name_lib


    def __match_rp_name_lib__(self, name_old):
        # 通过编辑距离不可取，识别号与名称，识别号与纳税人识别号，距离都是3
        # 采用重复字母个数作为相似度
        # 可能也会返回None
        name_new = None
        similarity_max = 0
        length_diff_min = 50
        for name_in_lib in self.rp_name_lib:
            similarity = 0
            for letter in name_old:
                if letter in name_in_lib:
                    similarity = similarity + 1
            if similarity > similarity_max:
                length_diff_min = abs(len(name_in_lib)-len(name_old))
                similarity_max = similarity
                name_new = name_in_lib
            elif similarity_max!=0 and similarity == similarity_max:
                length_diff = abs(len(name_in_lib)-len(name_old))
                if length_diff < length_diff_min:
                    length_diff_min = length_diff
                    similarity_max = similarity
                    name_new = name_in_lib
        return name_new

    def __find_trp_by_name__(self, name):
        for trp in self.template_rps:
            if trp.get_name() == name:
                return trp

    def __pointarea__(self,combine_point):
        self.line = []
        for j in combine_point:
            pointX = j.get_x()
            pointY = j.get_y()
            self.line.append(pointX)
            self.line.append(pointY)
        a = np.array(self.line).reshape(4, 2)  # 四边形二维坐标表示
        poly = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
        area = poly.area
        return area

    def __init__(self, txt, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer
        self.template_rps, self.rp_name_lib = self.__load_template__(txt)
        self.line = []
        self.polyarea = 0
        # self.temp_out_path = os.path.abspath(os.path.join(os.getcwd(), "../temp"))  # 用于放置参考点检测后识别前的小图片
        self.temp_out_path = os.path.join(constant.PROJECT_PATH, "temp", "preprocess")  # 用于放置参考点检测后识别前的小图片
        self.image_name = ""    # 用于记录图片名，将矫正后的图片写入

    # 第一步：检测参考点，并裁剪保存到临时目录，然后对临时目录中的图片进行识别，匹配，得到参考点的名字和位置信息
    def rp_detect(self, image):
        if os.path.exists(self.temp_out_path):
            shutil.rmtree(self.temp_out_path)
        os.makedirs(self.temp_out_path)
        print("参考点结果:", self.temp_out_path)
        self.boxes, self.imagearray_list = self.detector.run(image, is_write_in_file=True, output_path=self.temp_out_path)
        #识别参考点名字并和参考点库进行匹配
        reference_points = []
        for i in range(len(self.boxes)):
            name = self.recognizer.run(os.path.join(self.temp_out_path, str(i)+".jpg"))
            name_new = self.__match_rp_name_lib__(name)
            # print("匹配前：{}\t匹配后：{}".format(name, name_new))
            if name_new is not None:
                reference_point = ReferencePoint(name_new,self.imagearray_list[i], self.boxes[i])
                reference_points.append(reference_point)
        return reference_points

    # 第二步：通过面积，筛选出四个参考点
    def rp_select(self,drp_list):
        combineList = list(itertools.combinations(drp_list, 4))
        for i in combineList:
            polyarea1 = self.__pointarea__(i)
            if polyarea1 > self.polyarea:
                rp_point = i
                self.polyarea = polyarea1
        return rp_point

    # 第三部：根据参考点，进行透视变换
    def perspective_transform(self, image, drp_list):
        #  图像参考点
        drps = np.float32([drp_list[0].get_xy(), drp_list[1].get_xy(), drp_list[2].get_xy(), drp_list[3].get_xy()])
        trp0 = self.__find_trp_by_name__(drp_list[0].get_name())
        trp1 = self.__find_trp_by_name__(drp_list[1].get_name())
        trp2 = self.__find_trp_by_name__(drp_list[2].get_name())
        trp3 = self.__find_trp_by_name__(drp_list[3].get_name())
        trps = np.float32([trp0.get_xy(), trp1.get_xy(), trp2.get_xy(), trp3.get_xy()])
        M = cv2.getPerspectiveTransform(drps, trps)
        target_perspective = cv2.warpPerspective(image, M, (1024, 585))
        target_perspective = target_perspective.astype(np.uint8)
        # cv2.imshow("wrap", target_perspective)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(self.temp_out_path, self.image_name), target_perspective)
        return target_perspective

    # 串接之前的实现步骤
    def run(self, image, image_name):
        # _, self.image_name = os.path.split(image_path)
        # image = cv2.imread(image_path)
        self.image_name = image_name
        drps_list = self.rp_detect(image)
        # print("最后结果：" + str(len(drps_list)))
        # for drp in drps_list:
        #     drp = ReferencePoint(name=drp.get_name(), image_array=np.array([0]), x=drp.get_x(), y=drp.get_y())
        #     print(drp.get_name(), drp.get_xy())
        # print("选中结果：")
        drps_list_selected = self.rp_select(drps_list)
        # for drp in drps_list_selected:
        #     print(drp.get_name(), drp.get_xy())
        self.perspective_transform(image, drps_list_selected)

if __name__ == '__main__':
    rp_dector = Detector(pb_path ="D:\Picture\RP\model\MobileNet_1224\MobileNet_1224.pb")
    print("load rp_dector")
    recognizer = Recognizer(r"D:\PythonProject\PJ\recognize\model\weights_densenet.h5")
    print("load recognizer")
    position_adjuster = PositionAdjuster(r"D:\PythonProject\PJ\preprocess\mask\template_mask.txt", rp_dector, recognizer)
    print("init RP")
    image_path = r"C:\Users\Eisen\Desktop\baidu\baidu5.png"
    position_adjuster.run(image_path)


