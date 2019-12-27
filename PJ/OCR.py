import cv2
import os
import glob
from PJ.detect.code.ReferPointDetector import Detector as ReferPointDetector
from PJ.detect.code.ContextDetector import Detector as ContextDetector
from PJ.recognize.code.Recognizer import Recognizer
from PJ.preprocess.code.PositionAdjuster import PositionAdjuster
from PJ import constant


def run(image_path):

    rp_detector = ReferPointDetector(pb_path = "D:\Picture\RP\model\MobileNet_1224\MobileNet_1224.pb")
    print("load rp_dector")
    recognizer = Recognizer(r"D:\PythonProject\PJ\recognize\model\weights_densenet.h5")
    print("load recognizer")
    position_adjuster = PositionAdjuster(r"D:\PythonProject\PJ\preprocess\mask\template_mask.txt", rp_detector, recognizer)
    print("init RP")

    _, image_name = os.path.split(image_path)
    image = cv2.imread(image_path)
    position_adjuster.run(image, image_name)

    image_path = os.path.join(constant.PROJECT_PATH, "temp", "preprocess", image_name)
    image = cv2.imread(image_path)
    detect_out_path = os.path.join(constant.PROJECT_PATH, "temp", "detect")
    txt_path = os.path.join(constant.PROJECT_PATH, "detect/mask", "03.txt")
    context_detector = ContextDetector(pb_path = "D:\PythonProject\PJ\detect\model\MobileNet_1219.pb", txt=txt_path)
    context_detector.run(image, True, detect_out_path)
    image_file_list = glob.glob(detect_out_path + "/*.*")
    dict = {}
    for image_file in image_file_list:
        result = recognizer.run(image_file)
        _, name_ext = os.path.split(image_file)
        name, extension = os.path.splitext(name_ext)
        dict[name] = result
    return dict

if __name__ == '__main__':
    image_path = r"C:\Users\Eisen\Desktop\baidu\baidu5.png"
    result = run(image_path)
    for k in result.keys():
        print(k + ":" + result.get(k))





