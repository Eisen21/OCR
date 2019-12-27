from keras.models import load_model
from PIL import Image
import numpy as np
import recognize.code.keys_5990 as keys

class Recognizer:

    def __decode__(self, pred):
        characters = keys.alphabet[:]
        characters = characters[1:] + u'卍'
        nclass = len(characters)
        char_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != nclass - 1 and (
                    (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
                char_list.append(characters[pred_text[i]])
        return u''.join(char_list)

    def __predict__(self, img, basemodel):
        width, height = img.size[0], img.size[1]
        scale = height * 1.0 / 32
        width = int(width / scale)

        img = img.resize([width, 32], Image.ANTIALIAS)

        img = np.array(img).astype(np.float32) / 255.0 - 0.5

        X = img.reshape([1, 32, width, 1])

        y_pred = basemodel.predict(X)
        y_pred = y_pred[:, :, :]

        out = self.__decode__(y_pred)

        return out

    def __init__(self, model_path):
        self.basemodel = load_model(model_path, compile=False)

    def run(self, image_path):
        img_src = Image.open(image_path)
        img = img_src.convert("L")
        predict_text = self.__predict__(img, self.basemodel)
        return predict_text

if __name__ == '__main__':

    recognizer = Recognizer(r"D:\PythonProject\PJ\recognize\model\weights_densenet.h5")
    result = recognizer.run(r"D:\PythonProject\PJ\out\8.jpg")
    print(result)
    result = recognizer.run(r"D:\PythonProject\PJ\out\9.jpg")
    print(result)