import glob

import numpy
import numpy as np
from tensorflow.keras.models import model_from_json
from PIL import Image
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



def load():
    json_filename = "keras_trained_video.json"
    with open(json_filename, "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    # Загружаем веса в модель
    h5_filename = "keras_trained_video.h5"
    model.load_weights(h5_filename)
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop",
                  metrics=["accuracy"])
    return model


def predict_present():
    model = load()
    pravilno = 0
    k = 0
    for filename in glob.glob('test/present/*.jpg'):
        img = load_img(filename, color_mode='rgb',
                       target_size=(116, 116))
        img = img_to_array(img, dtype='float32') / 255.
        img = np.array(img).reshape(-1, 116, 116, 3)
        result = model.predict(img)
        if result + 0.5 > 1.:
            pravilno += 1
        k += 1
    print('Accuracy of present ' + str(pravilno) + ' out of ' + str(k))


def predict_not():
    model = load()
    pravilno = 0
    k = 0
    for filename in glob.glob('test/not/*.jpg'):
        img = load_img(filename, color_mode='rgb',
                       target_size=(116, 116))
        img = img_to_array(img, dtype='float32') / 255.
        img = np.array(img).reshape(-1, 116, 116, 3)
        result = model.predict(img)
        if result + 0.5 <= 1.:
            pravilno += 1
        k += 1
    print('Accuracy of not present ' + str(pravilno) + ' out of ' + str(k))

predict_present()
predict_not()