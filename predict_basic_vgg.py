import os
import random
import warnings
warnings.filterwarnings("ignore")
from utils import train_test_split

src = 'Dataset/Food/'

# Check if the dataset has been downloaded. If not, direct user to download the dataset first
if not os.path.isdir(src):
    print("""
          Dataset not found in your computer.
          Please follow the instructions in the link below to download the dataset:
          https://raw.githubusercontent.com/PacktPublishing/Neural-Network-Projects-with-Python/master/chapter4/how_to_download_the_dataset.txt
          """)
    quit()

# create the train/test folders if it does not exists already
if not os.path.isdir(src+'train/'):
    train_test_split(src)

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

from keras.preprocessing import image

# Define hyperparameters
FILTER_SIZE = 3
NUM_FILTERS = 256
INPUT_SIZE  = 256
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000//BATCH_SIZE
EPOCHS = 10

testing_data_generator = ImageDataGenerator(rescale = 1./255)
test_set = testing_data_generator.flow_from_directory(src+'Test/',
                                             target_size = (INPUT_SIZE, INPUT_SIZE),
                                             batch_size = BATCH_SIZE,
                                             class_mode = 'binary')

json_file = open('model_food_vgg.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_food_vgg.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# score = loaded_model.evaluate_generator(test_set, steps=100)

# for idx, metric in enumerate(loaded_model.metrics_names):
#     print("{}: {}".format(metric, score[idx]))







# dimensions of our images
img_width, img_height = 256, 256

# predicting images
burger_image = image.load_img('burger.jpg', target_size=(img_width, img_height))
pizza_image = image.load_img('pizza.jpg', target_size=(img_width, img_height))
burger = image.img_to_array(burger_image)
burger = np.expand_dims(burger, axis=0)

pizza = image.img_to_array(pizza_image)
pizza = np.expand_dims(pizza, axis=0)

images = np.vstack([burger, pizza])
classes = loaded_model.predict(images, batch_size=3)
print(classes)