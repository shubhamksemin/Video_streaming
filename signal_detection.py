import matplotlib.image as mpimg 
import os 

from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from keras.utils.np_utils import to_categorical 
from tensorflow.keras.utils import image_dataset_from_directory 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from tensorflow.keras.models import Sequential 
from keras import layers 
from tensorflow import keras 
from tensorflow.keras.layers.experimental.preprocessing import Rescaling 
from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt 
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from glob import glob 
import cv2 

import warnings 
warnings.filterwarnings('ignore') 

# Extracting the compressed dataset. 
from zipfile import ZipFile data_path 
= '/content/traffic-sign-dataset-classification.zip' with 
	ZipFile(data_path, 'r') as zip
	: zip.extractall()

# path to the folder containing our dataset 
dataset = '../content/traffic_Data/DATA'

# path of label file 
labelfile = pd.read_csv('labels.csv') 

# Visualize some images from the dataset 
img = cv2.imread("/content/traffic_Data/DATA/10/010_0011.png") 
plt.imshow(img) 


img = cv2.imread("/content/traffic_Data/DATA/23/023_0001.png") 
plt.imshow(img) 
labelfile.head()


train_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset, validation_split=0.2, 
															subset='training', 
															image_size=( 
																224, 224), 
															seed=123, 
															batch_size=32) 
val_ds = tf.keras.preprocessing.image_dataset_from_directory(dataset, validation_split=0.2, 
															subset='validation', 
															image_size=( 
																224, 224), 
															seed=123, 
															batch_size=32) 

class_numbers = train_ds.class_names 
class_names = [] 
for i in class_numbers: 
	class_names.append(labelfile['Name'][int(i)]) 

data_augmentation = tf.keras.Sequential( 
	[ 
		tf.keras.layers.experimental.preprocessing.RandomFlip( 
			"horizontal", input_shape=(224, 224, 3)), 
		tf.keras.layers.experimental.preprocessing.RandomRotation(0.1), 
		tf.keras.layers.experimental.preprocessing.RandomZoom(0.2), 
		tf.keras.layers.experimental.preprocessing.RandomFlip( 
			mode="horizontal_and_vertical") 
	] 
) 
