import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import keras
import plotly.graph_objects as go


from genre_classification import GenreClassificationModel
from skimage.transform import resize


from skimage.io import imread
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
from preprocess import Datasets
from utils import CustomModelSaver, PrintLayerOutput, DragDropApp
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(model, datasets, logs_path): #train model which splits up preprocessed data
    with tf.device('GPU'):
        callback_list = [
            CustomModelSaver('checkpoints', max_num_weights=5),
            tf.keras.callbacks.TensorBoard(
                log_dir="logs",
                histogram_freq=1,
                profile_batch=0)
            ]
        for images, labels in datasets.train_data:
            print(images.shape, labels)
            break # Just to check the first batch
        input_shape = (64, 173, 1)
        num_classes = 10
        x_train, y_train = zip(*datasets.train_data)
        x_test, y_test = zip(*datasets.test_data)
        x_train = np.array([x.reshape(input_shape) for x in x_train])
        x_test = np.array([x.reshape(input_shape) for x in x_test])
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
                epochs=15, callbacks=callback_list, batch_size=64)
        

def test(model, test_data):
    model.evaluate(x=test_data, verbose=2)

def main():
    #UNCOMMENT THIS TO TRAIN
    # time_now = datetime.now()
    # timestamp = time_now.strftime("%m%d%y-%H%M%S")
    # init_epoch = 0
    # datasets = Datasets("../cv_final")
    # model = GenreClassificationModel()
    # model(tf.keras.Input(shape=(64, 173, 1)))
    # model.summary()

    # model.compile(
    #     optimizer='adam',
    #     loss=model.loss_fn,
    #     metrics=['accuracy']
    # )
    # logs_path = "logs/" + timestamp
    # train(model, datasets, logs_path)

    #TKINTER GUI
    model = GenreClassificationModel()
    model(tf.keras.Input(shape=(64, 173, 1)))
    model.load_weights('../cv_final/your.weights.e014-acc0.8881.h5')
    root = TkinterDnD.Tk()
    app = DragDropApp(root, model)
    root.mainloop()

main()

