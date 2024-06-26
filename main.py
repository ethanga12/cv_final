import os

from datetime import datetime
import tensorflow as tf
import plotly.graph_objects as go


from genre_classification import GenreClassificationModel
import numpy as np

# %matplotlib inline
from preprocess import Datasets
from utils import CustomModelSaver, PrintLayerOutput, DragDropApp
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(model, datasets, logs_path): #train model which splits up preprocessed data and tests when final epoch is reached
    with tf.device('GPU'):
        callback_list = [
            CustomModelSaver('checkpoints', max_num_weights=5),
            tf.keras.callbacks.TensorBoard(
                log_dir="logs",
                update_freq='batch',
                profile_batch=0)
            ]
        for images, labels in datasets.train_data:
            print(images.shape, labels)
            break # Just to check the first batch
        input_shape = (64, 173, 1)
        num_classes = 10
        x_train, y_train = zip(*datasets.train_data)
        x_test, y_test = zip(*datasets.test_data)
        x_val, y_val = zip(*datasets.val_data)
        x_train = np.array([x.reshape(input_shape) for x in x_train])
        x_test = np.array([x.reshape(input_shape) for x in x_test])
        x_val = np.array([x.reshape(input_shape) for x in x_val])
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_val = np.array(y_val)
        print(f"x_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"x_val shape: {x_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
                epochs=20, callbacks=callback_list, batch_size=64)
        test(model, x_val, y_val)
        

def test(model, x, y):
    model.evaluate(x=x, y=y, verbose=2)

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
    
    #UNCOMMENT THIS TO TEST
    # datasets = Datasets("../cv_final")
    # my_val = datasets.val_data
    # model = GenreClassificationModel()
    # model(tf.keras.Input(shape=(64, 173, 1)))
    # model.load_weights('../cv_final/genre_classify_e019-acc0.8973.h5')
    # input_shape = (64, 173, 1)
    # x_val, y_val = zip(*my_val)
    # x_val = np.array([x.reshape(input_shape) for x in x_val])
    # y_val = np.array(y_val)
    # print(f"x_val shape: {x_val.shape}")
    # print(f"y_val shape: {y_val.shape}")
    # model.load_weights('../cv_final/genre_classify_e019-acc0.8973.h5')
    # model.compile(
    #     optimizer='adam',
    #     loss=model.loss_fn,
    #     metrics=['accuracy']
    # )
    # test(model, x_val, y_val)
    
    # TKINTER GUI
    model = GenreClassificationModel()
    model(tf.keras.Input(shape=(64, 173, 1)))
    model.load_weights('../cv_final/genre_classify_e019-acc0.8973.h5')
    root = TkinterDnD.Tk()
    app = DragDropApp(root, model)
    root.mainloop()

main()

