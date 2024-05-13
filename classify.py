import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from scipy.stats import mode
from genre_classification import GenreClassificationModel
from skimage.transform import resize


from skimage.io import imread
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style('whitegrid')
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
from preprocess import Datasets
from utils import CustomModelSaver, PrintLayerOutput
import librosa
import h5py
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

def classify(): 
    model = GenreClassificationModel()
    model(tf.keras.Input(shape=(64, 173, 1)))
    model.load_weights('../cv_final/your.weights.e014-acc0.8881.h5')

    # with h5py.File('../cv_final/your.weights.e014-acc0.8881.h5', 'r') as file:
    #     print("Keys in the HDF5 file:")
    #     for key in file.keys():
    #         print(key)
    # my_model =  keras.models.load_model('../cv_final/your.weights.e014-acc0.8881.h5')
    res = []
    for i in range(28):
        #Dancing queen: Disco
        #Miss the rage:  Pop
        #See you later: Jazz (fine, mostly piano)
        #retro city: pop (ya)
        #old school bass: pop (techno kinda)
        #pop trap: pop 
        y, sr = librosa.load('songs/disco/pop-trap-wav-182460.wav', mono=True, duration=2, offset=i*2)
        ps = librosa.feature.melspectrogram(y=y, sr=sr, hop_length = 256, n_fft = 512, n_mels=64)
        # print(ps.shape)
        ps = librosa.power_to_db(ps**2)
        ps = ps.reshape(1, 64, 173, 1)
        # print(ps.shape)
        x = model.predict(ps)
        print(np.argmax(x[0]))
        res.append(np.argmax(x[0]))
    res = np.array(res)
    # print(res)
    genres = {0: 'blues', 1: 'classical', 2: 'country', 3 : 'disco', 4 : 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    print(genres[mode(res).mode[0]])

classify()

