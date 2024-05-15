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
from video_edit_model import VideoEditModel


from skimage.io import imread
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
from preprocess import Datasets
import librosa
import h5py
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

def classify(model, songname): #classifies given song based off majority vote
 
    res = []
    for i in range(14): 
        y, sr = librosa.load(songname, mono=True, duration=2, offset=i*2)
        ps = librosa.feature.melspectrogram(y=y, sr=sr, hop_length = 256, n_fft = 512, n_mels=64)
        ps = librosa.power_to_db(ps**2)
        ps = ps.reshape(1, 64, 173, 1)
        x = model.predict(ps, verbose=0)
        res.append(np.argmax(x[0]))
    res = np.array(res)
    genres = {0: 'blues', 1: 'classical', 2: 'country', 3 : 'disco', 4 : 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
    return genres[mode(res).mode[0]]


