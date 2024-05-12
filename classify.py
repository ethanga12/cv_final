import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf
import keras

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
from utils import CustomModelSaver, PrintLayerOutput

def classify(): 
    my_model = tf.keras.models.load_model('model.h5')
    my_model