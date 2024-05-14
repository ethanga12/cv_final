import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization

class GenreClassificationModel(tf.keras.Model):
    def __init__(self):
        super(GenreClassificationModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.vgg16 = [
           
             # Block 1
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv1", input_shape=(64, 173, 1)),
            Conv2D(64, 3, 1, padding="same",
                   activation="relu", name="block1_conv2"),
            MaxPool2D(2, name="block1_pool"),
            # Block 2
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv1"),
            Conv2D(128, 3, 1, padding="same",
                   activation="relu", name="block2_conv2"),
            MaxPool2D(2, name="block2_pool"),
            # Block 3
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv1"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv2"),
            Conv2D(256, 3, 1, padding="same",
                   activation="relu", name="block3_conv3"),
            MaxPool2D(2, name="block3_pool"),
            # Block 4
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block4_conv3"),
            MaxPool2D(2, name="block4_pool"),
            # Block 5
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv1"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv2"),
            Conv2D(512, 3, 1, padding="same",
                   activation="relu", name="block5_conv3"),
            MaxPool2D(2, name="block5_pool")
        ]
        self.head = [ 
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.2),
            Dense(512, activation="relu"),
            Dropout(0.2),
            Dense(10, activation="softmax")
        ]

        self.vgg16 = tf.keras.Sequential(self.vgg16, name="vgg_base")
        self.head = tf.keras.Sequential(self.head, name="head")

        


    
    def call(self, x):
        count = 0
        vals = x
        x = self.vgg16(x)
        x = self.head(x)
        return x
    
    @staticmethod
    def loss_fn(labels, predictions):
       loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
       return loss(labels, predictions)
