import io
import os
import re
import sklearn.metrics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

class CustomModelSaver(tf.keras.callbacks.Callback):
    """ Custom Keras callback for saving weights of networks. """

    def __init__(self, checkpoint_dir, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        # self.task = task
        self.max_num_weights = max_num_weights

    def on_epoch_end(self, epoch, logs=None):
        """ At epoch end, weights are saved to checkpoint directory. """

        min_acc_file, max_acc_file, max_acc, num_weights = \
            self.scan_weight_files()

        cur_acc = logs["accuracy"]

        # Only save weights if test accuracy exceeds the previous best
        # weight file
        if cur_acc > max_acc:
            save_name = "weights.e{0:03d}-acc{1:.4f}.h5".format(
                epoch, cur_acc)

            # if self.task == '1':
            save_location = self.checkpoint_dir + os.sep + "your." + save_name
            print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
                    "maximum TEST accuracy.\nSaving checkpoint at {location}")
                    .format(epoch + 1, cur_acc, location = save_location))
            self.model.save_weights(save_location)
            # else:
            #     save_location = self.checkpoint_dir + os.sep + "vgg." + save_name
            #     print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) EXCEEDED previous "
            #            "maximum TEST accuracy.\nSaving checkpoint at {location}")
            #            .format(epoch + 1, cur_acc, location = save_location))
            #     # Only save weights of classification head of VGGModel
            #     self.model.head.save_weights(save_location)

            # Ensure max_num_weights is not exceeded by removing
            # minimum weight
            if self.max_num_weights > 0 and \
                    num_weights + 1 > self.max_num_weights:
                os.remove(self.checkpoint_dir + os.sep + min_acc_file)
        else:
            print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) DID NOT EXCEED "
                   "previous maximum TEST accuracy.\nNo checkpoint was "
                   "saved").format(epoch + 1, cur_acc))
    def scan_weight_files(self):
        """ Scans checkpoint directory to find current minimum and maximum
        accuracy weights files as well as the number of weights. """

        min_acc = float('inf')
        max_acc = 0
        min_acc_file = ""
        max_acc_file = ""
        num_weights = 0

        files = os.listdir(self.checkpoint_dir)

        for weight_file in files:
            if weight_file.endswith(".h5"):
                num_weights += 1
                file_acc = float(re.findall(
                    r"[+-]?\d+\.\d+", weight_file.split("acc")[-1])[0])
                if file_acc > max_acc:
                    max_acc = file_acc
                    max_acc_file = weight_file
                if file_acc < min_acc:
                    min_acc = file_acc
                    min_acc_file = weight_file

        return min_acc_file, max_acc_file, max_acc, num_weights


class PrintLayerOutput(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if not self.model.inputs:
            raise ValueError("Model has not been built. Ensure model is built and has input data.")
        
        # Access the output of the layer correctly, ensuring the layer has been built
        output_func = tf.keras.backend.function([self.model.input], [self.model.get_layer('layer_name').output])

        # Assuming you have some input data to feed
        # Example: let's say you use the first batch from your validation data
        input_data = next(iter(self.model.validation_data))[0]

        # Get the output for this batch
        layer_output = output_func([input_data])
        print("Output of layer at epoch {}: {}".format(epoch, layer_output))