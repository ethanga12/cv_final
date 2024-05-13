import io
import os
import re
import sklearn.metrics
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from classify import classify


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

class DragDropApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Drag and Drop File Interface")
        self.root.geometry("600x500")
        
        self.frame = tk.Frame(root, bd=2, relief="sunken", width=600, height=400)
        self.frame.pack(fill="both", expand=True)
        self.frame.drop_target_register(DND_FILES)
        self.frame.dnd_bind('<<Drop>>', self.drop)
        
        self.label = tk.Label(self.frame, text="Drag and drop files here", font=("Helvetica", 16))
        self.label.pack(pady=10)
        
        self.mp4_label = tk.Label(self.frame, text="MP4 Files", font=("Helvetica", 14))
        self.mp4_label.pack(pady=5)
        self.mp4_listbox = tk.Listbox(self.frame, width=80, height=5)
        self.mp4_listbox.pack(pady=5)
        
        self.wav_label = tk.Label(self.frame, text="WAV Files", font=("Helvetica", 14))
        self.wav_label.pack(pady=5)
        self.wav_listbox = tk.Listbox(self.frame, width=80, height=5)
        self.wav_listbox.pack(pady=5)
        self.model = model

    def drop(self, event):
        files = self.root.tk.splitlist(event.data)

        files = self.root.tk.splitlist(event.data)
        for file in files:
            if file.lower().endswith('.mp4'):
                self.mp4_listbox.insert(tk.END, file)
            elif file.lower().endswith('.wav'):
                classify(self.model, file)
                self.wav_listbox.insert(tk.END, file)
            else:
                tk.messagebox.showwarning("Unsupported File", f"Unsupported file type: {file}")

        # for file in files:
        #     print(file)
           
        #     self.file_listbox.insert(tk.END, file)