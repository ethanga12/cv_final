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
from video_edit_model import VideoEditModel
from music_feature_extraction import MusicFeatureExtractorModel as Extractor
import cv2


class CustomModelSaver(tf.keras.callbacks.Callback): #borrowed from hw5
    """ Custom Keras callback for saving weights of networks. """

    def __init__(self, checkpoint_dir, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
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
          
            if self.max_num_weights > 0 and \
                    num_weights + 1 > self.max_num_weights:
                os.remove(self.checkpoint_dir + os.sep + min_acc_file)
        else:
            print(("\nEpoch {0:03d} TEST accuracy ({1:.4f}) DID NOT EXCEED "
                   "previous maximum TEST accuracy.\nNo checkpoint was "
                   "saved").format(epoch + 1, cur_acc))
    def scan_weight_files(self): #borrowed from hw5
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


class PrintLayerOutput(tf.keras.callbacks.Callback): #borrowed from hw5
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

class DragDropApp: #our drag drop gui built with tkinter!
    def __init__(self, root, model):
        self.root = root
        self.root.title("Music Video Maker!")
        self.root.geometry("600x800")
        
        self.frame = tk.Frame(root, bd=2, relief="sunken", width=600, height=800, bg="lightblue")
        self.frame.pack(fill="both", expand=True)
        self.frame.drop_target_register(DND_FILES)
        self.frame.dnd_bind('<<Drop>>', self.drop)
        
        self.label = tk.Label(self.frame, text="Drag and drop files here (songs must be at least 30 seconds)", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.genre = ''
        self.songFile = ''
        self.music_extractor = None
        self.wav_label = tk.Label(self.frame, text="Files", font=("Helvetica", 14))
        self.wav_label.pack(pady=5)
        self.wav_listbox = tk.Listbox(self.frame, width=80, height=5)
        self.wav_listbox.pack(pady=5)
        self.button = tk.Button(self.frame, text="Use Webcam", command=self.use_webcam, bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.button.pack(padx=30, pady=30)
        self.model = model
        self.blues_genre =  tk.Button(self.frame, text="Blues", command=lambda: self.set_genre('blues'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.classical_genre =  tk.Button(self.frame, text="Classical", command=lambda: self.set_genre('classical'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.country_genre =  tk.Button(self.frame, text="Country", command=lambda: self.set_genre('country'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.disco_genre =  tk.Button(self.frame, text="Disco", command=lambda: self.set_genre('disco'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.hiphop_genre =  tk.Button(self.frame, text="Hip Hop", command=lambda: self.set_genre('hiphop'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.jazz_genre =  tk.Button(self.frame, text="Jazz", command=lambda: self.set_genre('jazz'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.metal_genre =  tk.Button(self.frame, text="Metal", command=lambda: self.set_genre('metal'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.pop_genre =  tk.Button(self.frame, text="Pop", command=lambda: self.set_genre('pop'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.reggae_genre =  tk.Button(self.frame, text="Reggae", command=lambda: self.set_genre('reggae'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.rock_genre =  tk.Button(self.frame, text="Rock", command=lambda: self.set_genre('rock'), bd=5, relief="solid", bg="blue", fg="black", padx=10, pady=5, font=("Helvetica", 14))
        self.blues_genre.pack(pady=5)
        self.classical_genre.pack(pady=5)
        self.country_genre.pack(pady=5)
        self.disco_genre.pack(pady=5)
        self.hiphop_genre.pack(pady=5)
        self.jazz_genre.pack(pady=5)
        self.metal_genre.pack(pady=5)
        self.pop_genre.pack(pady=5)
        self.reggae_genre.pack(pady=5)
        self.rock_genre.pack(pady=5)
    
    def set_genre(self, genre):
        self.genre = genre
        self.wav_label['text'] = "Genre set to " + genre
    def drop(self, event):
        files = self.root.tk.splitlist(event.data)
        for file in files:
            if file.lower().endswith('.mp4'):
                if self.genre == "blues":
                    features = self.music_extractor.extract_blues()
                    video_editor = VideoEditModel(file, self.songFile, features, 'res.mp4')
                    video_editor.video_edits_blues()
                    self.wav_label['text'] = "Results saved in res.mp4"
                elif self.genre == "reggae":
                    features = self.music_extractor.extract_reggae()
                    video_editor = VideoEditModel(file, self.songFile, features, 'res.mp4')
                    video_editor.video_edits_reggae()
                    self.wav_label['text'] = "Results saved in res.mp4"
                elif self.genre == "metal":
                    metal_features = self.music_extractor.extract_metal()
                    video_editor = VideoEditModel(file, self.songFile, metal_features, 'res2.mp4')
                    video_editor.video_edits_metal()
                    self.wav_label['text'] = "Results saved in res.mp4"
                elif self.genre == "pop":
                    features = self.music_extractor.extract_pop()
                    video_editor = VideoEditModel(file, self.songFile, features, 'res.mp4')
                    video_editor.video_edits_pop()
                    self.wav_label['text'] = "Results saved in res.mp4"
                elif self.genre == "jazz":
                    features = self.music_extractor.extract_jazz()
                    video_editor = VideoEditModel(file, self.songFile, features, 'res.mp4')
                    video_editor.video_edits_jazz()
                    self.wav_label['text'] = "Results saved in res.mp4"
                elif self.genre == "hiphop":
                    features = self.music_extractor.extract_hiphop()
                    video_editor = VideoEditModel(file, self.songFile, features, 'res.mp4')
                    video_editor.video_edits_hiphop()
                    self.wav_label['text'] = "Results saved in res.mp4"
                elif self.genre == "rock":
                    features = self.music_extractor.extract_rock()
                    video_editor = VideoEditModel(file, self.songFile, features, 'res.mp4')
                    video_editor.video_edits_rock()
                    self.wav_label['text'] = "Results saved in res.mp4"
                elif self.genre == "classical":
                    features = self.music_extractor.extract_classical()
                    video_editor = VideoEditModel(file, self.songFile, features, 'res.mp4')
                    video_editor.video_edits_classical()
                    self.wav_label['text'] = "Results saved in res.mp4"
                elif self.genre == "country":
                    features = self.music_extractor.extract_country()
                    video_editor = VideoEditModel(file, self.songFile, features, 'res.mp4')
                    video_editor.video_edits_country()
                    self.wav_label['text'] = "Results saved in res.mp4"
                elif self.genre == "disco":
                    features = self.music_extractor.extract_disco()
                    video_editor = VideoEditModel(file, self.songFile, features, 'res.mp4')
                    video_editor.video_edits_disco()
                    self.wav_label['text'] = "Results saved in res.mp4"
                else:
                    self.wav_label['text'] = "Please upload a wav file first."
                self.wav_listbox.insert(tk.END, file)
            elif file.lower().endswith('.wav'):
                self.songFile = file
                print(self.songFile)
                res = classify(self.model, file)
                print
                self.genre = res
                self.wav_listbox.insert(tk.END, 'Genre for ' + file + ': ' + res)
                self.music_extractor = Extractor(file)
                self.set_genre(res)
            else:
                tk.messagebox.showwarning("Unsupported File", f"Unsupported file type: {file}")


    def use_webcam(self): #https://www.geeksforgeeks.org/saving-a-video-using-opencv/
        print('using webcam')
        vid = cv2.VideoCapture(0) 
        if (vid.isOpened() == False):  
            print("Error reading video file") 
        
        frame_width = int(vid.get(3))
        frame_height = int(vid.get(4))
        
        size = (frame_width, frame_height) 
        result = cv2.VideoWriter('webcam_vid.mp4',  
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         24, size) 
        
        while(True): 
            ret, frame = vid.read() 
            if ret == True:  
  
                result.write(frame) 
                cv2.imshow('Frame', frame) 
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
            else: 
                break
        vid.release() 
        result.release()
        cv2.destroyAllWindows() 
    