import os 
import sys
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import librosa
# np.set_printoptions(threshold=sys.maxsize)
class Datasets():

    def __init__(self, data_path): 
        self.data_path = data_path
        self.mean = np.zeros((288, 432, 4))
        self.std = np.ones((288, 432, 4))
        # self.calc_mean_and_std()
        # self.classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        self.classes = [""] * 10
        self.idx_to_class = {}
        self.class_to_idx = {}
        
        data = self.wav_to_mel()
        self.train_data = data[:int(len(data)*0.8)]
        self.val_data = data[int(len(data)*0.8):int(len(data)*0.9)]
        self.test_data = data[int(len(data)*0.9):]
    
    def calc_mean_and_std(self):
        file_list = []
        for root, _, files in os.walk(os.path.join(self.data_path, "train/")):
            for name in files:
                if name.endswith(".png"):
                    file_list.append(os.path.join(root, name))


        # Shuffle filepaths
        # random.shuffle(file_list)
        # print('LOOKATME')
        # print(Image.open(file_list[0]))


        # Take sample of file paths
        file_list = file_list[:100]

        # Allocate space in memory for images
        data_sample = np.zeros(
            (900, 288, 432, 4))

        # Import images
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path)
            img = np.resize(img, (288, 432, 4))
            img = np.array(img, dtype=np.float32)
            img /= 255

            # # Grayscale -> RGB
            # if len(img.shape) == 2:
            #     img = np.stack([img, img, img], axis=-1)

            data_sample[i] = img
        self.mean = np.mean(data_sample, axis=0)
        self.std = np.std(data_sample, axis=0)

        # ==========================================================

        print("Dataset mean shape: [{0}, {1}, {2}]".format(
            self.mean.shape[0], self.mean.shape[1], self.mean.shape[2]))

        print("Dataset mean top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.mean[0,0,0], self.mean[0,0,1], self.mean[0,0,2]))

        print("Dataset std shape: [{0}, {1}, {2}]".format(
            self.std.shape[0], self.std.shape[1], self.std.shape[2]))

        print("Dataset std top left pixel value: [{0:.4f}, {1:.4f}, {2:.4f}]".format(
            self.std[0,0,0], self.std[0,0,1], self.std[0,0,2]))
        
    def standardize(self, img):
        # print(self.mean.shape)
        # print(self.std.shape)
        return (img - self.mean) / self.std
    
    def preprocess_fn(self, img):
        # print(img.shape)
        img = np.resize(img, (288, 432, 4))
        img = np.array(img, dtype=np.float32)
        # print(img)
        
        img /= 255
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        if random.random() < 0.3:
            img = img + tf.random.uniform(
                (288, 432, 4),
                minval=-0.1,
                maxval=0.1)
        # img = self.standardize(img)
        # print('here')
        # print(img.shape)
        return self.standardize(img) 
    
    def get_data(self, path):
        data_gen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                # rotation_range=0,
                # width_shift_range=0.,
                # height_shift_range=0.,
                # shear_range=0.,
                # zoom_range=0.,
                horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0.5, 1.5],
                preprocessing_function=self.preprocess_fn)
        classes_for_flow = None
        if bool(self.idx_to_class):
            classes_for_flow = self.classes
            print('classes for flow: ', classes_for_flow)
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(288, 432),
            class_mode='sparse',
            color_mode='rgba',
            batch_size=10,
            shuffle=True,
            classes=classes_for_flow)
        
        if not bool(self.idx_to_class):
            print('here')
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
            
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class
                # print(self.idx_to_class)
                # print(self.class_to_idx)
                # print(self.classes)
            print(img_class)
            print(data_gen.class_indices[img_class])
            print(data_gen.class_indices)
        return data_gen
    
    def wav_to_mel(self): #https://github.com/EsratMaria/MusicGenreRecogniton/blob/master/GenreClassificationWithCNN-LSTM.py
        dataset = []
        genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 
            'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

        for genre, genre_number in genres.items():
            for filename in os.listdir(f'../../gtzan/1.0.0/genres_original/{genre}'):
                songname = f'../../gtzan/1.0.0/genres_original/{genre}/{filename}'
                if songname.endswith('.wav'):
                    for index in range(14):
                        try: 
                            y, sr = librosa.load(songname, mono=True, duration=2, offset=index*2)
                            ps = librosa.feature.melspectrogram(y=y, sr=sr, hop_length = 256, n_fft = 512, n_mels=64)
                            ps = librosa.power_to_db(ps**2)
                            dataset.append( (ps, genre_number) )
                        except:
                            print('error')
                            print(songname)
                            sys.exit(1)
                        
        return dataset
            