from matplotlib import pyplot as plt
import numpy as np
from music_feature_extraction import MusicFeatureExtractorModel as Extractor
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
import random


class VideoEditModel:
    def __init__(self, video_path, audio_path, features, output_path):
        self.width = 640
        self.height = 480

        self.video_path = video_path
        self.audio_path = audio_path
        self.features = features
        self.output_path = output_path

        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def video_edits_pop(self):
        pass

    def video_edits_disco(self):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("temp.mp4", fourcc, self.fps, (640,480))

        frame_number = 0
        current_time = 0.0
        current_beat = 0

        purple_kernel  = (164,20,217)
        orange_kernel  = (255, 128, 43)
        yellow_kernel = (249, 225, 5)
        green_kernel  = (52, 199, 165)
        violet_kernel  = (93, 80, 206)


        color_filters = [purple_kernel, orange_kernel, yellow_kernel, green_kernel, violet_kernel]

        while True:

            ret, frame = cap.read()

            if not ret or current_beat >= len(self.features):
                break

            if current_time > self.features[current_beat]:
                current_beat += 1

            kernel_index = current_beat % 5

            frame = cv2.addWeighted(frame, 0.7, np.full((frame.shape[0],frame.shape[1],3), color_filters[kernel_index], np.uint8), 0.3, 0)
            
            frame_resized = cv2.resize(frame, (self.width, self.height))

            out.write(frame_resized)
            frame_number += 1
            current_time = frame_number / self.fps

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        #Here we swap the audio
        #video  = ffmpeg.input("temp.mp4").video # get only video channel
        #audio  = ffmpeg.input(self.audio_path).audio # get only audio channel
        #output = ffmpeg.output(video, audio, self.output_path, vcodec='copy', acodec='aac', strict='experimental')
        #ffmpeg.run(output)


    def video_edits_country(self):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("temp.mp4", fourcc, self.fps, (640,480))

        frame_number = 0
        current_time = 0.0
        current_beat = 0
        time_limit = self.fps / 4
        timer = 0

        while True:
            ret, frame = cap.read()

            left_shift = np.zeros((5, 5))
            right_shift = np.zeros((5, 5))
            left_shift[2, 4] = 1
            right_shift[2, 0] = 1

            if not ret or current_beat >= len(self.features):
                break

            if current_time > self.features[current_beat]: #if we hit a feature, start a timer for fps frames
                timer = time_limit
                current_beat += 1
            
            # Convert frame to sepia
            sepia_matrix = np.array([[0.131, 0.534, 0.272],
                             [0.168, 0.686, 0.349],
                             [0.189, 0.769, 0.393]])
            sepia = cv2.transform(frame, sepia_matrix)
            
            if timer > 0:
                if random.randint(0, 1) == 0:
                    sepia = cv2.filter2D(sepia, -1, left_shift)
                else:
                    sepia = cv2.filter2D(sepia, -1, right_shift)
                # Resize the cropped frame back to original size
                frame_resized = cv2.resize(sepia, (self.width, self.height))
                out.write(frame_resized)
                timer -= 1
            else:
                frame_resized = cv2.resize(sepia, (self.width, self.height))
                out.write(frame_resized)

            frame_number += 1
            current_time = frame_number / self.fps

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        #Here we swap the audio
        video_clip = VideoFileClip("temp.mp4")
        audio_clip = AudioFileClip(self.audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(self.output_path, codec='libx264', audio_codec='aac')


    def video_edits_rock(self):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("temp.mp4", fourcc, self.fps, (640,480))

        frame_number = 0
        current_time = 0.0
        current_beat = 0
        time_limit = self.fps / 4
        timer = 0
        noise_kernel = np.random.rand(3, 3)

        while True:
            ret, frame = cap.read()

            if not ret or current_beat >= len(self.features):
                break

            if current_time > self.features[current_beat][0]: #if we hit a feature, start a timer for fps frames
                timer = time_limit
                current_beat += 1
            
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            grain_texture = cv2.imread('grain.png')
            grain_texture = cv2.resize(grain_texture, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 0.85, grain_texture, 0.15, 0.0)
            
            if timer > 0:
                frame = cv2.filter2D(frame, -1, noise_kernel)

                # Resize the cropped frame back to original size
                frame_resized = cv2.resize(frame, (self.width, self.height))
                out.write(frame_resized)
                timer -= 1
            else:
                frame_resized = cv2.resize(frame, (self.width, self.height))
                out.write(frame_resized)

            frame_number += 1
            current_time = frame_number / self.fps

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        #Here we swap the audio
        video_clip = VideoFileClip("temp.mp4")
        audio_clip = AudioFileClip(self.audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(self.output_path, codec='libx264', audio_codec='aac')

    def video_edits_classical(self):
        pass

    def video_edits_hiphop(self):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("temp.mp4", fourcc, self.fps, (640,480))

        frame_number = 0
        current_time = 0.0
        current_beat = 0
        time_limit = self.fps / 2
        timer = 0

        box_blur = np.ones((15,15)) / 225

        while True:
            ret, frame = cap.read()

            if not ret or current_beat >= len(self.features):
                break

            if current_time > self.features[current_beat]: #if we hit a feature, start a timer for fps frames
                timer = time_limit
                current_beat += 1
            
            if timer > 0: #if there is time remaining on the timer, apply box blur
                frame_blurred = cv2.filter2D(frame, -1, box_blur)
                frame_resized = cv2.resize(frame_blurred, (self.width, self.height))
                timer -= 1
                out.write(frame_resized)
            else: #if there isn't time remaining on the timer, just play the video as normal
                timer = 0
                frame_resized = cv2.resize(frame, (self.width, self.height))
                out.write(frame_resized)

            frame_number += 1
            current_time = frame_number / self.fps

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        #Here we swap the audio
        #video  = ffmpeg.input("temp.mp4").video # get only video channel
        #audio  = ffmpeg.input(self.audio_path).audio # get only audio channel
        video_clip = VideoFileClip("temp.mp4")
        audio_clip = AudioFileClip(self.audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(self.output_path, codec='libx264', audio_codec='aac')
        #output = ffmpeg.output(video, audio, self.output_path, vcodec='copy', acodec='aac', strict='experimental')
        #ffmpeg.run(output)

    def video_edits_jazz(self):
        pass
    
    def video_edits_metal(self):
        pass

    def video_edits_reggae(self):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("temp.mp4", fourcc, self.fps, (640, 480))

        frame_number = 0
        rotate_angle = 0  # Initial angle for rotation effect
        rotation_direction = 1
        current_time = 0
        current_beat = 0
        rotate_speed = 1  # Speed of rotation increase per frame

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if current_time > self.features[current_beat]:
                current_beat += 1
                rotation_direction *= -1

            # Apply rotation
            if current_beat != 0:
                rotation_matrix = cv2.getRotationMatrix2D((frame.shape[1] // 2, frame.shape[0] // 2), rotate_angle, 1)
                frame_rotated = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
                rotate_angle += rotate_speed * rotation_direction
            else:
                frame_rotated = frame
                

            # Resize the rotated frame
            frame_resized = cv2.resize(frame_rotated, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            out.write(frame_resized)

            frame_number += 1
            current_time = frame_number / self.fps

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        #Here we swap the audio
        #video  = ffmpeg.input("temp.mp4").video # get only video channel
        #audio  = ffmpeg.input(self.audio_path).audio # get only audio channel
        #output = ffmpeg.output(video, audio, self.output_path, vcodec='copy', acodec='aac', strict='experimental')
        #ffmpeg.run(output)
        video_clip = VideoFileClip("temp.mp4")
        audio_clip = AudioFileClip(self.audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(self.output_path, codec='libx264', audio_codec='aac')

    def video_edits_blues(self):
        pass  


if __name__ == "__main__":
    # This code block will only execute if the file is executed directly, not imported
    
    music_extractor = Extractor("songs/rock/in_bloom.wav")

    rock_features = music_extractor.extract_rock()

    video_editor = VideoEditModel("video/dance.mp4", "songs/rock/in_bloom.wav", rock_features, 'test.mp4')

    video_editor.video_edits_rock()