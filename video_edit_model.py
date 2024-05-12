from matplotlib import pyplot as plt
import numpy as np
from music_feature_extraction import MusicFeatureExtractorModel as Extractor
import cv2

class VideoEditModel:
    def __init__(self, video_path, features):
        self.width = 640
        self.height = 480

        self.video_path = video_path
        self.features = features

        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def play(self, song_path):
       self.video_edits_disco()
        
    def video_edits_pop(self):
        pass

    def video_edits_disco(self):
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('test.mp4', fourcc, self.fps, (640,480))

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

            if not ret or current_beat >= len(self.features) or current_beat > 5:
                break

            if current_time > self.features[current_beat]:
                current_beat += 1

            kernel_index = current_beat % 5

            frame = cv2.addWeighted(frame, 0.8, np.full((frame.shape[0],frame.shape[1],3), color_filters[kernel_index], np.uint8), 0.2, 0)
            
            frame_resized = cv2.resize(frame, (self.width, self.height))

            out.write(frame_resized)
            frame_number += 1
            current_time = frame_number / self.fps
            print(current_time)

        cap.release()
        out.release()
        cv2.destroyAllWindows()


    def video_edits_country(self):
        pass

    def video_edits_rock(self):
        pass

    def video_edits_classical(self):
        pass

    def video_edits_hiphop(self):
        pass

    def video_edits_jazz(self):
        pass
    
    def video_edits_metal(self):
        pass

    def video_edits_reggae(self):
        pass

    def video_edits_blues(self):
        pass  


if __name__ == "__main__":
    # This code block will only execute if the file is executed directly, not imported
    
    music_extractor = Extractor("songs/disco/dancing_queen.wav")

    disco_features = music_extractor.extract_disco()

    example = VideoEditModel("video/dance.mp4", disco_features)

    example.play("songs/disco/dancing_queen.wav")
    
