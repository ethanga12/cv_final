import librosa
import numpy as np
import pyaudio  
import wave 
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import find_peaks

class MusicFeatureExtractorModel:
    def __init__(self, song_path):
        self.audio_path = song_path
        self.x , self.sr = librosa.load(song_path, sr=None)
    
    def play_song(self):
         
        chunk = 1024  
        
        f = wave.open(self.audio_path,"rb")
       
        p = pyaudio.PyAudio()  
        
        stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
         
        data = f.readframes(chunk)  
  
        #play stream  
        while data:  
            stream.write(data)  
            data = f.readframes(chunk)  
  
        #stop stream  
        stream.stop_stream()  
        stream.close()  
  
        #close PyAudio  
        p.terminate() 

    def display_raw_samples(self):
        #Zooming in on a plot to show raw sample values
        #Documentation: https://librosa.org/doc/latest/generated/librosa.display.waveshow.html#librosa.display.waveshow
        fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
        ax.set(xlim=[6.0, 6.01], title='Sample view', ylim=[-1, 1])
        y_harm, y_perc = librosa.effects.hpss(self.x)
        librosa.display.waveshow(self.x, sr=self.sr, ax=ax, marker='.', label='Full signal')
        librosa.display.waveshow(y_harm, sr=self.sr, alpha=0.5, ax=ax2, label='Harmonic')
        librosa.display.waveshow(y_perc, sr=self.sr, color='r', alpha=0.5, ax=ax2, label='Percussive')
        ax.label_outer()
        ax.legend()
        ax2.legend()
        plt.show()

    def display_waveform(self):
        #Or harmonic and percussive components with transparency
        #Documentation: https://librosa.org/doc/latest/generated/librosa.display.waveshow.html#librosa.display.waveshow
        plt.figure(figsize=(14, 5))
        librosa.display.waveshow(self.x, sr=self.sr)
        y_harm, y_perc = librosa.effects.hpss(self.x)
        librosa.display.waveshow(y_harm, sr=self.sr, alpha=0.5, label='Harmonic')
        librosa.display.waveshow(y_perc, sr=self.sr, color='r', alpha=0.5, label='Percussive')
        plt.title('Multiple waveforms')
        plt.legend()
        plt.show()

    def display_pitches(self):
        #visualize the music note content.
        #Documentation: https://medium.com/@swilliam.productions/music-extraction-7eb352d92bff 
        chroma = librosa.feature.chroma_stft(y=self.x)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax)
        ax.set(title='Chromagram of music notes')
        fig.colorbar(img, ax=ax)
        plt.show()

    def display_spectrogram(self):
        #Documentation: https://librosa.org/doc/latest/generated/librosa.display.specshow.html#librosa.display.specshow

        #Visualize an STFT power spectrum using default parameters
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.x)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=self.sr, ax=ax[0])
        ax[0].set(title='Linear-frequency power spectrogram')
        ax[0].label_outer()

        #On a logarithmic scale, and using a larger hop
        hop_length = 1024
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.x, hop_length=hop_length)),
                            ref=np.max)
        librosa.display.specshow(D, y_axis='log', sr=self.sr, hop_length=hop_length,
                         x_axis='time', ax=ax[1])
        ax[1].set(title='Log-frequency power spectrogram')
        ax[1].label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.show()

    def extract_pop(self):
        pass

    def extract_disco(self):
        pass

    # Vocal Separation to find where singer is singing
    # https://librosa.org/doc/main/auto_examples/plot_vocal_separation.html
    def extract_country(self):
        '''
        S_full, _ = librosa.magphase(librosa.stft(self.x))
        S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=self.sr)))
        S_filter = np.minimum(S_full, S_filter)
        margin_v = 10
        power = 2

        mask_v = librosa.util.softmask(S_full - S_filter,
                                    margin_v * S_filter,
                                    power=power)
        S_foreground = mask_v * S_full

        # Calculate the sum of magnitudes across frequency bins for each time frame
        energy_per_frame = np.sum(S_foreground, axis=0)
        
        # Threshold to determine when the singer is singing
        energy_threshold = 0.5 * np.max(energy_per_frame)  # Adjust the threshold as needed
        
        # Find the frames where energy exceeds the threshold
        singing_frames = np.where(energy_per_frame > energy_threshold)[0]
        
        # Convert frames to timestamps
        timestamps = librosa.frames_to_time(singing_frames, sr=self.sr)
        
        print("Timestamps where the singer is singing (seconds):", timestamps)
        print(len(timestamps))
        '''



    def extract_rock(self):
        #Check for peaks using scipy. Can play with distance but this one gives us a "peak" every few seconds
        peaks, _ = find_peaks(self.x, distance=200000)
        return {peak_index: "hit" for peak_index in peaks}

    def extract_classical(self):
        pass

    def extract_hiphop(self):
        # Separate the bass component from the audio
        y_harm, _ = librosa.effects.hpss(self.x)
        
        # Calculate the onset strength envelope for the bass component
        onset_env = librosa.onset.onset_strength(y=y_harm, sr=self.sr)
        
        # Set a threshold to identify significant bass events
        threshold = 0.5 * np.max(onset_env)  # Adjust threshold as needed
        
        # Find frames where the onset strength exceeds the threshold
        bass_event_frames = np.where(onset_env > threshold)[0]
        
        # Convert frames to timestamps
        bass_event_timestamps = librosa.frames_to_time(bass_event_frames, sr=self.sr)
        
        print("Bass event timestamps (seconds):", bass_event_timestamps)

    def extract_jazz(self):
        pass
    
    def extract_metal(self):
        pass

    def extract_reggae(self):
        #extract the percussion
        _, y_perc = librosa.effects.hpss(self.x)
        # Calculate the onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y_perc, sr=self.sr)
        
        # Find the beat locations using the onset strength envelope
        _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        
        # Convert beat frames to timestamps
        beat_timestamps = librosa.frames_to_time(beats, sr=self.sr)
        
        return beat_timestamps

    def extract_blues(self):
        pass  

if __name__ == "__main__":
    # This code block will only execute if the file is executed directly, not imported
    example = MusicFeatureExtractorModel("songs/disco/dancing_queen.wav")
    example.extract_country()
    #example.play_song()
    #example.display_waveform()
    #example.display_spectrogram()
    #example.display_pitches()
