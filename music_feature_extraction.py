import librosa
from librosa import onset
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
        tempo, beats = librosa.beat.beat_track(y=self.x, sr=self.sr)
        return librosa.frames_to_time(beats, sr=self.sr)


    # Vocal Separation to find where singer is singing
    # https://librosa.org/doc/main/auto_examples/plot_vocal_separation.html
    def extract_country(self):
        tempo, beats = librosa.beat.beat_track(y=self.x, sr=self.sr)
        return librosa.frames_to_time(beats, sr=self.sr)



    def extract_rock(self):
        loudness = librosa.feature.rms(y=self.x, frame_length=2048, hop_length=512, center=True, pad_mode='constant')[0]
        loudness = (loudness - np.min(loudness)) / (np.max(loudness) - np.min(loudness))
        loudness_timesteps = librosa.frames_to_time(range(len(self.x)), hop_length=512, sr=self.sr)
        features = [(t, l) for t, l in zip(loudness_timesteps, loudness) if l > 0.85*np.max(loudness)]
        return features

    def extract_classical(self):
        pass

    def extract_hiphop(self):
        # Compute mel-scaled spectrogram
        S = librosa.feature.melspectrogram(y=self.x, sr=self.sr)

        # Convert to decibels
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Extract bass regions (for example, frequencies below 100 Hz)
        bass_region = S_dB[20:200, :]

        # Compute the average energy in the bass region for each time frame
        bass_energy = np.mean(bass_region, axis=0)

        # Find the peaks (strongest bass hits)
        peaks, _ = find_peaks(bass_energy, distance=150)
        print(len(peaks))

        # Return the time positions of the strongest bass hits
        time_positions = librosa.frames_to_time(peaks, sr=self.sr)

        return time_positions

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
        return librosa.frames_to_time(beats, sr=self.sr)

    def extract_blues(self):
        pass  

if __name__ == "__main__":
    # This code block will only execute if the file is executed directly, not imported
    example = MusicFeatureExtractorModel("songs/disco/dancing_queen.wav")
    example.extract_country()
    #example.play_song()
    #example.display_waveform_segment(6, 6.01)
    #example.display_waveform()
    #example.display_spectrogram()
    #example.display_pitches()
