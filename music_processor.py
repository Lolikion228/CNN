import librosa.filters
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from glob import glob
from scipy import signal
from scipy.fftpack import fft
from librosa.filters import mel
from librosa.display import specshow
from librosa import stft
from librosa.effects import pitch_shift
import librosa.feature
import pickle
import sys
from numba import jit, prange
from sklearn.preprocessing import normalize

#Do some vizualization for all
#do some validation
#Do some UI
#do some automatization
#do general list of hyperparameters
class Audio:
    """
    audio class which holds music data and timestamp for notes.

    Args:
        filename: file name.
        stereo: True or False; wether you have Don/Ka streo file or not. normaly True.
    Variables:


    Example:
        # >>>from music_processor import *
        # >>>song = Audio(filename)
        # >>># to get audio data
        # >>>song.data
        # >>># to import .tja files:
        # >>>song.import_tja(filename)
        # >>># to get data converted
        # >>>song.data = (song.data[:,0]+song.data[:,1])/2
        # >>>fft_and_melscale(song, include_zero_cross=False)
    """

    def __init__(self, filename, stereo=True):

        self.data, self.samplerate = sf.read(filename, always_2d=True)



        if stereo is False:
            self.data = (self.data[:, 0]+self.data[:, 1])/2
        self.timestamp = []


    # def plotaudio(self, start_t, stop_t):
    #     plt.plot(np.linspace(start_t, stop_t, stop_t-start_t), self.data[start_t:stop_t, 0])
    #     plt.show()


    def save(self, filename="./savedmusic.wav", start_t=0, stop_t=None):

        if stop_t is None:
            stop_t = self.data.shape[0]
        sf.write(filename, self.data[start_t:stop_t], self.samplerate)


    def import_osu(self,filename):
        input = open(filename)
        flag = 0
        self.timestamp = []
        for x in input:
            y = x[:-1]
            if y == "[HitObjects]":
                flag = 1
            if flag and y != "[HitObjects]":
                res = x[:-1].split(',')
                time = int(res[2]) / 1000
                ##проверить что это нота
                if int(res[3]) == 1:
                    if int(res[4])==0 or int(res[4])==4:
                        self.timestamp.append([time,1])
                    else:
                        self.timestamp.append([time, 2])
        self.timestamp=np.array(self.timestamp)
        input.close()



    def import_tja(self, filename, verbose=False, diff=False, difficulty=None):
        """imports tja file and convert it into timestamp"""
        #CHANGE TO OSU! FILE FORMAT
        # SELF.TIMESTAMP=[ [time1,type1], [time2,type2], .... [timeN,typeN]   ] time(seconds) type from {1,2}
        now = 0.0
        bpm = 100
        measure = [4, 4]  # hyousi
        self.timestamp = []
        skipflag = False

        with open(filename, "rb") as f:
            while True:
                line = f.readline()
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')
                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0:5] == "TITLE":
                    if verbose:
                        print("importing: ", line[6:])
                elif line[0:6] == "OFFSET":
                    now = -float(line[7:-2])
                elif line[0:4] == "BPM:":
                    bpm = float(line[4:-2])
                if line[0:6] == "COURSE":
                    if difficulty and difficulty > 0:
                        skipflag = True
                        difficulty -= 1
                elif line == "#START\r\n":
                    if skipflag:
                        skipflag = False
                        continue
                    break
            
            sound = []
            while True:
                line = f.readline()
                # print(line)
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')

                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0] <= '9' and line[0] >= '0':
                    if line.find(',') != -1:
                        sound += line[0:line.find(',')]
                        beat = len(sound)
                        for i in range(beat):
                            if diff:
                                if int(sound[i]) in (1, 3, 5, 6, 7):
                                    self.timestamp.append([int(100*(now+i*60*measure[0]/bpm/beat))/100, 1])
                                elif int(sound[i]) in (2, 4):
                                    self.timestamp.append([int(100*(now+i*60*measure[0]/bpm/beat))/100, 2])
                            else:
                                if int(sound[i]) != 0:
                                    self.timestamp.append([int(100*(now+i*60*measure[0]/bpm/beat))/100, int(sound[i])])
                        now += 60/bpm*measure[0]
                        sound = []
                    else:
                        sound += line[0:-2]
                elif line[0] == ',':
                    now += 60/bpm*measure[0]
                elif line[0:10] == '#BPMCHANGE':
                    bpm = float(line[11:-2])
                elif line[0:8] == '#MEASURE':
                    measure[0] = int(line[line.find('/')-1])
                    measure[1] = int(line[line.find('/')+1])
                elif line[0:6] == '#DELAY':
                    now += float(line[7:-2])
                elif line[0:4] == "#END":
                    if(verbose):
                        print("import complete!")
                    self.timestamp = np.array(self.timestamp)
                    break


    def synthesize(self, diff=True, don="./data/don.wav", ka="./data/ka.wav"):
        
        donsound = sf.read(don)[0]
        donsound = (donsound[:, 0] + donsound[:, 1]) / 2
        kasound = sf.read(ka)[0]
        kasound = (kasound[:, 0] + kasound[:, 1]) / 2
        donlen = len(donsound)
        kalen = len(kasound)
        
        if diff is True:
            for stamp in self.timestamp:
                timing = int(stamp[0]*self.samplerate)
                try:
                    if stamp[1] in (0,4):
                        self.data[timing:timing+donlen] += donsound
                    elif stamp[1] in (2, 8):
                        self.data[timing:timing+kalen] += kasound
                except ValueError:
                    pass

        elif diff == 'don':

            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp*self.samplerate+donlen < self.data.shape[0]:
                        self.data[int(stamp[0]*self.samplerate):int(stamp[0]*self.samplerate)+donlen] += donsound
            else:
                for stamp in self.timestamp:
                    if stamp*self.samplerate+donlen < self.data.shape[0]:
                        self.data[int(stamp*self.samplerate):int(stamp*self.samplerate)+donlen] += donsound

        
        elif diff == 'ka':

            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp*self.samplerate+kalen < self.data.shape[0]:
                        self.data[int(stamp[0]*self.samplerate):int(stamp[0]*self.samplerate)+kalen] += kasound
            else:
                for stamp in self.timestamp:
                    if stamp*self.samplerate+kalen < self.data.shape[0]:
                        self.data[int(stamp*self.samplerate):int(stamp*self.samplerate)+kalen] += kasound


    def plotaudio(self, start_t=0, stop_t=None):

        if stop_t is None:
            stop_t = self.data.shape[0]/self.samplerate
        stop_t=int(self.samplerate*stop_t)
        start_t=int(self.samplerate*start_t)
        if stop_t>=len(self.data):
            stop_t=len(self.data)-1

        if (self.data.shape[1] == 1):
            plt.plot(np.linspace(start_t, stop_t, (stop_t - start_t))/self.samplerate, (self.data[start_t:stop_t, 0]+self.data[start_t:stop_t, 0])/2)
        else:
            plt.plot(np.linspace(start_t, stop_t, (stop_t - start_t)) / self.samplerate,
                     (self.data[start_t:stop_t, 0] + self.data[start_t:stop_t, 1]) / 2)
        plt.show()
        # return plt

    def specshow(self, nhop=512, nfft=1024,start_t=0,stop_t=None):
        if stop_t is None:
            stop_t = self.data.shape[0]/self.samplerate
        stop_t=int(self.samplerate*stop_t)
        start_t=int(self.samplerate*start_t)
        if stop_t>=len(self.data):
            stop_t=len(self.data)-1
        if (self.data.shape[1] == 1):
            stft_out = librosa.stft( (self.data[start_t:stop_t,0]+self.data[start_t:stop_t,0])/2 ,n_fft=nfft)
        else:
            stft_out = librosa.stft((self.data[start_t:stop_t, 0] + self.data[start_t:stop_t, 1]) / 2, n_fft=nfft)
        stft_out = librosa.amplitude_to_db(abs(stft_out))
        plt.figure(figsize=(14, 5))

        librosa.display.specshow(stft_out, sr=self.samplerate, x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.show()


    def mel_specshow(self, nhop=512, nfft=1024,start_t=0,stop_t=None,n_mels=80):
        if stop_t is None:
            stop_t = self.data.shape[0]/self.samplerate
        stop_t=int(self.samplerate*stop_t)
        start_t=int(self.samplerate*start_t)
        if stop_t>=len(self.data):
            stop_t=len(self.data)-1
        if (self.data.shape[1] == 1):
            sig=(self.data[start_t:stop_t,0]+self.data[start_t:stop_t,0])/2
        else:
            sig = (self.data[start_t:stop_t, 0] + self.data[start_t:stop_t, 1]) / 2
        fs=self.samplerate
        S = librosa.feature.melspectrogram(y=sig, sr=fs,n_fft=nfft,hop_length=nhop,n_mels=n_mels,fmin=27.5,fmax=16000.0)
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel',x_axis='time',fmin=27.5,fmax=16000)
        plt.colorbar()
        plt.show()


def make_frame(data, nhop, nfft):
    """
    helping function for fftandmelscale.
    細かい時間に切り分けたものを学習データとするため，nhop(512)ずつずらしながらnfftサイズのデータを配列として返す
    """
    
    length = data.shape[0]
    # print("frame00=", length)
    framedata = np.concatenate((data, np.zeros(nfft)))  # zero padding
    # print("frame0=", framedata.shape)
    return np.array([framedata[i*nhop:i*nhop+nfft] for i in range(length//nhop)])  



def fft_and_melscale(song, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    """
    fft and melscale method.
    fft: nfft = [1024, 2048, 4096]; サンプルの切り取る長さを変えながらデータからnp.arrayを抽出して高速フーリエ変換を行う．
    melscale: 周波数の次元を削減するとともに，log10の値を取っている．
    """

    feat_channels = []
    
    for nfft in nffts:
        
        feats=[]
        window = signal.windows.blackmanharris(nfft)
        filt = librosa.filters.mel(sr=song.samplerate, n_fft=nfft, n_mels=mel_nband, fmin=mel_freqlo, fmax=mel_freqhi)
        # print("sample_rate=",song.samplerate)
        # print("filt=",filt.shape)

        # get normal frame
        frame = make_frame(song.data, nhop, nfft)
        # print("song_data=",song.data.shape)
        # print("frame=",frame.shape)

        # melscaling
        # print("window =",window.shape)
        processedframe = fft(window*frame)[:, :nfft//2+1]
        # print("processed frame=",processedframe.shape)
        # print("abs=",np.transpose(np.abs(processedframe)**2).shape)#избавиться от комплексной части
        processedframe = np.dot(filt, np.transpose(np.abs(processedframe)**2))#умножение матриц
        # print("processed frame2=", processedframe.shape)
        processedframe = 20*np.log10(processedframe+0.1)#CONVERT TO DBs
        # print("processed frame3=", processedframe.shape)
        # print(processedframe.shape)
        # print("-" * 30)
        feat_channels.append(processedframe)
    if include_zero_cross:
        song.zero_crossing = np.where(np.diff(np.sign(song.data)))[0]
        # print(song.zero_crossing)

    return np.array(feat_channels)



def multi_fft_and_melscale(songs, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    
    for i in prange(len(songs)):
        songs[i].feats = fft_and_melscale(songs[i], nhop, nffts, mel_nband, mel_freqlo, mel_freqhi)



def milden(data):
    """put smaller value(0.25) to plus minus 1 frame."""
    
    for i in range(data.shape[0]):
        
        if data[i] == 1:
            if i > 0:
                data[i-1] = 0.25
            if i < data.shape[0] - 1:
                data[i+1] = 0.25
        
        if data[i] == 0.26:
            if i > 0:
                data[i-1] = 0.1
            if i < data.shape[0] - 1:
                data[i+1] = 0.1
    
    return data


def smooth(x, window_len=11, window='hanning'):
    
    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    
    return y


def music_for_listening(serv, synthesize=True, difficulty=0):

    song = Audio(glob(serv+"/audio.*")[0])
    if synthesize:
        # song.import_tja(glob(serv+"/*.tja")[-1], difficulty=difficulty)
        song.import_osu(glob(serv+"/*.osu")[-1])
        song.synthesize()
    # plt.plot(song.data[1000:1512, 0])
    # plt.show()
    song.save("./data/saved_music.wav")


def music_for_validation(serv, deletemusic=True, verbose=False, difficulty=1):

    song = Audio(glob(serv+"/audio.*")[0], stereo=False)
    # song.import_tja(glob(serv+"/*.tja")[-1], difficulty=difficulty)
    song.import_osu(glob(serv+"/*.osu")[-1])
    song.feats = fft_and_melscale(song, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False)

    if deletemusic:
        song.data = None
    with open('./data/pickles/val_data.pickle', mode='wb') as f:
        pickle.dump(song, f)


def music_for_train(serv, deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    
    songplaces = glob(serv)
    songs = []
    
    for songplace in songplaces:
        
        if verbose:
            print(songplace)
        
        song = Audio(glob(songplace+"/audio.*")[0])
        # song.import_tja(glob(songplace+"/*.tja")[-1], difficulty=difficulty, diff=True)
        song.import_osu(glob(songplace + "/*.osu")[-1])
        song.data = (song.data[:, 0]+song.data[:, 1])/2
        songs.append(song)

    multi_fft_and_melscale(songs, nhop, nffts, mel_nband, mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)
    if deletemusic:
        for song in songs:
            song.data = None
    
    with open('./data/pickles/train_data.pickle', mode='wb') as f:
        pickle.dump(songs, f)

def music_for_train_reduced(serv, deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    
    songplaces = glob(serv)
    songs = []
    
    for songplace in songplaces:
        print(glob(songplace+"/audio.*"))
        song = Audio(glob(songplace+"/audio.*")[0])

        # song.import_tja(glob(songplace+"/*.tja")[-1], difficulty=difficulty, diff=True)
        song.import_osu(glob(songplace+"/*.osu")[-1],)

        song.data = (song.data[:, 0]+song.data[:, 1])/2
        if len(song.timestamp) != 0:
            songs.append(song)
       
       
        # print("timestamp=",song.timestamp[-100:],len(song.timestamp[0]))
    
    
    multi_fft_and_melscale(songs, nhop, nffts, mel_nband, mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)
  
    if deletemusic:
        for song in songs:
            song.data = None
    
    with open('./data/pickles/train_reduced.pickle', mode='wb') as f:
        pickle.dump(songs, f)
    
   



def music_for_test(serv, deletemusic=True, verbose=False):

    song = Audio(glob(serv+"/audio.*")[0], stereo=False)
    # song.import_tja(glob(serv+"/*.tja")[-1])
    song.feats = fft_and_melscale(song, include_zero_cross=False)
    with open('./data/pickles/test_data.pickle', mode='wb') as f:
        pickle.dump(song, f)


if __name__ == "__main__":

    if sys.argv[1] == 'train':
        print("preparing all train data processing...")
        serv = "./data/train/*"
        music_for_train(serv, verbose=True, difficulty=0, diff=True)
        print("all train data processing done!")

    if sys.argv[1] == 'test':
        print("test data proccesing...")
        serv = "./data/test/"
        music_for_test(serv)
        print("test data processing done!")

    if sys.argv[1] == 'val':
        print("validation data processing...")
        serv = "./data/validation"
        music_for_validation(serv)
        print("done!")

    if sys.argv[1] == 'reduced':
        serv = './data/train_reduced/*'
        music_for_train_reduced(serv, verbose=True, difficulty=0, diff=True)


# songplace = './data/train_reduced/*'
# print(glob(songplace))
# song = Audio(glob(songplace+"/*.ogg")[0])
# song.import_tja(glob(songplace+"/*.tja")[-1], difficulty=0, diff=True)
# song.data = (song.data[:, 0]+song.data[:, 1])/2
# nhop=512
# nffts=[1024, 2048, 4096]
# mel_nband=80
# mel_freqlo=27.5
# mel_freqhi=16000.0
# song.feats = fft_and_melscale(song, nhop, nffts, mel_nband, mel_freqlo, mel_freqhi)
# # print(song.timestamp[:50])
# res=song.import_osu('./data/ex1.osu')
# print(res[:50])

""""
Audio.data Audio.samplerate = from librosa 
Audio.timestamp = [ [time1,type_note1], [time2,type_note2], .... [timeN,type_noteN]   ] time(seconds) type_note from {1,2}
Audio.feats(features) =  [ [ [], [],....[]], [--//--],  [--//--]  ]   [spectrogram_number][mel_band][time_bin]  for each spectrogram nffts=[1024, 2048, 4096]
"""
# print(vars(song).keys())
# print(song.feats.shape)
# print(song.feats[0][10][100:200])
