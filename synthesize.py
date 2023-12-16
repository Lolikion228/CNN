import pickle

import numpy
import numpy as np
from scipy.signal import argrelmax
from librosa.util import peak_pick
from librosa.onset import onset_detect
from music_processor import *
from random import randint as rnd

def detection(don_inference, ka_inference, song,who=0):
    """detects notes disnotesiresultg don and ka"""
    #who1=[0,1,2] =[ka,don,both]
    print(don_inference.shape)
    don_inference = smooth(don_inference, 5)
    ka_inference = smooth(ka_inference, 5)

    don_timestamp = (peak_pick(x=don_inference, pre_max=1, post_max=2, pre_avg=4, post_avg=5, delta=0.05, wait=3)+7)  # 実際は7フレーム目のところの音
    ka_timestamp = (peak_pick(x=ka_inference, pre_max=1, post_max=2, pre_avg=4, post_avg=5, delta=0.05, wait=3)+7)



    song.don_timestamp = don_timestamp[np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])]

    song.timestamp = song.don_timestamp*512/song.samplerate


    if who == 1:
    	song.synthesize(diff='don')
    if who == 0:
    	song.synthesize(diff='ka')
    if who == 2:
    	song.synthesize(diff='ka')

    song.ka_timestamp = ka_timestamp[np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])]
    song.timestamp=song.ka_timestamp*512/song.samplerate


    if who == 1:
    	song.synthesize(diff='don')
    if who == 0:
    	song.synthesize(diff='ka')
    if who == 2:
    	song.synthesize(diff='don')
    song.save("./data/inference/audio.mp3")

def create_osu(filename, song, file2,don_timestamp, ka_timestamp=None):
    ##REdO FOR OSU
    f=open(file2,'r')
    lines=[]
    for x in f:
        if x!="[HitObjects]\n":
            lines.append(x)
        else:
            break
    f.close()

    if ka_timestamp is None:
        timestamp=don_timestamp*512/song.samplerate
        with open(filename, "w") as f:
            for x in lines:
                f.write(x)
            f.write("[HitObjects]")
            i = 0
            nums = [2, 0]
            while(i < len(timestamp)):
                tm=int(timestamp[i]*1000)
                f.write("\n256,192,"+str(tm)+",1,"+str(nums[rnd(0,1)])+",0:0:0:0:")
                i += 1


    else:
        f = open(file2, 'r')
        lines = []
        for x in f:
            if x != "[HitObjects]\n":
                lines.append(x)
            else:
                break
        f.close()
        don_timestamp=np.rint(don_timestamp*512/song.samplerate*100).astype(np.int32)
        ka_timestamp=np.rint(ka_timestamp*512/song.samplerate*100).astype(np.int32)
        with open("./data/inference"+file2[file2.rindex("/"):], "w") as f:
            for x in lines:
                f.write(x)
            f.write("[HitObjects]")
            i = 0
            nts=[]

            while (i <len(ka_timestamp)):
                tm1 = int(ka_timestamp[i] * 10)
                nts.append([tm1,2])
                i += 1
            i=0

            while (i <  len(don_timestamp)):
                tm2 = int(don_timestamp[i] * 10)
                nts.append([tm2, 0])
                i += 1

            nts.sort()
            for x in nts:
                f.write("\n256,192," + str(x[0]) + ",1,"+str(x[1])+",0:0:0:0:")



def create_tja(filename, song, don_timestamp, ka_timestamp=None):
    ##REdO FOR OSU
    if ka_timestamp is None:
        timestamp=don_timestamp*512/song.samplerate
        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            i = 0
            time = 0
            while(i < len(timestamp)):
                if time/100 >= timestamp[i]:
                    f.write('1')
                    i += 1
                else:
                    f.write('0')
                if time % 100 == 99:
                    f.write(',\n')
                time += 1
            f.write('#END')

    else:
        don_timestamp=np.rint(don_timestamp*512/song.samplerate*100).astype(np.int32)
        ka_timestamp=np.rint(ka_timestamp*512/song.samplerate*100).astype(np.int32)
        # print("res=",ka_timestamp[-100:]/100) #secs*100
        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            for time in range(np.max((don_timestamp[-1],ka_timestamp[-1]))):
                if np.isin(time,don_timestamp) == True:
                    f.write('1')
                elif np.isin(time,ka_timestamp) == True:
                    f.write('2')
                else:
                    f.write('0')
                if time%100==99:
                    f.write(',\n')
            f.write('#END')



def plot_infer(song,start_t=0,stop_t=None):
    sig1=song.ka_timestamp*512
    sig2=song.don_timestamp*512
    start_t=int(start_t*song.samplerate)
    stop_t=int(stop_t*song.samplerate)
    if stop_t>=len(song.data):
        stop_t=len(song.data)-1
    song.data=numpy.array(song.data)

    sig=[0 for i in range(len(song.data))]
    sig=numpy.array(sig)
    for x in sig1:
        sig[x]=2

    sig4 = [0 for i in range(len(song.data))]
    sig4 = numpy.array(sig4)
    for x in sig2:
        sig4[x] = 2

    # print(sig1.shape,song.data.shape)
    if( len(song.data.shape)==1 or song.data.shape[1]==1):
        plt.plot(np.linspace(start_t, stop_t, stop_t - start_t)/song.samplerate, song.data[start_t:stop_t])
    else:
        plt.plot(np.linspace(start_t, stop_t, stop_t - start_t) / song.samplerate, (song.data[start_t:stop_t,0]+song.data[start_t:stop_t,1])/2)
    plt.plot(np.linspace(start_t, stop_t, stop_t - start_t)/song.samplerate, sig[start_t:stop_t])
    plt.plot(np.linspace(start_t, stop_t, stop_t - start_t)/song.samplerate, sig4[start_t:stop_t])
    plt.show()
    # return plt

if __name__ == "__main__":
    
    with open('./data/pickles/test_data.pickle', mode='rb') as f:
        song = pickle.load(f)

    with open('./data/pickles/don_inference.pickle', mode='rb') as f:
        don_inference = pickle.load(f)

    with open('./data/pickles/ka_inference.pickle', mode='rb') as f:
        ka_inference = pickle.load(f)
#161.5 164
#5 10
    detection(don_inference, ka_inference, song)
    
  
    #plot_infer(song, start_t=s, stop_t=e)
       

    create_osu("./data/inference/inferred_notes.osu", song,glob("./data/test/*.osu")[-1], song.don_timestamp,song.ka_timestamp)

