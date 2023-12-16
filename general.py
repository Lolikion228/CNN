



from music_processor import *


# s=161.5
# e=164
song=Audio('./data/test/audio.mp3')
# song.plotaudio(start_t=s,stop_t=e)
# song.specshow(start_t=s,stop_t=e)
# song.mel_specshow(start_t=s,stop_t=e)


pth='/home/lolikion/Документы/study/НИР/features/ex'
ind=5
t=[ [125,130], [232,237] ] + [ [x,x+5] for x in range(247,250,5)]
for x in t:
    s,e=x

    ind+=1
