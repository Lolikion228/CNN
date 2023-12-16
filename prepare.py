import pathlib
import os
import shutil


main_dir='./data/train_reduced/'

i=1
for x in os.listdir(main_dir):
    pth=main_dir+x
    files=os.listdir(pth)
    if ("audio.ogg" not in files) and ("audio.mp3" not in files) and ("audio.wav" not in files):
        shutil.rmtree(pth)
        continue
    lvls=[]
    for file in files:
        
        if file[-4:]==".osu":
            f=open(pth+"/"+file,"r")
            for line in f:
                if "OverallDifficulty:" in line:
                    x=str(line[(line.index(":"))+1:].strip())
                    lvls.append([file,float(x)])
                    
                    break
            f.close()
    
    
    
    for x in lvls:
        if x[1]>min(y[1] for y in lvls):
            os.remove(pth+"/"+x[0])
    if lvls:
        if lvls[0][1]>5:
            shutil.rmtree(pth)
            continue
    
   
    flag=0
    for file in files:
        if file[-4:]==".osu": flag=1
    if flag==0:
        shutil.rmtree(pth)
        continue
    i+=1


