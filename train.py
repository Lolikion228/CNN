from model import *
from music_processor import *


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = convNet()
    net = net.to(device)

    try:
        with open('./data/pickles/train_data.pickle', mode='rb') as f:
            songs = pickle.load(f)
    except FileNotFoundError:
        with open('./data/pickles/train_reduced.pickle', mode='rb') as f:
            songs = pickle.load(f)


    minibatch = 128
    soundlen = 15
    epoch = 300

    if sys.argv[1] == 'don':
        net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=epoch, device=device, soundlen=soundlen, save_place='./models/don_model.pth', log='./data/log/don.txt', don_ka=1)
    
    if sys.argv[1] == 'ka':
        net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=epoch, device=device, soundlen=soundlen, save_place='./models/ka_model.pth', log='./data/log/ka.txt', don_ka=2)
