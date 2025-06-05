import os
## if your data is .mp4 form, please use RepCountA_raw_Loader.py
# from dataset.RepCountA_raw_Loader import MyData
## if your data is .npz form, please use RepCountA_Loader.py. It can speed up the training
from models.TransRAC import TransferModel
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def listdir(path, list_name, thresh=0.5):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif float(file_path.split('/')[-1].split('_')[1][:-3]) < thresh:
            list_name.append(file_path)

def get_frames(npz_path):
    # get frames from .npz files
    with np.load(npz_path, allow_pickle=True) as data:
        frames = data['imgs']  
        frames_g = data['imgs_g']
        frames_s = data['imgs_s']
        label = data['label']
        label_g = data['label_g']
        name = data['name']
        label =  torch.tensor(label)
        frames = torch.FloatTensor(frames)
        frames -= 127.5
        frames /= 127.5
        label_g = torch.tensor(label_g)
        frames_g = torch.FloatTensor(frames_g)
        frames_g -= 127.5
        frames_g /= 127.5
        frames_s = torch.FloatTensor(frames_s)
        frames_s -= 127.5
        frames_s /= 127.5
    return frames, frames_g, frames_s, label, label_g, name

def data_reorgan(datalist):
    result = []
    name = []
    datalist.sort()
    for data in datalist:
        name.append(data[0:-7])
    name = list(set(name))
    name.sort()
    for n in name:
        res = []
        for data in datalist:
            if data[0:-7] == n:
                res.append(data)
        result.append(res)
    return result

def test_loop(n_epochs, model, root_path, thresh, inference=True, batch_size=1, lastckpt=None, paint=False, device_ids=[0]):

    ckptlist = []
    listdir(lastckpt, ckptlist, thresh)
    datalist = os.listdir(root_path)
    datalist = data_reorgan(datalist)
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    # testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=10)
    model = nn.DataParallel(model.to(device), device_ids=device_ids)


    for ckpt in tqdm(ckptlist):

        print("loader ckpt: " + ckpt)
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint

        testOBO = []
        testMAE = []
        predCount = []
        Count = []
        if inference:
            with torch.no_grad():
                batch_idx = 0
                acc = 0
                for data in datalist:
                    countnum = 0
                    labelnum = 0
                    for d in data:
                        frames, frames_g, frames_s, label, label_g, name = get_frames(root_path + d)  # [64, 3, 224, 224]
                        # import pdb;pdb.set_trace()
                        frames = frames.transpose(0, 1)  # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
                        frames_g = frames_g.transpose(0, 1)
                        frames_s = frames_s.transpose(0, 1)
                        model.eval()
                        frames = frames.type(torch.FloatTensor).unsqueeze(0).to(device)
                        frames_g = frames_g.type(torch.FloatTensor).unsqueeze(0).to(device)
                        frames_s = frames_s.type(torch.FloatTensor).unsqueeze(0).to(device)

                        label = label.type(torch.FloatTensor).unsqueeze(0).to(device)
                        label_g = label_g.type(torch.FloatTensor).unsqueeze(0).to(device)
                        count = torch.sum(label, dim=1).type(torch.FloatTensor).to(device)
                        # count_g = torch.sum(label_g, dim=1).type(torch.FloatTensor).to(device)
                        # start = time.time()
                        output, output_g = model(frames, frames_g, frames_s)
                        # density_map(maps=output_g, index=0, cmp='YlGnBu', file_name=d.split('.')[0])
                        # density_map(maps=label_g, index=1, cmp='crest', file_name=d.split('.')[0])
                        # print(time.time()-start)
                        predict_count = torch.sum(output, dim=1)
                        # predict_count_g = torch.sum(output_g, dim=1)
                        countnum += predict_count.item()
                        labelnum += count.item()
                    countnum = round(countnum)
                    labelnum = round(labelnum)
                    mae = abs(labelnum - countnum) / (labelnum + 1e-1)
                    testMAE.append(mae)
                    gap = labelnum - countnum
                    if abs(gap) <= 1:
                        acc += 1
                    print('{0} predict count: {1}, groundtruth: {2}'.format(batch_idx, countnum, labelnum))
                    batch_idx += 1
        print('=====================================================')
        print('checkpoint: ' + ckpt)
        print("MAE:{0},OBO:{1}".format(np.mean(testMAE), acc / len(datalist)))
        print('=====================================================')
device_ids = [0]


root_path = '/path/to/UCF/npzselectval/'

test_video_dir = 'test'
test_label_dir = 'test.csv'

# video swin transformer pretrained model and config
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'

# TransRAC trained model checkpoint, we will upload soon.
# lastckpt = './transrac_ckpt_pytorch_171.pt'
lastckpt = './checkpoint/ours/'

NUM_FRAME = 64
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [4]
# test_dataset = MyData(root_path, test_video_dir, test_label_dir, num_frame=NUM_FRAME)
my_model = TransferModel(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)
NUM_EPOCHS = 1

test_loop(NUM_EPOCHS, my_model, root_path, thresh=0.276, lastckpt=lastckpt, device_ids=device_ids)
