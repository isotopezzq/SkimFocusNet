import numpy as np
from PIL import Image
from torchvision import transforms
import os
import cv2
import torch
import pandas as pd
import math
from scipy import integrate
# from label2rep import rep_label
from torch.utils.data import Dataset, DataLoader
from scipy import integrate
import matplotlib.pyplot as plt
import time
import random

def create_data(frames, label, w1, w2):
    frames0 = frames[:32]
    frames1 = frames[32:]
    label0 = label[:32]
    label1 = label[32:]
    
    r1 = random.randint(0, len(w1)-32)
    r2 = random.randint(0, len(w2)-32)

    w1 = w1[r1:r1+32]
    w2 = w2[r2:r2+32]
    fakelab = [0.0] * 32
    
    cat = random.randint(0, 5)
    catlist = [[1,2,0,0], [1,0,2,0], [1,0,0,2],
               [0,1,2,0], [0,1,0,2], [0,0,1,2]]
    rframes = []
    rlabel = []
    for num in catlist[cat]:
        if num == 0:
            rframes.append(w1 if random.randint(0,1) else w2)
            rlabel.extend(fakelab)
        if num == 1:
            rframes.append(frames0)
            rlabel.extend(label0)
        if num == 2:
            rframes.append(frames1)
            rlabel.extend(label1)

    rframes = np.concatenate(rframes, axis=0)
    return rframes, rlabel

def readvid(filename, frames):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if success is False:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (224, 224))
            frames.append(frame_rgb)
    cap.release()
    
def PDF(x, u, sig):
    # f(x)
    return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

# integral f(x)
def get_integrate(x_1, x_2, avg, sig):
    y, err = integrate.quad(PDF, x_1, x_2, args=(avg, sig))
    return y


def normalize_label(y_frame, y_length):
    # y_length: total frames
    # return: normalize_label  size:nparray(y_length,)

    y_label = [0 for i in range(y_length)]  # 坐标轴长度，即帧数
    for i in range(0, len(y_frame), 2):
        x_a = y_frame[i]
        x_b = y_frame[i + 1]
        avg = (x_b + x_a) / 2
        sig = (x_b - x_a) / 6
        num = x_b - x_a + 1  # 帧数量 update 1104
        if num != 1:
            for j in range(num):
                x_1 = x_a - 0.5 + j
                x_2 = x_a + 0.5 + j
                y_ing = get_integrate(x_1, x_2, avg, sig)
                y_label[x_a + j] = y_ing
        else:
            y_label[x_a] = 1
    return y_label

def preprocess(video_frame_length, time_points, num_frames):
    """
    process label(.csv) to density map label
    Args:
        video_frame_length: video total frame number, i.e 1024frames
        time_points: label point example [1, 23, 23, 40,45,70,.....] or [0]
        num_frames: 64
    Returns: for example [0.1,0.8,0.1, .....]
    """
    if len(time_points) == 0:
        return [0.0] * num_frames
    new_crop = []
    for i in range(len(time_points)):  # frame_length -> 64
        item = min(math.ceil((float((time_points[i])) / float(video_frame_length)) * num_frames), num_frames - 1)
        new_crop.append(item)
    new_crop = np.sort(new_crop)
    label = normalize_label(new_crop, num_frames)

    return label

def isdata(self, file, label):
    """
    :param file: original video
    :param label: original label
    :return: the number of load error
    """
    video_path = os.path.join(self.video_dir, file)
    cap = cv2.VideoCapture(video_path)
    frames_num = cap.get(7)
    if label.size == 0:
        # print('empty data:', file)
        self.error += 1
        return False
    elif frames_num >= max(label):
        return True
    else:
        # print("error data:", file, frames_num)
        # print('error data:', label)
        # print("error data:", len(label))
        self.error += 1
        return False


class MyDataset(Dataset):

    def __init__(self, root_dir, label_dir, frames, method):
        """
        :param root_dir: dataset root dir
        :param label_dir: dataset child dir
         # data_root = r'/data/train'
         # data_lable = data_root + 'label.xlsx'
         # data_video = data_root + 'video/'
        """
        super().__init__()
        self.root_dir = root_dir
        if method == 'train':
            self.video_dir = os.path.join(self.root_dir, r'train')
        elif method == 'valid':
            self.video_dir = os.path.join(self.root_dir, r'valid')
        elif method == 'test':
            self.video_dir = os.path.join(self.root_dir, r'test')
        else:
            raise ValueError('module is wrong.')

        self.flag_dir = os.path.join(self.root_dir, "flag")
        # self.path = os.path.join(self.root_dir, self.child_dir)
        # self.video_filename = os.listdir(self.video_dir)
        self.label_filename = os.path.join(self.root_dir, label_dir)
        self.file_list = []
        self.label_list = []
        self.flag_list = []
        self.wrong_list = []
        self.num_idx = 4
        self.num_frames = frames  # model frames
        self.error = 0
        self.csv = pd.read_csv(self.label_filename)
        # df = pd.read_csv(self.label_filename)
        # for i in range(0, len(df)):
        #     filename = df.loc[i, 'name']
        #     label_tmp = df.values[i][self.num_idx:].astype(np.float64)
        #     label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
        #     # if self.isdata(filename, label_tmp):  # data is right format ,save in list
        #     self.file_list.append(filename)
        #     self.label_list.append(label_tmp)
    def establish(self):
        actlist = os.listdir(self.video_dir)
        for act in actlist:
            vidlist = os.listdir(os.path.join(self.video_dir, act))
            for vid in vidlist:
                vid_path = os.path.join(self.video_dir, act, vid)
                self.file_list.append(vid_path)
                for i in range(len(self.csv)):
                    if self.csv.loc[i, 'name'] == vid:
                        break
                if (i == len(self.csv)-1 and self.csv.loc[i, 'name'] != vid):
                    print("!!!VIDEO DID NOT FOUND!!!")
                # if i % 10 != 0:
                label_tmp = self.csv.values[i][self.num_idx:].astype(np.float64)
                label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
                self.label_list.append(label_tmp)

                f1 = random.randint(0, len(os.listdir(os.path.join(self.flag_dir, act)))-1)
                for j in range(len(self.csv)):
                    if self.csv.loc[i, 'name'] == os.listdir(os.path.join(self.flag_dir, act))[f1]:
                        break
                flaglabel_tmp = self.csv.values[j][self.num_idx:].astype(np.float64)
                flaglabel_tmp = flaglabel_tmp[~np.isnan(flaglabel_tmp)].astype(np.int32)
                self.flag_list.append([os.path.join(self.flag_dir, act, os.listdir(os.path.join(self.flag_dir, act))[f1]), flaglabel_tmp])

                tmp_actlist = actlist.copy()
                tmp_actlist.remove(act)
                act_w = random.sample(tmp_actlist, 2)
                w1 = random.randint(0, len(os.listdir(os.path.join(self.video_dir, act_w[0])))-1)
                w2 = random.randint(0, len(os.listdir(os.path.join(self.video_dir, act_w[1])))-1)
                self.wrong_list.append([os.path.join(self.video_dir, act_w[0], os.listdir(os.path.join(self.video_dir, act_w[0]))[w1]), 
                                        os.path.join(self.video_dir, act_w[1], os.listdir(os.path.join(self.video_dir, act_w[1]))[w2])])
                
                # else:
                #     self.label_list.append([])

                #     tmp_actlist = actlist.copy()
                #     tmp_actlist.remove(act)
                #     act_w = random.sample(tmp_actlist, 3)
                #     f1 = random.randint(0, len(os.listdir(os.path.join(self.flag_dir, act_w[0])))-1)
                #     for j in range(len(self.csv)):
                #         if self.csv.loc[i, 'name'] == os.listdir(os.path.join(self.flag_dir, act_w[0]))[f1]:
                #             break
                #     flaglabel_tmp = self.csv.values[j][self.num_idx:].astype(np.float64)
                #     flaglabel_tmp = flaglabel_tmp[~np.isnan(flaglabel_tmp)].astype(np.int32)
                #     self.flag_list.append([os.path.join(self.flag_dir, act_w[0], os.listdir(os.path.join(self.flag_dir, act_w[0]))[f1]), flaglabel_tmp])
                    
                #     w1 = random.randint(0, len(os.listdir(os.path.join(self.video_dir, act_w[1])))-1)
                #     w2 = random.randint(0, len(os.listdir(os.path.join(self.video_dir, act_w[2])))-1)
                #     self.wrong_list.append([os.path.join(self.video_dir, act_w[1], os.listdir(os.path.join(self.video_dir, act_w[1]))[w1]), 
                #                             os.path.join(self.video_dir, act_w[2], os.listdir(os.path.join(self.video_dir, act_w[2]))[w2])])
                    


                
        

    def __getitem__(self, index):
        """
        Save the preprocess frames and original video length in NPZ.
        npz[img = frames, fps = original_frames_length]
        """

        filename = self.file_list[index]
        label = self.label_list[index]
        flag = self.flag_list[index]
        wrong = self.wrong_list[index]

        # video_path = os.path.join(self.video_dir, filename)

        # try:
        
        frames = []
        readvid(filename, frames)
        original_frames_length = len(frames)
        density = preprocess(original_frames_length, label, int(original_frames_length/4))
        if len(density) % 64 != 0:
            dummy = [0 for i in range(64 - len(density) % 64)]
            density.extend(dummy)
        
        
        flagframes = []
        readvid(flag[0], flagframes)
        flag_frames_length = len(flagframes)
        density_global = preprocess(flag_frames_length, flag[1], self.num_frames)

        data = self.adjust_frames_part(frames)  # uncomment to adjust frames
        data = np.asarray(data)  # [f,h,w,c]
        if (data.size != 0):
            data = data.transpose(0, 3, 2, 1)  # [f,c,h,w]
        else:
            print(filename, ' is wrong video. size = 0')
            return -1
        data_global = self.adjust_frames(flagframes)  # uncomment to adjust frames
        data_global = np.asarray(data_global)  # [f,h,w,c]
        if (data_global.size != 0):
            data_global = data_global.transpose(0, 3, 2, 1)  # [f,c,h,w]
        else:
            print(filename, ' is wrong video. size = 0')
            return -1
        # npy_pth = r'/p300/LSP_npz/npy_data/' + os.path.splitext(filename)[0]
        assert len(data) == len(density)
        
        w1frames = []
        w2frames = []
        readvid(wrong[0], w1frames)
        readvid(wrong[1], w2frames)
        w1frames = self.adjust_frames_part(w1frames)  # uncomment to adjust frames
        w1frames = np.asarray(w1frames)  # [f,h,w,c]
        if (w1frames.size != 0):
            w1frames = w1frames.transpose(0, 3, 2, 1)  # [f,c,h,w]
        else:
            print(filename, ' is wrong video. size = 0')
            return -1
        w2frames = self.adjust_frames_part(w2frames)  # uncomment to adjust frames
        w2frames = np.asarray(w2frames)  # [f,h,w,c]
        if (w2frames.size != 0):
            w2frames = w2frames.transpose(0, 3, 2, 1)  # [f,c,h,w]
        else:
            print(filename, ' is wrong video. size = 0')
            return -1
        for i in range(int(len(data) / 64)):
            npy_pth = dira + os.path.splitext(filename)[0].split('/')[-1] + '_' + str(i).zfill(2) + '.npz'
            imgsc, labelc = create_data(data[64*i: 64*(i+1)], density[64*i: 64*(i+1)], w1frames, w2frames)
            np.savez(npy_pth, imgs=imgsc, imgs_g=data_global, label=labelc, label_g=density_global, name=os.path.splitext(filename)[0])  # [f,c,h,w]
            print('save npz: ' + npy_pth)
        # except:
        #     print('error: ', video_path, ' cannot open')

        # uncomment to check data
        # y1, y2, num_period = self.adjust_label(self.label_list[index], original_frames_length, self.num_frames)
        # count = len(self.label_list[index]) / 2
        # if count == 0:
        #     print('file:', filename)
        # if count - num_period > 0.1:
        #     print('file:', filename)
        #     print('y1:', y1)
        #     print('y2:', y2)
        #     print('count:', count)
        #     print('sum_y:', num_period)
        return 1

    def __len__(self):
        return len(self.file_list)

    def adjust_frames_part(self, frames):
        """
        # adjust the number of total video frames to the target frame num.
        :param frames: original frames
        :return: target number of frames
        """
        frames_adjust = []
        frame_length = len(frames)
        for i in range(1, int(frame_length / 4) + 1):
            frame = frames[i * frame_length // int(frame_length / 4) - 1]
            frames_adjust.append(frame)
        if len(frames_adjust) % 64 != 0:
            for _ in range(64 - len(frames_adjust) % 64):
                frame = frames[-1]
                frames_adjust.append(frame)
        return frames_adjust  # [f,h,w,3]

    def adjust_frames(self, frames):
        """
        # adjust the number of total video frames to the target frame num.
        :param frames: original frames
        :return: target number of frames
        """
        frames_adjust = []
        frame_length = len(frames)
        if self.num_frames <= len(frames):
            for i in range(1, self.num_frames + 1):
                frame = frames[i * frame_length // self.num_frames - 1]
                frames_adjust.append(frame)
        else:
            for i in range(frame_length):
                frame = frames[i]
                frames_adjust.append(frame)
            for _ in range(self.num_frames - frame_length):
                if len(frames) > 0:
                    frame = frames[-1]
                    frames_adjust.append(frame)
        return frames_adjust  # [f,h,w,3]

if __name__ == '__main__':
    data_root = r'/home/zhaozhengqi/DATA/LLSP/video_category/'
    tag = ['train', 'test', 'valid']
    # tag = ['test']
    npy_dir = r'/home/zhaozhengqi/DATA/LLSP/newnpzcat/'
    for _ in range(3):
        mod = tag[_]
        label_file = mod + '.csv'
        test = MyDataset(data_root, label_file, 32, mod)
        dira = npy_dir + mod + '/'
        print('=========================================')
        print(mod, ' : ', dira)
        if not os.path.exists(dira):
            os.mkdir(dira)
        test.establish()
        print("LIST ESTABLISHED")
        
        for i in range(len(test)):
            a = test[i]
