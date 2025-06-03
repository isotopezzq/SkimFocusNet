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
        # self.path = os.path.join(self.root_dir, self.child_dir)
        self.video_filename = os.listdir(self.video_dir)
        self.label_filename = os.path.join(self.root_dir, label_dir)
        self.file_list = []
        self.label_list = []
        self.num_idx = 4
        self.num_frames = frames  # model frames
        self.error = 0
        df = pd.read_csv(self.label_filename)
        for i in range(0, len(df)):
            filename = df.loc[i, 'name']
            label_tmp = df.values[i][self.num_idx:].astype(np.float64)
            label_tmp = label_tmp[~np.isnan(label_tmp)].astype(np.int32)
            # if self.isdata(filename, label_tmp):  # data is right format ,save in list
            self.file_list.append(filename)
            self.label_list.append(label_tmp)

    def __getitem__(self, index):
        """
        Save the preprocess frames and original video length in NPZ.
        npz[img = frames, fps = original_frames_length]
        """

        filename = self.file_list[index]
        label = self.label_list[index]
        video_path = os.path.join(self.video_dir, filename)
        npy_pth = npy_dir + os.path.splitext(filename)[0] + '.npz'

        cap = cv2.VideoCapture(video_path)
        frames = []
        if cap.isOpened():
            while True:
                success, frame_bgr = cap.read()
                if success is False:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (224, 224))
                frames.append(frame_rgb)
        cap.release()
        original_frames_length = len(frames)
        density = preprocess(original_frames_length, label, int(original_frames_length/4))
        if len(density) % 64 != 0:
            dummy = [0 for i in range(64 - len(density) % 64)]
            density.extend(dummy)
        density_global = preprocess(original_frames_length, label, self.num_frames)
        data = self.adjust_frames_part(frames)  # uncomment to adjust frames
        data = np.asarray(data)  # [f,h,w,c]
        if (data.size != 0):
            data = data.transpose(0, 3, 2, 1)  # [f,c,h,w]
        else:
            print(filename, ' is wrong video. size = 0')
            return -1
        data_global = self.adjust_frames(frames)
        data_global = np.asarray(data_global)  # [f,h,w,c]
        if (data_global.size != 0):
            data_global = data_global.transpose(0, 3, 2, 1)  # [f,c,h,w]
        else:
            print(filename, ' is wrong video. size = 0')
            return -1

        assert len(data) == len(density)
        for i in range(int(len(data) / 64)):
            npy_pth = npy_dir + os.path.splitext(filename)[0] + '_' + str(i).zfill(2)
            np.savez(npy_pth, imgs=data[64*i: 64*(i+1)], imgs_g=data_global, label=density[64*i: 64*(i+1)], label_g=density_global, name=os.path.splitext(filename)[0])  # [f,c,h,w]
            print('save npz: ' + npy_pth)

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
    data_root = "/video/dir/"
    tag = ['train', 'valid', 'test']
    npy_dir = '/output/npy/data/dir/'
    for _ in range(3):
        mod = tag[_]
        label_file = mod + '.csv'
        test = MyDataset(data_root, label_file, 64, mod)
        npy_dir = npy_dir + mod + '/'
        print('=========================================')
        print(mod, ' : ', npy_dir)
        if not os.path.exists(npy_dir):
            os.mkdir(npy_dir)
        for i in range(len(test)):
            a = test[i]
