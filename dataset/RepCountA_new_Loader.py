''' 
Repcount data loader from fixed frames file(.npz) which will be uploaded soon.
if you don't pre-process the data file,for example,your raw file is .mp4,
you can use the *RepCountA_raw_Loader.py*(slowly).
or
you can use 'tools.video2npz.py' to transform .mp4 tp .npz
'''
import csv
import os
import os.path as osp
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader
import torch
from .label_norm import normalize_label


class MyData(Dataset):

    def __init__(self, root_path, video_path, label_path, num_frame):
        """
        :param root_path: root path
        :param video_path: video child path (folder)
        :param label_path: label child path(.csv)
        """
        self.root_path = root_path
        self.video_path = os.path.join(self.root_path, video_path)  # train or valid
        self.label_path = os.path.join(self.root_path, label_path)
        self.video_dir = os.listdir(self.video_path)
        self.label_dict = get_labels_dict(self.label_path)  # get all labels
        self.num_frame = num_frame

    def __getitem__(self, inx):
        """ get data item
        :param  video_tensor, label
        """
        video_file_name = self.video_dir[inx]
        file_path = os.path.join(self.video_path, video_file_name)
        frames, frames_g, label, label_g, name = get_frames(file_path)  # [64, 3, 224, 224]
        frames = frames.transpose(0, 1)  # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
        frames_g = frames_g.transpose(0, 1)

        return [frames, frames_g, label, label_g]

    def __len__(self):
        """:return the number of video """
        return len(self.video_dir)


def get_frames(npz_path):
    # get frames from .npz files
    with np.load(npz_path, allow_pickle=True) as data:
        frames = data['imgs']  
        frames_g = data['imgs_g']
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
    return frames, frames_g, label, label_g, name


def get_labels_dict(path):
    # read label.csv to RAM
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(float(row[key])) for key in row.keys() if 'L' in key and row[key] != '']
            if not row['count']:
                print(row['name'] + 'error')
            else:
                labels_dict[row['name'].split('.')[0] + str('.npz')] = cycle

    return labels_dict


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


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
