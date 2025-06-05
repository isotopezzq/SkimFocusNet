"""test TransRAC model"""
import os
## if your data is .mp4 form, please use RepCountA_raw_Loader.py
# from dataset.RepCountA_raw_Loader import MyData
## if your data is .npz form, please use RepCountA_Loader.py. It can speed up the training
from dataset.RepCountA_Loader import MyData
from models.TransRAC import TransferModel
from testing.test_looping import test_loop

device_ids = [0]

# # # we pick out the fixed frames from raw video file, and we store them as .npz file
# # # we currently support 64 or 128 frames
# data root path
root_path = '/home/zhaozhengqi/DATA/LLSP/newnpzSelect/test/'

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

test_loop(NUM_EPOCHS, my_model, root_path, thresh=0.27, lastckpt=lastckpt, device_ids=device_ids)