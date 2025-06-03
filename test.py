import os
from models.SkimFocusNet import SkimFocusNet
from testing.test_looping import test_loop

device_ids = [0] #GPU index 0,1,2,3

root_path = '/your/data/root/'

test_video_dir = 'test'
test_label_dir = 'test.csv'

# load the pretrained SwinT
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'

lastckpt = './checkpoint/ours/' #ckpt folder

NUM_FRAME = 64
SCALES = [4]
NUM_EPOCHS = 1

thresh = 0.32       #select valid c_diff below thresh
my_model = SkimFocusNet(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)

test_loop(NUM_EPOCHS, my_model, root_path, thresh=thresh, lastckpt=lastckpt, device_ids=device_ids) 