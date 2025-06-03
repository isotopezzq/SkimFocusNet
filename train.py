from dataset.RepCountA_new_Loader import MyData
from models.SkimFocusNet import SkimFocusNet
from training.train_looping import train_loop

# CUDA environment
device_ids = [0] #GPU index 0,1,2,3

# data root path
root_path = '/your/data/root/'

train_video_dir = 'train'
train_label_dir = 'train.csv'
valid_video_dir = 'valid'
valid_label_dir = 'valid.csv'

# load the pretrained swinT
checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

lastckpt = None #continue training

NUM_FRAME = 64
SCALES = [4]  #sliding window size

train_dataset = MyData(root_path, train_video_dir, train_label_dir, num_frame=NUM_FRAME)
valid_dataset = MyData(root_path, valid_video_dir, valid_label_dir, num_frame=NUM_FRAME)
my_model = SkimFocusNet(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)

NUM_EPOCHS = 200
LR = 8e-6
BATCH_SIZE = 32

train_loop(NUM_EPOCHS, my_model, train_dataset, valid_dataset, train=True, valid=True,
           batch_size=BATCH_SIZE, lr=LR, saveckpt=True, ckpt_name='ours', log_dir='ours', device_ids=device_ids,
           lastckpt=lastckpt, mae_error=False)
