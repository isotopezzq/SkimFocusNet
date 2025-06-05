""" test of TransRAC """
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.my_tools import paint_smi_matrixs, plot_inference, density_map

torch.manual_seed(1)

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
        # plot_inference(predict_count, count)

# shortlist = ['stu3_5', 'stu6_39', 'stu13_30', 'stu13_31', 'stu4_47', 'stu12_31', 'stu13_33', 'stu6_23', 'stu7_35', 'stu4_5', 'stu2_25', 'stu6_24', 'stu13_34', 'stu6_68', 'train1615', 'stu13_35', 'stu8_34', 'stu12_30', 'stu13_32', 'stu6_15', 'stu5_23', 'stu9_31', 'stu9_57', 'stu9_52', 'stu10_36', 'stu3_25', 'stu2_55', 'stu13_37', 'stu9_34', 'stu8_25', 'stu8_28', 'stu4_19', 'stu4_46', 'stu5_27', 'stu13_38', 'stu12_34', 'stu4_27', 'stu3_31', 'train3344', 'stu13_36', 'stu3_20', 'stu5_18', 'stu6_22', 'stu13_39', 'stu2_36', 'stu12_33', 'stu3_28', 'stu12_32', 'stu7_30', 'stu9_29', 'train151']
# longlist = ['test2315', 'stu2_62', 'stu1_30', 'stu9_64', 'stu10_1', 'stu1_59', 'stu9_47', 'stu6_42', 'stu8_38', 'stu4_69', 'train2534', 'val1257', 'train3899', 'stu10_25', 'stu7_13', 'val1244', 'train2659', 'stu8_70', 'stu3_53', 'stu2_37', 'stu4_68', 'stu10_43', 'stu9_71', 'stu8_58', 'train3335', 'stu6_30', 'stu7_8', 'stu6_3', 'stu2_14', 'stu6_13', 'train2737', 'stu3_55', 'stu4_29', 'stu7_45', 'stu2_6', 'stu8_11', 'stu5_12', 'stu8_10', 'stu8_49', 'stu8_23', 'stu3_15', 'stu2_39', 'stu7_17', 'stu2_65', 'stu5_16', 'stu10_17', 'stu5_55', 'stu3_39', 'test539', 'stu10_5', 'stu4_12']
# lowlist = ['stu6_22', 'train2534', 'stu5_47', 'stu7_17', 'stu13_33', 'stu8_46', 'stu8_25', 'stu4_19', 'stu2_34', 'stu1_60', 'stu3_31', 'stu7_30', 'test538', 'stu6_23', 'train1615', 'stu4_27', 'stu10_36', 'stu9_36', 'stu3_25', 'stu6_42', 'train3899', 'stu5_23', 'val1257', 'train2737', 'test539', 'stu2_25', 'stu12_32', 'stu12_30', 'test135', 'stu6_24', 'stu3_58', 'stu9_52', 'stu10_17', 'train3344', 'test534', 'stu5_32', 'val579', 'stu8_57', 'stu8_71', 'stu10_60', 'stu9_29', 'stu8_11', 'stu8_42', 'test2025', 'train3335', 'stu8_58']
# highlist = ['stu9_47', 'stu10_6', 'stu4_29', 'stu10_48', 'stu3_55', 'stu4_68', 'stu1_36', 'stu7_45', 'stu10_24', 'stu3_15', 'stu6_39', 'stu6_30', 'stu8_12', 'stu3_18', 'stu4_63', 'stu4_46', 'stu8_70', 'stu8_28', 'stu7_4', 'stu4_64', 'stu5_27', 'stu2_55', 'stu9_71', 'stu5_68', 'stu10_5', 'stu6_3', 'stu6_68', 'stu8_9', 'stu7_8', 'stu2_36', 'stu9_17', 'stu12_34', 'stu4_47', 'stu9_48', 'stu8_34', 'stu2_5', 'stu8_23', 'stu10_1', 'stu4_69', 'stu6_15', 'stu2_6', 'stu13_32', 'stu7_10', 'train151', 'stu1_27', 'stu6_63']
# def test_loop(n_epochs, model, root_path, thresh, inference=True, batch_size=1, lastckpt=None, paint=False, device_ids=[0]):

#     ckptlist = []
#     listdir(lastckpt, ckptlist, thresh)
#     datalist = os.listdir(root_path)
#     datalist = data_reorgan(datalist)
#     device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
#     # testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=10)
#     model = nn.DataParallel(model.to(device), device_ids=device_ids)


#     for ckpt in tqdm(ckptlist):

#         print("loader ckpt: " + ckpt)
#         checkpoint = torch.load(ckpt, map_location=device)
#         model.load_state_dict(checkpoint['state_dict'], strict=False)
#         del checkpoint

#         testOBO = []
#         testMAE = []
#         predCount = []
#         Count = []
#         if inference:
#             with torch.no_grad():
#                 batch_idx = 0
#                 acc = 0
#                 for data in datalist:
#                     if data[0][:-7] in lowlist:

#                         countnum = 0
#                         labelnum = 0
#                         for d in data:
#                             frames, frames_g, frames_s, label, label_g, name = get_frames(root_path + d)  # [64, 3, 224, 224]
#                             # import pdb;pdb.set_trace()
#                             frames = frames.transpose(0, 1)  # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
#                             frames_g = frames_g.transpose(0, 1)
#                             frames_s = frames_s.transpose(0, 1)
#                             model.eval()
#                             frames = frames.type(torch.FloatTensor).unsqueeze(0).to(device)
#                             frames_g = frames_g.type(torch.FloatTensor).unsqueeze(0).to(device)
#                             frames_s = frames_s.type(torch.FloatTensor).unsqueeze(0).to(device)

#                             label = label.type(torch.FloatTensor).unsqueeze(0).to(device)
#                             label_g = label_g.type(torch.FloatTensor).unsqueeze(0).to(device)
#                             count = torch.sum(label, dim=1).type(torch.FloatTensor).to(device)
#                             # count_g = torch.sum(label_g, dim=1).type(torch.FloatTensor).to(device)
#                             # start = time.time()
#                             output, output_g = model(frames, frames_g, frames_s)
#                             # density_map(maps=output_g, index=0, cmp='YlGnBu', file_name=d.split('.')[0])
#                             # density_map(maps=label_g, index=1, cmp='crest', file_name=d.split('.')[0])
#                             # print(time.time()-start)
#                             predict_count = torch.sum(output, dim=1)
#                             # predict_count_g = torch.sum(output_g, dim=1)
#                             countnum += predict_count.item()
#                             labelnum += count.item()
#                         countnum = round(countnum)
#                         labelnum = round(labelnum)
#                         mae = abs(labelnum - countnum) / (labelnum + 1e-1)
#                         testMAE.append(mae)
#                         gap = labelnum - countnum
#                         if abs(gap) <= 1:
#                             acc += 1
#                         print('{0} predict count: {1}, groundtruth: {2}'.format(batch_idx, countnum, labelnum))
#                         batch_idx += 1
#         print('=====================================================')
#         print('checkpoint: ' + ckpt)
#         print("MAE:{0},OBO:{1}".format(np.mean(testMAE), acc / (batch_idx)))
#         print('=====================================================')
#         # plot_inference(predict_count, count)
