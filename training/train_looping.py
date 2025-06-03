"""train or valid looping """
import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.my_tools import paint_smi_matrixs

torch.manual_seed(1)


def train_loop(n_epochs, model, train_set, valid_set, train=True, valid=True, inference=False, batch_size=1, lr=1e-6,
               ckpt_name='ckpt', lastckpt=None, saveckpt=False, log_dir='scalar', device_ids=[0], mae_error=False):
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=0)
    validloader = DataLoader(valid_set, batch_size=batch_size, pin_memory=False, shuffle=True, num_workers=0)
    model = nn.DataParallel(model.to(device), device_ids=device_ids)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    milestones = [i for i in range(0, n_epochs, 40)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.8)  # three step decay

    scaler = GradScaler()

    if lastckpt is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    lossMSE = nn.MSELoss()
    lossMSE_G = nn.MSELoss()
    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        trainLosses = []
        validLosses = []
        trainOBO = []
        trainOBOG = []
        validOBO = []
        validOBOG = []
        trainMAE = []
        trainMAEG = []
        validMAE = []
        validMAEG = []

        if train:
            batch_idx = 0
            for (frames, frames_g, label, label_g) in trainloader:
                with autocast():
                    model.train()
                    optimizer.zero_grad()
                    acc = 0
                    acc_g = 0
                    frames = frames.type(torch.FloatTensor).to(device)
                    frames_g = frames_g.type(torch.FloatTensor).to(device)

                    label = label.type(torch.FloatTensor).to(device)
                    label_g = label_g.type(torch.FloatTensor).to(device)

                    count = torch.sum(label, dim=1).type(torch.FloatTensor).to(device)
                    count_g = torch.sum(label_g, dim=1).type(torch.FloatTensor).to(device)

                    output, output_g = model(frames, frames_g)

                    predict_count = torch.sum(output, dim=1).type(torch.FloatTensor).to(device)
                    predict_count_g = torch.sum(output_g, dim=1).type(torch.FloatTensor).to(device)

                    loss = lossMSE(output, label)
                    loss_g = lossMSE_G(output_g, label_g)
                    loss3 = torch.sum(torch.abs(predict_count - count)) / (torch.sum(count) + 1e-2)
                    loss3_g = torch.sum(torch.abs(predict_count_g - count_g)) / (torch.sum(count_g) + 1e-2)

                    loss += loss_g

                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    gaps_g = torch.sub(predict_count_g, count_g).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / predict_count.flatten().shape[0]
                    for item in gaps_g:
                        if abs(item) <= 1:
                            acc_g += 1
                    OBO_G = acc_g / predict_count_g.flatten().shape[0]

                    trainOBO.append(OBO)
                    trainOBOG.append(OBO_G)
                    trainMAE.append(loss3.item())
                    trainMAEG.append(loss3_g.item())
                    trainLosses.append(loss.item())
                    batch_idx += 1
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            print("Epoch: {}, train_loss: {}, train_cdiff: {} G {}, train_OBO: {} G {}".format(\
                epoch, np.mean(trainLosses), np.mean(trainMAE), np.mean(trainMAEG), np.mean(trainOBO), np.mean(trainOBOG)))

        if valid and epoch > 100:
            with torch.no_grad():
                batch_idx = 0
                for (frames, frames_g, label, label_g) in validloader:
                    model.eval()
                    acc = 0
                    acc_g = 0
                    frames = frames.type(torch.FloatTensor).to(device)
                    frames_g = frames_g.type(torch.FloatTensor).to(device)

                    label = label.type(torch.FloatTensor).to(device)
                    label_g = label_g.type(torch.FloatTensor).to(device)

                    count = torch.sum(label, dim=1).type(torch.FloatTensor).to(device)
                    count_g = torch.sum(label_g, dim=1).type(torch.FloatTensor).to(device)

                    output, output_g = model(frames, frames_g)

                    predict_count = torch.sum(output, dim=1).type(torch.FloatTensor).to(device)
                    predict_count_g = torch.sum(output_g, dim=1).type(torch.FloatTensor).to(device)

                    loss = lossMSE(output, label)
                    loss_g = lossMSE_G(output_g, label_g)
                    loss3 = torch.sum(torch.abs(predict_count - count)) / (torch.sum(count) + 1e-2)
                    loss3_g = torch.sum(torch.abs(predict_count_g - count_g)) / (torch.sum(count_g) + 1e-2)

                    loss += loss_g

                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    gaps_g = torch.sub(predict_count_g, count_g).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / predict_count.flatten().shape[0]
                    for item in gaps_g:
                        if abs(item) <= 1:
                            acc_g += 1
                    OBO_G = acc_g / predict_count_g.flatten().shape[0]

                    validOBO.append(OBO)
                    validOBOG.append(OBO_G)
                    validMAE.append(loss3.item())
                    validMAEG.append(loss3_g.item())
                    validLosses.append(loss.item())

                    batch_idx += 1
                    
            print('=====================valid begin==========================')
            print("Epoch: {}, valid_loss: {}, valid_cdiff: {} G {}, valid_OBO: {} G {}".format(\
                epoch, np.mean(validLosses), np.mean(validMAE), np.mean(validMAEG), np.mean(validOBO),np.mean(validOBOG)))
            print('==========================================================')
            
        scheduler.step()
        if not os.path.exists('checkpoint/{0}/'.format(ckpt_name)):
            os.mkdir('checkpoint/{0}/'.format(ckpt_name))
        if saveckpt:
            if  epoch > 100:
                checkpoint = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'trainLosses': trainLosses,
                    'valLosses': validLosses
                }
                torch.save(checkpoint,
                           'checkpoint/{0}/'.format(ckpt_name) + str(epoch).zfill(3) + '_' + str(
                               round(np.mean(validMAE), 4)) + '.pt')
