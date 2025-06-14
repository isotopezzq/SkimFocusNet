"""TransRAC network"""
from tokenize import Single
from turtle import forward
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from .Temporal_Encoder import TemporalEncoder, SingleStage
from random import randint

def find_nearest(array, value):
    idx = (torch.abs(array - value)).argmin()
    return int(idx)

class attention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, scale=64, att_dropout=None):
        super().__init__()
        # self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(att_dropout)
        self.scale = scale

    def forward(self, q, k, v, attn_mask=None):
        # q: [B, head, F, model_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.scale)  # [B,Head, F, F]
        if attn_mask:
            scores = scores.masked_fill_(attn_mask, -np.inf)
        scores = self.softmax(scores)
        scores = self.dropout(scores)  # [B,head, F, F]
        # context = torch.matmul(scores, v)  # output
        return scores  # [B,head,F, F]


class Similarity_matrix(nn.Module):
    ''' buliding similarity matrix by self-attention mechanism '''

    def __init__(self, num_heads=4, model_dim=512):
        super().__init__()

        # self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.input_size = 512
        self.linear_q = nn.Linear(self.input_size, model_dim)
        self.linear_k = nn.Linear(self.input_size, model_dim)
        self.linear_v = nn.Linear(self.input_size, model_dim)

        self.attention = attention(att_dropout=0)
        # self.out = nn.Linear(model_dim, model_dim)
        # self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        # linear projection
        query = self.linear_q(query)  # [B,F,model_dim]
        key = self.linear_k(key)
        value = self.linear_v(value)
        # split by heads
        # [B,F,model_dim] ->  [B,F,num_heads,per_head]->[B,num_heads,F,per_head]
        query = query.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        key = key.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        value = value.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        # similar_matrix :[B,H,F,F ]
        matrix = self.attention(query, key, value, attn_mask)

        return matrix


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class TransEncoder(nn.Module):
    '''standard transformer encoder'''

    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1, num_frames=64):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, num_frames)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op

class GLMix(nn.Module):
    def __init__(self, input_dim=1024, hdim1=512):
        super(GLMix, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hdim1),
            nn.LayerNorm(hdim1),
            nn.Dropout(p=0.25, inplace=False),
            nn.ReLU(True),
            nn.Linear(hdim1, input_dim),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x): #[b, 64, 1024]
        proj = self.MLP(x)
        proj = self.sigmoid(proj)
        proj = proj * x + x
        return proj



class Prediction(nn.Module):
    ''' predict the density map with densenet '''

    def __init__(self, input_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Prediction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=0.25, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
 
    
class GlobalFeature(nn.Module):
    def __init__(self, scales, frames):
        super(GlobalFeature, self).__init__()
        self.scales = scales
        self.num_frames = frames
        
        self.sims = Similarity_matrix()
        self.conv3x3 = nn.Conv2d(in_channels=4 * len(self.scales),  # num_head*scale_num
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)

        self.bn2 = nn.BatchNorm2d(32)
        self.onestage = SingleStage()
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)  # 线性投射层
        self.ln1 = nn.LayerNorm(512)

        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=0.2, dim_ff=512, num_layers=1,
                                         num_frames=self.num_frames)
        self.FC = Prediction(512, 512, 256, 1)  #


    def forward(self, xpre):
                # -------- similarity matrix ---------
        # for x_scale in xpre:
        #     # x_scale = x_scale.transpose(1, 2)
        #     # x_scale = self.onestage(x_scale)
        #     # x_scale = x_scale.transpose(1, 2)  # -> [b,f,512]
        #     x_sims = F.relu(self.sims(x_scale, x_scale, x_scale))  # -> [b,4,f,f]
        #     multi_scales.append(x_sims)
        # x = torch.cat(multi_scales, dim=1)  # [B,4*scale_num,f,f]
        # ## x are the similarity matrixs
        # x = F.relu(self.bn2(self.conv3x3(x)))  # [b,32,f,f]
        # x = self.dropout1(x)

        # x = x.permute(0, 2, 3, 1)  # [b,f,f,32]
        # # --------- transformer encoder ------
        # x = x.flatten(start_dim=2)  # ->[b,f,32*f]
        # x = F.relu(self.input_projection(x))  # ->[b,f, 512]
        # x = self.ln1(x)

        # x = x.transpose(0, 1)  # [f,b,512]
        # x = self.transEncoder(x)  #
        # x = x.transpose(0, 1)  # ->[b,f, 512]

        # x = self.FC(x)  # ->[b,f,1]
        # x = x.squeeze(2)
        # globpred = x
        max_feature = []
        for feature in xpre:
            # norm_label = (x / torch.sum(x, dim=1, keepdim=True)).unsqueeze(2)
            # max_feature.append((feature * norm_label).sum(1)) 
            max_feature.append(torch.max(feature, dim=1)[0])
        return max_feature





class TransferModel(nn.Module):
    def __init__(self, config, checkpoint, num_frames, scales, OPEN=False):
        super(TransferModel, self).__init__()
        self.num_frames = num_frames
        self.config = config
        self.checkpoint = checkpoint
        self.scales = scales
        self.OPEN = OPEN

        self.backbone, self.small = self.load_model()  # load pretrain model

        indexlist = torch.ones(32).long()
        self.register_buffer("indexlist", indexlist)
        self.conv3D = nn.Conv3d(in_channels=768,
                                out_channels=512,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))
        
        self.conv3Dselect = nn.Conv3d(in_channels=192,
                                out_channels=128,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))
        self.bns = nn.BatchNorm3d(128)
        self.bn1 = nn.BatchNorm3d(512)
        self.pool = nn.MaxPool3d(kernel_size=(1,7,7))
        self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))
        self.globfeature = GlobalFeature(self.scales, int(self.num_frames/2))
        self.glmix = GLMix(1024, 512)
        self.longshort = SingleStage(in_feat_dim=1024)
        self.sims = Similarity_matrix()
        self.conv3x3 = nn.Conv2d(in_channels=4 * len(self.scales),  # num_head*scale_num
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)

        self.bn2 = nn.BatchNorm2d(32)

        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)  # 线性投射层
        self.ln1 = nn.LayerNorm(512)

        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=0.2, dim_ff=512, num_layers=1,
                                         num_frames=self.num_frames)
        self.trans = TransEncoder(d_model=128, n_head=4, dropout=0.2, dim_ff=128, num_layers=1,
                                         num_frames=self.num_frames * 4)
        self.FC = Prediction(512, 512, 256, 1)  #
        self.FCs = Prediction(128, 128, 64, 1)

    def load_model(self):
        # # # load  pretrained model of video swin transformer using mmaction and mmcv API
        cfg = Config.fromfile(self.config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

        # # # load hyperparameters by mmcv api
        load_checkpoint(model, self.checkpoint, map_location='cpu')
        backbone = model.backbone
        small = nn.Sequential(backbone.patch_embed, backbone.pos_drop, backbone.layers[0])

        # # # load hyperparameters by pytorch
        # loaded_ckpt = torch.load(self.checkpoint)
        # backbone = model.backbone
        # net_dict = backbone.state_dict()
        # state_dict = {k: v for k, v in loaded_ckpt.items() if k in net_dict.keys()}
        # net_dict.update(state_dict)
        # backbone.load_state_dict(net_dict, strict=False)
        print('--------- backbone loaded ------------')

        return backbone, small

    def Feature_Extractor(self, x, num_frames):
        with torch.no_grad():
            x_scale = self.backbone(x)
        # import pdb;pdb.set_trace()
        crops = [x_scale[:, :, i:i+1, :, :] for i in range(x_scale.shape[2])]
        slice = []
        for crop in crops:
            slice.append(crop.repeat(1,1,2,1,1))
        x_scale = torch.cat(slice, dim=2)  # -> [b,768,f,size,size]
        x_scale = F.relu(self.bn1(self.conv3D(x_scale)))  # ->[b,512,f,7,7]
        # print(x_scale.shape)
        x_scale = self.SpatialPooling(x_scale)  # ->[b,512,f,1,1]
        x_scale = x_scale.squeeze(3).squeeze(3)  # -> [b,512,f]
        x_scale = x_scale.transpose(1, 2)
        # multi_scales.append(x_scale)

        return [x_scale]
    
    def select(self, x):
        with torch.no_grad():
            x_scale = self.small(x)
        crops = [x_scale[:, :, i:i+1, :, :] for i in range(x_scale.shape[2])]
        slice = []
        for crop in crops:
            slice.append(crop.repeat(1,1,2,1,1))
        x_scale = torch.cat(slice, dim=2)
        x_scale = F.relu(self.bns(self.conv3Dselect(x_scale)))
        x_scale = self.pool(x_scale)  # ->[b,128,f,1,1]
        x_scale = x_scale.squeeze(3).squeeze(3)  # -> [b,128,f]
        x = x_scale.transpose(1, 2)
        x = x.transpose(0, 1)  # [f,b,512]
        x = self.trans(x)  #
        x = x.transpose(0, 1)  # ->[b,f, 512]
        x = self.FCs(x)

        return x

    def forward(self, x, xg, xs):

        out = F.relu(self.select(xs))

        # cumout = (torch.cumsum(out, dim=1) / torch.sum(out, dim=1, keepdim=True)).squeeze(-1)
        vidlist = []

        # for i in range(cumout.shape[0]):
        #     for j in range(32):
        #         now = find_nearest(cumout[i], (j+1)/32)

        #         self.indexlist[j] = now

        #     vidlist.append(torch.index_select(xg[i], dim=1, index=self.indexlist).unsqueeze(0))
        # del xg, xs

        for i in range(out.shape[0]):
            index = torch.topk(out[i], 32, dim=0)[1].squeeze(-1)
            vidlist.append(torch.index_select(xg[i], dim=1, index=index).unsqueeze(0))
        xg = torch.cat(vidlist)

        # x: tensor([batch_size, channel, temporal_dim, height, width])
        feature = self.Feature_Extractor(x, self.num_frames) 
        feature_g = self.Feature_Extractor(xg, xg.shape[2])

        glob = self.globfeature(feature_g)
        multi_scales = []
        mix = []
        for x_scale, glof in zip(feature, glob):
            x_scale = torch.cat((x_scale, glof.unsqueeze(1).repeat(1, self.num_frames, 1)), dim=-1)
            x_scale = self.glmix(x_scale)
            x_scale = x_scale.transpose(1, 2)
            x_scale = self.longshort(x_scale)
            x_scale = x_scale.transpose(1, 2)
            x_sims = F.relu(self.sims(x_scale, x_scale, x_scale))  # -> [b,4,f,f]
            multi_scales.append(x_sims)
        x = torch.cat(multi_scales, dim=1)  # [B,4*scale_num,f,f]
        ## x are the similarity matrixs
        del multi_scales, mix
        x = F.relu(self.bn2(self.conv3x3(x)))  # [b,32,f,f]
        x = self.dropout1(x)

        x = x.permute(0, 2, 3, 1)  # [b,f,f,32]
        # --------- transformer encoder ------
        x = x.flatten(start_dim=2)  # ->[b,f,32*f]
        x = F.relu(self.input_projection(x))  # ->[b,f, 512]
        x = self.ln1(x)

        x = x.transpose(0, 1)  # [f,b,512]
        x = self.transEncoder(x)  #
        x = x.transpose(0, 1)  # ->[b,f, 512]

        x = self.FC(x)  # ->[b,f,1]
        x = x.squeeze(2)
        return x, out.squeeze(-1)

