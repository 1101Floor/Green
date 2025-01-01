import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)
class MGFUWithConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MGFUWithConv2d, self).__init__()
        
        # Convolutional layers for each feature (instead of Linear)
        self.conv_Ft = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv_Fs = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv_Fc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)

        # Gate convolution layers
        self.conv_Gt = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv_Gs = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.conv_Gc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        
    def forward(self, Ft, Fs, Fc):
        
        Ft_tilde = torch.tanh(self.conv_Ft(Ft)) 
        Fs_tilde = torch.tanh(self.conv_Fs(Fs))
        Fc_tilde = torch.tanh(self.conv_Fc(Fc))

        Gt = torch.sigmoid(self.conv_Gt(Ft))
        Gs = torch.sigmoid(self.conv_Gs(Fs))
        Gc = torch.sigmoid(self.conv_Gc(Fc))


        gates = torch.cat([Gt, Gs, Gc], dim=1)  
        normalized_weights = torch.softmax(gates, dim=1)  
        
        Gt_norm, Gs_norm, Gc_norm = torch.split(normalized_weights, normalized_weights.shape[1] // 3, dim=1) 
          
        F_fused = (Ft_tilde * Gt_norm) + (Fs_tilde * Gs_norm) + (Fc_tilde * Gc_norm)#1,4,5,8#1,5,8,4
        return F_fused 
class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        self.residual = residual
        padding = ((kernel_size[0] - 1) // 2, 0)  # 保持相同大小的默认填充
        self.tcn = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,(kernel_size[0], 1),(stride, 1),padding,),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))
        self.prelu = nn.PReLU()
    def forward(self, x):
        x = self.tcn(x) + self.residual(x)
        x = self.prelu(x)
        return x

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i-1] if i > 0 else input_size
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 1, dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.tcn(x)
        return y
class SU(nn.Module):
    def __init__(self, dim, n_levels=2):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.AFG = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)]
        )
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.AFG[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
                
            else:
                s = self.AFG[i](xc[i])
            out.append(s)
        
        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class TU(nn.Module):
    def __init__(self, k_size=[9, 17, 33, 65]):
        super(TU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv0 = nn.Conv1d(1, 1, kernel_size=k_size[0], padding=(k_size[0] - 1) // 2, bias=False)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size[1], padding=(k_size[1] - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size[2], padding=(k_size[2] - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=k_size[3], padding=(k_size[3] - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.line = nn.Linear(4, 1, bias=False)
    
    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        y0 = self.conv0(y).transpose(1, 2).unsqueeze(-1)
        y1 = self.conv1(y).transpose(1, 2).unsqueeze(-1)
        y2 = self.conv2(y).transpose(1, 2).unsqueeze(-1)
        y3 = self.conv3(y).transpose(1, 2).unsqueeze(-1)
        
        y_full = self.line(torch.cat([y0, y1, y2, y3], dim=2).squeeze(-1).squeeze(-1)).unsqueeze(-1)
        
        y = self.sigmoid(y_full)
        return x * y
    
class TGEEL(nn.Module):
    def __init__(self, obs_len, num_feature, hidden_dim, n_vessel):
        super(TGEEL, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_vessel = n_vessel
        self.lstm = nn.LSTM(input_size=obs_len * num_feature, hidden_size=hidden_dim)
        dropout=0.3
        self.LeakyReLU = torch.nn.LeakyReLU(0.1)
        self.MLP = nn.Sequential(
            nn.Linear(self.num_vessel*self.hidden_dim, self.num_vessel*self.hidden_dim),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(self.num_vessel*self.hidden_dim, self.num_vessel*self.hidden_dim),
            nn.ReLU6()
        )
        self.MLP = nn.Linear(self.num_vessel*self.hidden_dim, self.num_vessel*self.hidden_dim)
    def forward(self, T_f):
        batch_size, num_vessel, num_traj, pred_len, num_feature = T_f.shape
        vessel_encoded_outputs = []
        for vessel_idx in range(num_vessel):
            vessel_data = T_f[:, vessel_idx, :, :, :]
            vessel_data = vessel_data.reshape(-1, num_traj, pred_len * num_feature)  # Flatten features for LSTM
            lstm_output, (h_n, c_n) = self.lstm(vessel_data)
            last_output = self.LeakyReLU(lstm_output[:, -1, :])  # Take the last output from LSTM
            vessel_encoded_outputs.append(last_output)
        
        vessel_encoded_outputs = torch.stack(vessel_encoded_outputs, dim=1)
        vessel_encoded_outputs = vessel_encoded_outputs.reshape(batch_size, -1)  # [batch_size, num_vessel * hidden_dim]
        vessel_encoded_outputs = self.MLP(vessel_encoded_outputs)
        mu = vessel_encoded_outputs[:, :int(num_vessel * self.hidden_dim / 2)]
        log_var = vessel_encoded_outputs[:, int(num_vessel * self.hidden_dim / 2):]
        var = log_var.mul(0.5).exp_()
        eps = torch.FloatTensor(var.size()).normal_()
        eps = eps.cuda()
        z = eps.mul(var).add_(mu)  # Reparameterization trick
        return z, mu, log_var
    
    def reparameters(self, batch_size,num_vessel):
        z = get_noise((1, int(num_vessel * self.hidden_dim / 2)), "gaussian")
        z = z.repeat(batch_size, 1)
        return z  
    
class GREEN(nn.Module):
    def __init__(self,n_stgcnn =1, input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3,embed_size = 8, dim_z = 16, z_feature = 4,n_vessel = 3,num_channels=[10,10,10]):
        super(GREEN,self).__init__()
        
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        self.kernel_size = kernel_size
        self.embed_size = embed_size
        self.dim_z = dim_z
        self.z_feature = z_feature
        self.n_vessel = n_vessel
        self.n_stgcnn= n_stgcnn
        
        self.st_gcns_dis = nn.ModuleList()
        self.st_gcns_dis.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns_dis.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
                 
        self.st_gcns_tpca = nn.ModuleList()
        self.st_gcns_tpca.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns_tpca.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
           
        self.st_gcns_vs = nn.ModuleList()
        self.st_gcns_vs.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns_vs.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
            
        self.TGEEL = TGEEL(obs_len=self.pred_seq_len, num_feature = self.z_feature, hidden_dim=self.dim_z, n_vessel = 
                                                   self.n_vessel)
        
        self.MGFU = MGFUWithConv2d(in_channels =self.seq_len, \
                                      out_channels = self.seq_len, kernel_size = 3)
        
        self.SU = SU(dim = self.seq_len, n_levels=2)
        self.TU = TU([3,5,7,9])
                  
        self.TCN=TCN(int((self.n_vessel * dim_z / 2)+output_feat), output_feat, num_channels,(kernel_size, seq_len))
        
        self.TCN_1 = nn.Conv2d(seq_len,pred_seq_len,3,padding=1) 
        self.TCN_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1) 
        self.prelus = nn.PReLU()
    def forward(self,v,a_dis,a_tpca,a_s,FF):
        batch, num_fea, obs_len, num_vessel = v.shape
        z, mu, log_var = self.TGEEL(FF)
        Z = z.unsqueeze(1).unsqueeze(3).repeat(1,self.seq_len, 1, self.n_vessel)
        
        for k in range(self.n_stgcnn):
            v_dis,a_dis = self.st_gcns_dis[k](v,a_dis)
        for k in range(self.n_stgcnn):
            v_tpca,a_tpca = self.st_gcns_tpca[k](v,a_dis)
        for k in range(self.n_stgcnn):
            v_vs,a_vs = self.st_gcns_vs[k](v,a_dis) 
            
        v_dis = v_dis.view(v_dis.shape[0],v_dis.shape[2],v_dis.shape[1],v_dis.shape[3])
        v_tpca = v_tpca.view(v_tpca.shape[0],v_tpca.shape[2],v_tpca.shape[1],v_tpca.shape[3]) 
        v_vs = v_vs.view(v_vs.shape[0],v_vs.shape[2],v_vs.shape[1],v_vs.shape[3])

        v_dis = torch.cat((v_dis,Z), dim=2)
        v_tpca = torch.cat((v_tpca,Z), dim=2)
        v_vs = torch.cat((v_vs,Z), dim=2)
        
        v = self.MGFU(v_dis,v_tpca,v_vs)      
        
        SU_output = self.SU(v)
        TU_output = self.TU(v)
        
        v_output = v + SU_output + TU_output    
        
        v_output = v_output.view(v_output.shape[0],v_output.shape[2],v_output.shape[1],v_output.shape[3])
        v_output = self.prelus(self.TCN(v_output))#128,5,20,3      
        v_output = v_output.view(v_output.shape[0],v_output.shape[2],v_output.shape[1],v_output.shape[3])
        v_output = self.prelus(self.TCN_1(v_output))
        v_output = self.TCN_ouput(v_output)
        v_output = v_output.view(v_output.shape[0],v_output.shape[2],v_output.shape[1],v_output.shape[3])                         
        return v_output,a_dis, mu, log_var
                                    
    def inference(self,v,a_dis,a_tpca,a_s,FF):
        batch, num_fea, obs_len, num_vessel = v.shape
        z = self.TGEEL.reparameters(batch,self.n_vessel)
        Z = z.unsqueeze(1).unsqueeze(3).repeat(1,self.seq_len, 1, self.n_vessel)
        for k in range(self.n_stgcnn):
            v_dis,a_dis = self.st_gcns_dis[k](v,a_dis)
        for k in range(self.n_stgcnn):
            v_tpca,a_tpca = self.st_gcns_tpca[k](v,a_dis)
        for k in range(self.n_stgcnn):
            v_vs,a_vs = self.st_gcns_vs[k](v,a_dis)  
            
        v_dis = v_dis.view(v_dis.shape[0],v_dis.shape[2],v_dis.shape[1],v_dis.shape[3])
        v_tpca = v_tpca.view(v_tpca.shape[0],v_tpca.shape[2],v_tpca.shape[1],v_tpca.shape[3]) 
        v_vs = v_vs.view(v_vs.shape[0],v_vs.shape[2],v_vs.shape[1],v_vs.shape[3])

        v_dis = torch.cat((v_dis,Z), dim=2)
        v_tpca = torch.cat((v_tpca,Z), dim=2)
        v_vs = torch.cat((v_vs,Z), dim=2)
        
        v = self.MGFU(v_dis,v_tpca,v_vs)      
        
        SU_output = self.SU(v)
        TU_output = self.TU(v)
        
        v_output = v + SU_output + TU_output    
        
        v_output = v_output.view(v_output.shape[0],v_output.shape[2],v_output.shape[1],v_output.shape[3])
        v_output = self.prelus(self.TCN(v_output))     
        v_output = v_output.view(v_output.shape[0],v_output.shape[2],v_output.shape[1],v_output.shape[3])
        v_output = self.prelus(self.TCN_1(v_output))
        v_output = self.TCN_ouput(v_output)
        v_output = v_output.view(v_output.shape[0],v_output.shape[2],v_output.shape[1],v_output.shape[3])
        return v_output,a_dis