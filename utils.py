from math import *
import os
import math
import sys
import torch.nn.functional as F
from dtw import dtw
from math import sqrt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
import random

def generate_seq_sub(frames_1, len_sub):

    seq_sub = []
    
    for i in frames_1:
        for j in len_sub:
            seq_sub.append(np.arange(i, i + j + 1, 1))
    
    return seq_sub

def generate_Traj_index(seq_sub):

    Traj_index = []
    
    for i in seq_sub:
        if len(i) >= 2:
            index = next((idx for idx, elem in enumerate(seq_sub) if np.array_equal(elem, i)), None)
            Traj_index.append((index, i[0], i[-1]))
        elif len(i) == 1:
            index = next((idx for idx, elem in enumerate(seq_sub) if np.array_equal(elem, i)), None)
            Traj_index.append((index, i[0], i[0]))
    
    return Traj_index

def generate_Traj_graph(Traj_index):

    Traj_graph = []
    
    for i in Traj_index:
        for j in Traj_index:
            if i[2] + 1 == j[1]:
                Traj_graph.append((i[0], j[0]))
    
    return Traj_graph

def generate_Node_embedding(Traj_index, data):

    Node_embedding = []
    
    for i in Traj_index:
        embedding = np.mean(data[(data[:, 0] >= i[1]) & (data[:, 0] <= i[2])][:, 2:6], axis=0)
        Node_embedding.append(embedding)
    
    return Node_embedding 

def create_graph_from_data(data):

    frames_1 = np.unique(data[:, 0]).tolist()
    len_sub = np.arange(0, 6) 

    seq_sub = generate_seq_sub(frames_1, len_sub)

    Traj_index = generate_Traj_index(seq_sub)
    Trend = []
    Trend_frame = []
    for i in Traj_index:
        if i[1]==i[2]:
            Trend.append(i[0])
            Trend_frame.append(i[1])
    Traj_graph = generate_Traj_graph(Traj_index)
    
    Node_embedding = generate_Node_embedding(Traj_index, data)

    G = nx.DiGraph()
    G.add_edges_from(Traj_graph)
    
    return Trend,Trend_frame, Traj_index, G
def random_walk_fixed_length(G, start_node, walk_length):

    while True:
        walk = [start_node]
        
        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = list(G.neighbors(current_node))
            
            if len(neighbors) > 0:
                next_node = random.choice(neighbors)
                walk.append(next_node)
            else:
                break

        if len(walk) == walk_length:
            return walk 

def random_walk_sampling_for_list_fixed_length(G, start_nodes, walk_length, num_traj_samples):

    walks_dict = {}
    
    for start_node in start_nodes:
        walks = []
        for _ in range(num_traj_samples):
            walk = random_walk_fixed_length(G, start_node, walk_length)
            walks.append(walk)
        walks_dict[start_node] = walks
    
    return walks_dict
class tarship():
    def __init__(self, lat, lon, cog, sog):
        self.lat = lat
        self.lon = lon
        self.cog = cog
        self.sog = sog


class refship():
    def __init__(self, lat, lon, cog, sog):
        self.lat = lat
        self.lon = lon
        self.cog = cog
        self.sog = sog


class Cal():
    def __init__(self, tar_ship, ref_ship):
        self.tar_lat = tar_ship.lat
        self.tar_lon = tar_ship.lon
        self.tar_cog = tar_ship.cog
        self.tar_sog = tar_ship.sog
        self.ref_lat = ref_ship.lat
        self.ref_lon = ref_ship.lon
        self.ref_cog = ref_ship.cog
        self.ref_sog = ref_ship.sog
        self.differ_lon = self.tar_lon - self.ref_lon  # 经差
        self.differ_cog = self.ref_cog - self.tar_cog
        self.differ_lon2 = self.tar_lon - self.ref_lon
        self.ref_lat2 = ref_ship.lat  # 存储本船纬度的正负号

    def dist(self):
        if self.ref_lat >= 0 and self.ref_lat * self.tar_lat >= 0:  # 本船纬度无论南北都是正值，他船与我船同名时为正值，异名时为负值
            self.ref_lat = self.ref_lat
            self.tar_lat = self.tar_lat
        elif self.ref_lat >= 0 and self.ref_lat * self.tar_lat < 0:  # 在这里把ref_lat的正负号都弄成了正值！error:这也是导致一开始算数不对的原因
            self.ref_lat = self.ref_lat
            self.tar_lat = self.tar_lat
        elif self.ref_lat < 0 and self.ref_lat * self.tar_lat >= 0:
            self.tar_lat = -self.tar_lat
            self.ref_lat = -self.ref_lat
        elif self.ref_lat < 0 and self.ref_lat * self.tar_lat < 0:
            self.tar_lat = -self.tar_lat
            self.ref_lat = -self.ref_lat
        if fabs(self.differ_lon) >= 180:  # 经差超过180°时，用360°减去它
            self.differ_lon = 360 - fabs(self.differ_lon)
        D = acos(sin(radians(self.tar_lat)) * sin(radians(self.ref_lat)) + cos(radians(self.tar_lat)) * cos(
            radians(self.ref_lat)) * cos(radians(fabs(self.differ_lon))))  # 边的余弦公式
        # print("距离为：%s"%float(D*180/pi*60))                        #算两船距离
        return D * 180 / pi * 60

    def true_bearing(self):
        # differ_lon=self.tar_lon-self.ref_lon                       #它船与本船的经差。注意：经差无论东西，一律正值
        if self.ref_lat >= 0 and self.ref_lat * self.tar_lat >= 0:  # 本船纬度无论南北都是正值，他船与我船同名时为正值，异名时为负值
            self.ref_lat = self.ref_lat
            self.tar_lat = self.tar_lat
        elif self.ref_lat >= 0 and self.ref_lat * self.tar_lat < 0:
            self.ref_lat = self.ref_lat
            self.tar_lat = self.tar_lat
        elif self.ref_lat < 0 and self.ref_lat * self.tar_lat >= 0:
            self.tar_lat = -self.tar_lat
            self.ref_lat = -self.ref_lat
        elif self.ref_lat < 0 and self.ref_lat * self.tar_lat < 0:
            self.tar_lat = -self.tar_lat
            self.ref_lat = -self.ref_lat
        if fabs(self.differ_lon) >= 180:  # 经差超过180°时，用360°减去它
            self.differ_lon = 360 - fabs(self.differ_lon)
            # 这里的经差不能为0
        # d=self.dist()*pi/180
        # a2=cos(radians(self.tar_lat))*sin(radians(self.differ_lon))/sin(d)
        # print("方位角2：",asin(a2)*180/pi)
        TB = 0
        if self.differ_lon == 0 or self.differ_lon == 180:  # 在同一条经线上时
            if self.ref_lat > self.tar_lat:
                p = 180
            else:
                p = 0
        else:
            a = tan(radians(self.tar_lat)) * cos(radians(self.ref_lat)) * 1 / sin(radians(fabs(self.differ_lon))) - sin(
                radians(self.ref_lat)) * 1 / tan(  # 四联公式
                radians(fabs(self.differ_lon)))
            # a=(cos(radians(self.ref_lat))*tan(radians(self.tar_lat))-sin(radians(self.ref_lat))*cos(fabs(radians(self.differ_lon))))/sin(fabs(radians(self.differ_lon)))
            if a == 0:
                a = 0.00001
            p = (atan(1 / a)) * 180 / pi
        # print("p:%s"%p)#算的这个数没有问题,这个是半圆周法
        # 求算经差，这个是为了后面的单位转换
        if self.differ_lon2 > 180:  # 求取经差的正负号
            self.differ_lon2 = -(360 - self.differ_lon2)
        elif self.differ_lon2 < -180:
            self.differ_lon2 = (360 + self.differ_lon2)
        if self.ref_lat2 >= 0:  # 转换为圆周法
            if self.differ_lon2 >= 0:
                if p > 0:
                    TB = p
                elif p < 0:
                    TB = 180 + p
            elif self.differ_lon2 < 0:
                if p > 0:
                    TB = 360 - p
                elif p < 0:
                    TB = 180 - p
        elif self.ref_lat2 < 0:
            if self.differ_lon2 >= 0:
                if p > 0:
                    TB = 180 - p
                elif p < 0:
                    TB = -p
            elif self.differ_lon2 < 0:
                if p > 0:
                    TB = 180 - p
                else:
                    TB = 360 - fabs(p)
        # print("TB:%s"%TB)
        # print("方位角为：%s" % (TB))
        return TB

    def cal_dcpa(self):
        if self.differ_cog >= 0:
            b = self.differ_cog
        else:
            b = 360 + self.differ_cog
        # print("b:%s"%b)
        a = self.tar_sog * self.tar_sog + pow(self.ref_sog, 2) - 2 * self.ref_sog * self.tar_sog * cos(radians(b))
        TB = self.true_bearing()
        d = fabs(TB - self.ref_cog)
        if d <= 180:
            Q = d
        elif d > 180:
            Q = 360 - d
        vx = sqrt(a)
        if self.ref_sog == 0 or self.tar_sog == 0:  # 避免船速为0
            self.ref_sog = 0.001
            self.tar_sog = 0.001
        if vx < 0.00001:  # 避免相对速度为0，导致程序出错
            vx = 0.0000001
        f = (pow(vx, 2) + pow(self.ref_sog, 2) - pow(self.tar_sog, 2)) / (2 * self.ref_sog * vx)

        alpha = acos(f) * 180 / pi  # 求相对速度的角
        # print("alpha:",alpha)
        D = self.dist()

        if b == 0:  # 如果他们相对航向为0
            dcpa = D
            tcpa = 0
        elif b <= 180:
            dcpa = D * sin(radians(fabs(Q - alpha)))
            tcpa = D * cos(radians(Q - alpha)) / vx
        else:
            dcpa = D * sin(radians(fabs(Q + alpha)))
            tcpa = D * cos(radians(Q + alpha)) / vx

        if vx < 0.000001:
            tcpa = 0  # 如果两船相对静止，让他们的最近会遇时间为0
        elif vx < 0.000001 and self.ref_sog < 0.0001:
            tcpa = 10000000
        # print("dcpa:%s,tcpa:%s"%(fabs(dcpa*60),tcpa*60*60))#dcpa的单位为分，也就是海里，tcpa的单位是分钟（先从度转化成分，再转成分钟）
        return dcpa, tcpa * 60
    
def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)
def seq_to_graph_TCPA(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    # V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))  # 【12，7，7】
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            # V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(len(step_)):
                if k != h:
                    # print(s,h,k)
                    z = step_.numpy().tolist()
                    ownship = refship(z[h][0], z[h][1], z[h][2], z[h][3])
                    targetship = tarship(z[k][0], z[k][1], z[k][2], z[k][3])
                    d = Cal(targetship, ownship)
                    # distance=d.dist()       #海里
                    # TB=d.true_bearing()     #圆周法，单位为：度
                    DCPA, TCPA = d.cal_dcpa()
                    if TCPA > 0:
                        A[s, h, k] = 1 / TCPA
                    else:
                        A[s, h, k] = 0
                    # A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(A).type(torch.float)

def seq_to_graph_ves_size(seq_,size_list, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    # V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    #print(size_list)# 【12，7，7】
    for s in range(seq_len):
        for h in range(len(size_list)):
            # V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, max_nodes):
                ownship = size_list[h]
                targetship = size_list[k]
                cos_sim = F.cosine_similarity(ownship, targetship, dim=0)
                A[s, h, k] = cos_sim
                A[s, k, h] = cos_sim
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()
    return torch.from_numpy(A).type(torch.float)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, data_dir_Mo, size_list_path, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True,
        end_Traj = 40,
        walk_length = 12,
        num_traj_samples = 5):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.data_dir_Mo = data_dir_Mo
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.walk_length = walk_length
        self.end_Traj = end_Traj
        self.num_traj_samples = num_traj_samples
        self.size_list=size_list_path
        size_list=torch.from_numpy(read_file(size_list_path, delim))
        
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        
        for path in all_files:
            data = read_file(path, delim)
            Trend,Trend_frame, Traj_index, G = create_graph_from_data(data)
            print('The trajectory graph has been constructed')
            start_nodes = Trend[self.obs_len-1:-self.end_Traj-self.pred_len]
            walk_length = self.walk_length+1 

            walks_dict = random_walk_sampling_for_list_fixed_length(G, start_nodes, walk_length, self.num_traj_samples)
            vessels = np.unique(data[:, 1]).tolist()
            vessel_node_emb = {}
            for vessel in vessels:
                data_vessel = data[data[:, 1] == vessel]
                vessel_node_emb[vessel] = generate_Node_embedding(Traj_index, data_vessel)

            Traj_feature = []
            pbar = tqdm(total=len(walks_dict))
            for index in walks_dict.keys():
                pbar.update(1)
                for vessel in vessels:
                    Node_embedding = vessel_node_emb[vessel] 
                    vessel_frame_features = []
                    walks = walks_dict[index]
                    walks_array = np.array(walks)
                    vessel_frame_features = np.array(Node_embedding)[walks_array[:,1:]]
                    Traj_feature.append(vessel_frame_features)
            pbar.close()
            Traj_feature = np.array(Traj_feature)
        all_files_Mo = os.listdir(self.data_dir_Mo)
        all_files_Mo = [os.path.join(self.data_dir_Mo, _path) for _path in all_files_Mo]
        
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0])[:-end_Traj].tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 4,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=8)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq[0:2, :], pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.A_obs_TCPA = []
        self.A_pred_TCPA = []
        self.A_obs_V_S = []
        self.A_pred_V_S = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]
                        
            a_ = seq_to_graph_TCPA(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
            self.A_obs_TCPA.append(a_.clone())
            a_ = seq_to_graph_TCPA(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :],
                                   self.norm_lap_matr)
            self.A_pred_TCPA.append(a_.clone())
            
            a_ = seq_to_graph_ves_size(self.obs_traj[start:end, :], size_list, self.norm_lap_matr)
            self.A_obs_V_S.append(a_.clone())
            a_ = seq_to_graph_ves_size(self.pred_traj[start:end, :], size_list, self.norm_lap_matr)
            self.A_pred_V_S.append(a_.clone())
        pbar.close()
        print("TCPA DCPA and Size have been completed")
        
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path_Mo in all_files_Mo:
            data = read_file(path_Mo, delim)
            frames = np.unique(data[:, 0])[:-end_Traj].tolist()                        
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        self.Traj_feature = torch.from_numpy(Traj_feature).type(torch.float)
        print(f"Final Traj_feature shape in init: {self.Traj_feature.shape}")
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()
        print("Social force has been completed")
        print('obs_traj',np.array(self.obs_traj).shape)
        print('pred_traj',np.array(self.pred_traj).shape)
        print('obs_traj_rel',np.array(self.obs_traj_rel).shape)
        print('pred_traj_rel',np.array(self.pred_traj_rel).shape)
        print('non_linear_ped',np.array(self.non_linear_ped).shape)
        print('loss_mask',np.array(self.loss_mask).shape)
        print('v_obs',np.array(self.v_obs).shape)
        print('A_obs',np.array(self.A_obs).shape)
        print('v_pred',np.array(self.v_pred).shape)
        print('A_pred',np.array(self.A_pred).shape)
        print('self.A_obs_TCPA',np.array(self.A_obs_TCPA).shape)
        print('self.A_pred_TCPA',np.array(self.A_pred_TCPA).shape)
        print('self.A_obs_V_S',np.array(self.A_obs_V_S).shape)
        print('self.A_pred_V_S',np.array(self.A_pred_V_S).shape)
        print('Traj_feature',np.array(self.Traj_feature).shape)
    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],
            self.A_obs_TCPA[index], self.A_pred_TCPA[index],
            self.A_obs_V_S[index], self.A_pred_V_S[index],
            self.Traj_feature[start:end, :]
        ]
        return out
