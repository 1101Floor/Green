import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from metrics import * 
from model import GREEN
import copy

def test(KSTEPS=20):
    global loader_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0 
    for batch in loader_test: 
        step+=1
        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr, A_TPCA_obs,\
        A_TPCA_tr, A_vs_obs, A_vs_tr, FF = batch


        num_of_objs = obs_traj_rel.shape[1]
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model.inference(V_obs_tmp, A_obs.squeeze(),A_TPCA_obs.squeeze(), A_vs_obs.squeeze(), FF)
        V_pred = V_pred.permute(0,2,3,1)


        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]

        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).to(device)
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2].to(device)
        mvnormal = torchdist.MultivariateNormal(mean,cov)

        ade_ls = {}
        fde_ls = {}
        #print('obs_traj.data.cpu().numpy()',obs_traj.data.cpu().numpy().shape,obs_traj.data.cpu().numpy())
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()



            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
           # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))
        
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)
    return ade_,fde_,raw_data_dict


paths = ['./checkpoint/*social-stgcnn*']
KSTEPS=20

print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)
ade_ls = [] 
fde_ls = [] 
path = paths[-1]
exps = './checkpoint/tag_1/'
print('Model being tested are:',exps)

exp_path = exps
print("*"*50)
print("Evaluating model:",exp_path)

model_path = exp_path+'/val_best.pth'
args_path = exp_path+'/args.pkl'
with open(args_path,'rb') as f: 
    args = pickle.load(f)

stats= exp_path+'/constant_metrics.pkl'
with open(stats,'rb') as f: 
    cm = pickle.load(f)
print("Stats:",cm)

obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
end_Traj = args.end_Traj
walk_length = args.pred_seq_len
num_traj_samples = args.num_traj_samples
data_set_Mo = './datasets/' + args.dataset_Mo + '/'
data_set = './datasets/' + args.dataset + '/'

dset_test = TrajectoryDataset(
        data_dir =  data_set + 'test/',
        data_dir_Mo = data_set_Mo + 'test/',
        size_list_path = size_list_path,
        obs_len = obs_seq_len,
        pred_len = pred_seq_len,
        end_Traj = end_Traj,
        walk_length = pred_seq_len,
        num_traj_samples = num_traj_samples,
        skip=1,norm_lap_matr=True)
torch.save(dset_test, "./datasets/dset_test.pt")
dset_test = torch.load("./datasets/dset_test.pt")
loader_test = DataLoader(
        dset_test,
        batch_size=1,#This is irrelative to the args batch size parameter
        shuffle =False,
        num_workers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Defining the model 
model = GREEN(n_stgcnn=args.n_stgcnn, output_feat=args.output_size, seq_len=args.obs_seq_len,
                      kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len, 
                      embed_size=args.embed_size, dim_z=args.dim_z, z_feature = args.z_feature, num_channels = args.num_channels).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))


ade_ =999999
fde_ =999999
print("Testing ....")
ad,fd,raw_data_dic_= test()
ade_= min(ade_,ad)
fde_ =min(fde_,fd)
ade_ls.append(ade_)
fde_ls.append(fde_)
print("ADE:",ade_," FDE:",fde_)

print("*"*50)

print("Avg ADE:",sum(ade_ls)/5)
print("Avg FDE:",sum(fde_ls)/5)