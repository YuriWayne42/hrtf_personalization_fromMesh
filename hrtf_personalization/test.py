import torch
import scipy.io as sio
import numpy as np
from tqdm import tqdm, trange
import random
import os

num_of_subjects = 53

# for j in tqdm(range(num_of_subjects)):
#     os.system("python train_.py -o /data3/neil/HRTF/models1130/hrtf_sht%02d --use_logFreq no --freq_bin 100 -i %d --gpu %d" %(j, j, (j % 2)*3))

model_dir = "/data3/neil/HRTF/models1128/hrtf_sht"
hrtf_SHT_mat_path = "/data3/neil/HRTF/HUTUBS_matrix_measured.mat"
shvec_path = "/data3/neil/HRTF/SH_vec_matrix.mat"
freqind = torch.from_numpy(sio.loadmat(hrtf_SHT_mat_path)["freq_logind"].squeeze(0))

shvec = torch.from_numpy(sio.loadmat(shvec_path)["SH_Vec_matrix"])
shvec = shvec.float().unsqueeze(0).repeat(freqind.shape[0], 1, 1)
# shvec = shvec.float().unsqueeze(0).repeat(100, 1, 1)

freqind = (freqind-1).tolist()

def mse(first, second):
    return (first.float() - second.float()) **2

def calLSD(predicted_hrtf, smoothed_hrtf, actual_hrtf):
    # compare with actual hrtf
    recon_lsd = mse(predicted_hrtf, actual_hrtf)
    # frontal direction, compare with actual
    recon_lsd00 = mse(predicted_hrtf[:, 202, :], actual_hrtf[:, 202, :])
    # compare with SHT reconstructed
    lsd_recon = mse(predicted_hrtf, smoothed_hrtf)
    # frontal direction, compare with SHT reconstructed
    lsd_recon00 = mse(predicted_hrtf[:, 202, :], smoothed_hrtf[:, 202, :])
    # leftmost direction, compare with SHT reconstructed
    lsd_reconleft = mse(predicted_hrtf[:, 211, :], smoothed_hrtf[:, 211, :])
    # rightmost direction, compare with SHT reconstructed
    lsd_reconright = mse(predicted_hrtf[:, 229, :], smoothed_hrtf[:, 229, :])

    return recon_lsd, recon_lsd00, lsd_recon, lsd_recon00, lsd_reconleft, lsd_reconright

all_res = []
for i in tqdm(range(num_of_subjects)):
    result_mat_path = os.path.join(model_dir + "%02d" % i, "result_%02d.mat" % i)
    gt_sht = torch.from_numpy(sio.loadmat(result_mat_path)["sht_array"])[freqind, ...]
    gen_sht = torch.from_numpy(sio.loadmat(result_mat_path)["gen_sht_array"])[freqind, ...]
    predicted_hrtf = torch.bmm(shvec, gen_sht)
    smoothed_hrtf = torch.bmm(shvec, gt_sht)
    actual_hrtf = torch.from_numpy(sio.loadmat(hrtf_SHT_mat_path)["hrtf_freq_allDB"][:, freqind, ...])[i]
    # actual_hrtf = torch.from_numpy(sio.loadmat(hrtf_SHT_mat_path)["hrtf_freq_allDB"][:, :100, ...])[i]
    all_res.append(calLSD(predicted_hrtf, smoothed_hrtf, actual_hrtf))

recon_lsd_lst, recon_lsd00_lst, lsd_recon_lst, lsd_recon00_lst = [], [], [], []
lsd_reconleft_lst, lsd_reconright_lst = [], []
for (recon_lsd, recon_lsd00, lsd_recon, lsd_recon00, lsd_reconleft, lsd_reconright) in all_res:
    recon_lsd_lst.append(recon_lsd)
    recon_lsd00_lst.append(recon_lsd00)
    lsd_recon_lst.append(lsd_recon)
    lsd_recon00_lst.append(lsd_recon00)
    lsd_reconleft_lst.append(lsd_reconleft)
    lsd_reconright_lst.append(lsd_reconright)
recon_lsd_lst_ = torch.stack(recon_lsd_lst)
recon_lsd00_lst_ = torch.stack(recon_lsd00_lst)
lsd_recon_lst_ = torch.stack(lsd_recon_lst)
lsd_recon00_lst_ = torch.stack(lsd_recon00_lst)
lsd_reconleft_lst_ = torch.stack(lsd_reconleft_lst)
lsd_reconright_lst_ = torch.stack(lsd_reconright_lst)

print("Compare with smoothed HRTF", np.sqrt(np.mean(np.array(lsd_recon_lst_))))

print("Compare with smoothed HRTF in frontal direction", np.sqrt(np.mean(np.array(lsd_recon00_lst_))))

print("Compare with smoothed HRTF in leftmost direction", np.sqrt(np.mean(np.array(lsd_reconleft_lst_))))

print("Compare with smoothed HRTF in rightmost direction", np.sqrt(np.mean(np.array(lsd_reconright_lst_))))

print("Compare with measured HRTF", np.sqrt(np.mean(np.array(recon_lsd_lst_))))

print("Compare with measured HRTF in frontal direction", np.sqrt(np.mean(np.array(recon_lsd00_lst_))))

