import scipy.io as sio
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils import *
import argparse


class HUTUBS(Dataset):
    def __init__(self, args, val):
        super(HUTUBS, self).__init__()
        valid_hrtf_index = list(range(0, 17)) + list(range(18,78)) + list(range(79,91)) + list(range(92,96))
        anthro = pd.read_csv(args.anthro_mat_path)
        self.val = val
        self.norm_anthro = args.norm_anthro

        self.anthro_mat = np.array(anthro)[np.array(valid_hrtf_index), 1:].astype(np.float64)

        self.anthro_mat_val = self.anthro_mat[[args.val_idx]]
        self.anthro_mat_train = np.delete(self.anthro_mat, args.val_idx, axis=0)

        if self.norm_anthro:
            anthro_avg = np.mean(self.anthro_mat_train, axis=0)
            anthro_std = np.std(self.anthro_mat_train, axis=0)
            self.anthro_mat_train = self.normalize(args.anthro_norm_method, self.anthro_mat_train, anthro_avg, anthro_std)
            self.anthro_mat_val = self.normalize(args.anthro_norm_method, self.anthro_mat_val, anthro_avg, anthro_std)

        self.anthro_mat_X_train = self.anthro_mat_train[:, :13]
        self.anthro_mat_D_L_train = self.anthro_mat_train[:, 13:25]
        self.anthro_mat_D_R_train = self.anthro_mat_train[:, 25:]

        self.anthro_mat_X_val = self.anthro_mat_val[:, :13]
        self.anthro_mat_D_L_val = self.anthro_mat_val[:, 13:25]
        self.anthro_mat_D_R_val = self.anthro_mat_val[:, 25:]
        hrtf_mat = np.expand_dims(sio.loadmat(args.hrtf_SHT_mat_path)["HRTF_dBmat"], -2)
        self.hrtf_mat = hrtf_mat[valid_hrtf_index]

        self.hrtf_mat_val = self.hrtf_mat[[args.val_idx]]
        self.hrtf_mat_train = np.delete(self.hrtf_mat, args.val_idx, axis=0)
        sht_mat = np.expand_dims(sio.loadmat(args.hrtf_SHT_mat_path)["SHT_dBmat"], -2)

        self.sht_mat = sht_mat[valid_hrtf_index]

        self.sht_mat_val = self.sht_mat[[args.val_idx]]
        self.sht_mat_train = np.delete(self.sht_mat, args.val_idx, axis=0)

    def normalize(self, norm_method, anthro, avg, std):
        if norm_method == "standard":
            return (anthro - avg) / std
        elif norm_method == "chun2017":
            return np.reciprocal(1 + np.exp((anthro - avg) / std))
        else:
            raise ValueError("anthropometric normalization method not recognized")

    def __len__(self):
        if self.val:
            return self.hrtf_mat_val.shape[0]*self.hrtf_mat_val.shape[1]*2
        else:
            return self.hrtf_mat_train.shape[0]*self.hrtf_mat_train.shape[1]*2

    def __getitem__(self, idx):
        if self.val:
            left_or_right = idx // (self.anthro_mat_X_val.shape[0]*self.hrtf_mat_val.shape[1])
            new_idx = idx % (self.anthro_mat_X_val.shape[0]*self.hrtf_mat_val.shape[1])
            freq = new_idx // self.anthro_mat_X_val.shape[0]
            subject = new_idx % self.anthro_mat_X_val.shape[0]
            if left_or_right == 0:
                ear_anthro = self.anthro_mat_D_L_val[subject]
            else:
                ear_anthro = self.anthro_mat_D_R_val[subject]
            head_anthro = self.anthro_mat_X_val[subject]
            hrtf = self.hrtf_mat_val[subject, freq, :, :, left_or_right]
            sht = self.sht_mat_val[subject, freq, :, :, left_or_right]
        else:
            left_or_right = idx // (self.anthro_mat_X_train.shape[0] * self.hrtf_mat_train.shape[1])
            new_idx = idx % (self.anthro_mat_X_train.shape[0] * self.hrtf_mat_train.shape[1])
            freq = new_idx // self.anthro_mat_X_train.shape[0]
            subject = new_idx % self.anthro_mat_X_train.shape[0]
            if left_or_right == 0:
                ear_anthro = self.anthro_mat_D_L_train[subject]
            else:
                ear_anthro = self.anthro_mat_D_R_train[subject]
            head_anthro = self.anthro_mat_X_train[subject]
            hrtf = self.hrtf_mat_train[subject, freq, :, :, left_or_right]
            sht = self.sht_mat_train[subject, freq, :, :, left_or_right]
        return ear_anthro, head_anthro, hrtf, sht, subject, freq, left_or_right


class HUTUBS_SCH(Dataset):
    def __init__(self, args, val=False):
        super(HUTUBS_SCH, self).__init__()
        sch_mat_file = sio.loadmat(args.mesh_SCH_mat_path)
        valid_hrtf_index = sch_mat_file["mesh_ind"].squeeze(-1)
        self.val = val

        anthro = pd.read_csv(args.anthro_mat_path)
        self.norm_anthro = args.norm_anthro

        self.anthro_mat = np.array(anthro)[np.array(valid_hrtf_index-1), 1:].astype(np.float64)

        self.anthro_mat_val = self.anthro_mat[[args.val_idx]]
        self.anthro_mat_train = np.delete(self.anthro_mat, args.val_idx, axis=0)

        if self.norm_anthro:
            anthro_avg = np.mean(self.anthro_mat_train, axis=0)
            anthro_std = np.std(self.anthro_mat_train, axis=0)
            self.anthro_mat_train = self.normalize(args.anthro_norm_method, self.anthro_mat_train, anthro_avg,
                                                   anthro_std)
            self.anthro_mat_val = self.normalize(args.anthro_norm_method, self.anthro_mat_val, anthro_avg, anthro_std)

        self.anthro_mat_X_train = self.anthro_mat_train[:, :13]
        self.anthro_mat_D_L_train = self.anthro_mat_train[:, 13:25]
        self.anthro_mat_D_R_train = self.anthro_mat_train[:, 25:]

        self.anthro_mat_X_val = self.anthro_mat_val[:, :13]
        self.anthro_mat_D_L_val = self.anthro_mat_val[:, 13:25]
        self.anthro_mat_D_R_val = self.anthro_mat_val[:, 25:]

        self.use_schSquare = args.use_schSquare
        if args.use_schSquare:
            self.ear_sch_mat = sch_mat_file["ear_SCH_all_squared"]
        else:
            self.ear_sch_mat = sch_mat_file["ear_SCH_all"]

        self.ear_sch_mat_val = self.ear_sch_mat[[args.val_idx]]
        self.ear_sch_mat_train = np.delete(self.ear_sch_mat, args.val_idx, axis=0)

        hrtf_mat = np.expand_dims(sio.loadmat(args.hrtf_SHT_mat_path)["hrtf_freq_allDB"], -2)
        self.hrtf_mat = hrtf_mat[valid_hrtf_index-1] # because matlab start from 1

        if args.use_logFreq:
            logind = sio.loadmat(args.hrtf_SHT_mat_path)["freq_logind"]
            logind = logind.squeeze(0)
            self.hrtf_mat = self.hrtf_mat[:, logind, :, :, :]
        else:
            self.hrtf_mat = self.hrtf_mat[:, :100, :, :, :]
            # print(self.hrtf_mat.shape)
            # exit()

        self.hrtf_mat_val = self.hrtf_mat[[args.val_idx]]
        self.hrtf_mat_train = np.delete(self.hrtf_mat, args.val_idx, axis=0)

        # sht_mat = np.expand_dims(sio.loadmat(args.hrtf_SHT_mat_path)["hrtf_SHT_dBmat"], -2)
        sht_mat = np.expand_dims(sio.loadmat(args.hrtf_SHT_mat_path)["hrtf_SHT_dBmat_kaCorrect"], -2)
        self.sht_mat = sht_mat[valid_hrtf_index-1]
        # print(self.sht_mat.shape)
        # exit()
        if args.use_logFreq:
            sht_mat = np.expand_dims(sio.loadmat(args.hrtf_SHT_mat_path)["hrtf_SHT_dBmat_logFreq"], -2)
            self.sht_mat = sht_mat[valid_hrtf_index - 1]
            # print(self.sht_mat.shape)
            # exit()
        else:
            self.sht_mat = self.sht_mat[:, :100, :, :, :]

        # print(self.sht_mat[..., 0] - self.sht_mat[..., 1])

        self.sht_mat_val = self.sht_mat[[args.val_idx]]
        self.sht_mat_train = np.delete(self.sht_mat, args.val_idx, axis=0)

        # print(self.hrtf_mat.shape)
        # print(self.sht_mat.shape)

    def normalize(self, norm_method, anthro, avg, std):
        if norm_method == "standard":
            return (anthro - avg) / std
        elif norm_method == "chun2017":
            return np.reciprocal(1 + np.exp((anthro - avg) / std))
        else:
            raise ValueError("anthropometric normalization method not recognized")

    def __len__(self):
        if self.val:
            return self.hrtf_mat_val.shape[0]*self.hrtf_mat_val.shape[1]*2
        else:
            return self.hrtf_mat_train.shape[0]*self.hrtf_mat_train.shape[1]*2

    def __getitem__(self, idx):
        if self.val:
            left_or_right = idx // (self.ear_sch_mat_val.shape[0]*self.hrtf_mat_val.shape[1])
            new_idx = idx % (self.ear_sch_mat_val.shape[0]*self.hrtf_mat_val.shape[1])
            freq = new_idx // self.ear_sch_mat_val.shape[0]
            subject = new_idx % self.ear_sch_mat_val.shape[0]
            ear_sch = self.ear_sch_mat_val[subject, left_or_right]
            hrtf = self.hrtf_mat_val[subject, freq, :, :, left_or_right]
            sht = self.sht_mat_val[subject, freq, :, :, left_or_right]
            if left_or_right == 0:
                ear_anthro = self.anthro_mat_D_L_val[subject]
            else:
                ear_anthro = self.anthro_mat_D_R_val[subject]
            head_anthro = self.anthro_mat_X_val[subject]
        else:
            left_or_right = idx // (self.ear_sch_mat_train.shape[0] * self.hrtf_mat_train.shape[1])
            new_idx = idx % (self.ear_sch_mat_train.shape[0] * self.hrtf_mat_train.shape[1])
            freq = new_idx // self.ear_sch_mat_train.shape[0]
            subject = new_idx % self.ear_sch_mat_train.shape[0]
            ear_sch = self.ear_sch_mat_train[subject, left_or_right]
            # print(self.hrtf_mat_train.shape)
            hrtf = self.hrtf_mat_train[subject, freq, :, :, left_or_right]
            sht = self.sht_mat_train[subject, freq, :, :, left_or_right]
            if left_or_right == 0:
                ear_anthro = self.anthro_mat_D_L_train[subject]
            else:
                ear_anthro = self.anthro_mat_D_R_train[subject]
            head_anthro = self.anthro_mat_X_train[subject]
        if self.use_schSquare:
            ear_sch = ear_sch.transpose(2, 0, 1)
        else:
            ear_sch = ear_sch.T
        return ear_anthro, head_anthro, ear_sch, hrtf, sht, subject, freq, left_or_right


class HRTF_onset(Dataset):
    def __init__(self, args, val=False):
        super(HRTF_onset, self).__init__()
        sch_mat_file = sio.loadmat(args.mesh_SCH_mat_path)
        valid_hrtf_index = sch_mat_file["mesh_ind"].squeeze(-1)
        self.val = val
        # valid_hrtf_index = list(range(0, 17)) + list(range(18, 78)) + list(range(79, 91)) + list(range(92, 96))
        anthro = pd.read_csv(args.anthro_mat_path)
        self.norm_anthro = args.norm_anthro

        self.anthro_mat = np.array(anthro)[valid_hrtf_index-1, 1:].astype(np.float64)

        self.anthro_mat_val = self.anthro_mat[[args.val_idx]]
        self.anthro_mat_train = np.delete(self.anthro_mat, args.val_idx, axis=0)

        if self.norm_anthro:
            anthro_avg = np.mean(self.anthro_mat_train, axis=0)
            anthro_std = np.std(self.anthro_mat_train, axis=0)
            self.anthro_mat_train = self.normalize(args.anthro_norm_method, self.anthro_mat_train, anthro_avg,
                                                   anthro_std)
            self.anthro_mat_val = self.normalize(args.anthro_norm_method, self.anthro_mat_val, anthro_avg, anthro_std)

        self.anthro_mat_X_train = self.anthro_mat_train[:, :13]
        self.anthro_mat_D_L_train = self.anthro_mat_train[:, 13:25]
        self.anthro_mat_D_R_train = self.anthro_mat_train[:, 25:]

        self.anthro_mat_X_val = self.anthro_mat_val[:, :13]
        self.anthro_mat_D_L_val = self.anthro_mat_val[:, 13:25]
        self.anthro_mat_D_R_val = self.anthro_mat_train[:, 25:]

        onset_mat = sio.loadmat(args.hrtf_SHT_mat_path)["input_hrir_onset"]
        self.onset_mat = onset_mat[valid_hrtf_index-1]
        onset_shtmat = sio.loadmat(args.hrtf_SHT_mat_path)["HRTF_onset_SHT"]
        self.onset_shtmat = onset_shtmat[valid_hrtf_index - 1]

        self.onset_mat_val = self.onset_mat[[args.val_idx]]
        self.onset_mat_train = np.delete(self.onset_mat, args.val_idx, axis=0)
        self.onset_shtmat_val = self.onset_shtmat[[args.val_idx]]
        self.onset_shtmat_train = np.delete(self.onset_shtmat, args.val_idx, axis=0)

    def normalize(self, norm_method, anthro, avg, std):
        if norm_method == "standard":
            return (anthro - avg) / std
        elif norm_method == "chun2017":
            return np.reciprocal(1 + np.exp((anthro - avg) / std))
        else:
            raise ValueError("anthropometric normalization method not recognized")

    def __len__(self):
        if self.val:
            return self.onset_mat_val.shape[0]*2
        else:
            return self.onset_mat_train.shape[0]*2

    def __getitem__(self, idx):
        if self.val:
            left_or_right = idx // self.onset_mat_val.shape[0]
            subject = idx % self.onset_mat_val.shape[0]
            if left_or_right == 0:
                ear_anthro = self.anthro_mat_D_L_val[subject]
            else:
                ear_anthro = self.anthro_mat_D_R_val[subject]
            head_anthro = self.anthro_mat_X_val[subject]
            onset = self.onset_mat_val[subject, :, left_or_right]
            onset_sht = self.onset_shtmat_val[subject, :, left_or_right]
        else:
            left_or_right = idx // self.onset_mat_train.shape[0]
            subject = idx % self.onset_mat_train.shape[0]
            if left_or_right == 0:
                ear_anthro = self.anthro_mat_D_L_train[subject]
            else:
                ear_anthro = self.anthro_mat_D_R_train[subject]
            head_anthro = self.anthro_mat_X_train[subject]
            onset = self.onset_mat_train[subject, :, left_or_right]
            onset_sht = self.onset_shtmat_train[subject, :, left_or_right]

        return ear_anthro, head_anthro, onset_sht, onset, subject, left_or_right



def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    # Dataset parameters
    parser.add_argument("-s", "--mesh_SCH_mat_path", type=str,
                        default='/data3/neil/HRTF/ear_SCH_all_lite.mat')
    parser.add_argument("-a", "--anthro_mat_path", type=str,
                        default='/data3/neil/HRTF/AntrhopometricMeasures.csv')
    parser.add_argument("-t", "--hrtf_SHT_mat_path", type=str,
                        default='/data3/neil/HRTF/HUTUBS_matrix_measured.mat')
    parser.add_argument("-v", "--shvec_path", type=str,
                        default='/data3/neil/HRTF/SH_vec_matrix.mat')
    parser.add_argument("-i", "--val_idx", type=int, default=0, help="index for Leave-one-out validation")
    parser.add_argument("--norm_anthro", type=str2bool, nargs='?', const=True, default=True,
                        help="whether to normalize anthro measures.")
    parser.add_argument('--anthro_norm_method', type=str, default='chun2017',
                        choices=['standard', 'chun2017'],
                        help="normalization method for input anthropometric measurements")
    parser.add_argument("--use_logFreq", type=str2bool, nargs='?', const=True, default=True,
                        help="whether to use log freq index.")
    # Dataset prepare
    parser.add_argument("--ear_anthro_dim", type=int, help="ear anthro dimension", default=12)
    parser.add_argument("--use_schSquare", type=str2bool, nargs='?', const=True, default=True,
                        help="whether to use SCH square.")
    parser.add_argument("--head_anthro_dim", type=int, help="head anthro dimension", default=13)
    parser.add_argument("--freq_bin", type=int, help="number of frequency bin", default=41)

    parser.add_argument("--gpu", type=str, help="GPU index", default="1")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.shvec = torch.from_numpy(sio.loadmat(args.shvec_path)["SH_Vec_matrix"])
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

if __name__ == "__main__":
    args = initParams()
    dataset = HUTUBS_SCH(args, val=False)
    print(len(dataset))
    ear_anthro, head_anthro, ear_sch, hrtf, sht, subject, freq, left_or_right = dataset[4234]
    print(ear_sch.shape)
    print(hrtf.shape)
    print(sht.shape)
    print(freq)
    print(left_or_right)

    # dataset = HRTF_onset(args, val=False)
    # print(len(dataset))
    # ear_anthro, head_anthro, onset, subject, left_or_right = dataset[16]
    # print(ear_anthro.shape)
    # print(head_anthro.shape)
    # print(onset.shape)
    # print(subject)
    # print(left_or_right)

