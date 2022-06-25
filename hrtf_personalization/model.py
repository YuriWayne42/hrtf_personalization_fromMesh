import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from utils import str2bool
import argparse

class EarMeasureEncoder(nn.Module):
    def __init__(self, ear_anthro_dim, ear_emb_dim):
        super(EarMeasureEncoder, self).__init__()
        self.ear_anthro_dim = ear_anthro_dim
        self.fc = nn.Sequential(
            nn.Linear(ear_anthro_dim, ear_emb_dim),
        )

    def forward(self, ear_anthro):
        assert ear_anthro.shape[1] == self.ear_anthro_dim
        return self.fc(ear_anthro)

class HeadMeasureEncoder(nn.Module):
    def __init__(self, head_anthro_dim, head_emb_dim):
        super(HeadMeasureEncoder, self).__init__()
        self.head_anthro_dim = head_anthro_dim
        self.fc = nn.Sequential(
            nn.Linear(head_anthro_dim, head_emb_dim),
        )

    def forward(self, head_anthro):
        assert head_anthro.shape[1] == self.head_anthro_dim
        return self.fc(head_anthro)


class ConvNNHrtfSht(nn.Module):
    def __init__(self, args):
        super(ConvNNHrtfSht, self).__init__()
        self.ear_enc = EarMeasureEncoder(args.ear_anthro_dim, args.ear_emb_dim)
        self.head_enc = HeadMeasureEncoder(args.head_anthro_dim, args.head_emb_dim)
        self.lr_enc = nn.Embedding(2, args.lr_emb_dim)
        self.freq_enc = nn.Embedding(args.freq_bin, args.freq_emb_dim)
        self.condition_dim = args.condition_dim
        emb_concat_dim = args.ear_emb_dim + args.head_emb_dim + args.freq_emb_dim
        emb_concat_dim += args.lr_emb_dim
        self.fc = nn.Linear(emb_concat_dim, args.condition_dim)
        self.norm = args.norm
        if self.norm == "batch":
            self.norm_method = nn.BatchNorm1d
        elif self.norm == "layer":
            self.norm_method = nn.GroupNorm
        elif self.norm == "instance":
            self.norm_method = nn.InstanceNorm1d
        else:
            raise ValueError("normalization method not recognized")

        self.conv1 = self.make_gen_block(1, 4, kernel_size=7, stride=3)
        self.conv2 = self.make_gen_block(4, 16, kernel_size=5, stride=2)
        self.conv3 = self.make_gen_block(16, 32, kernel_size=5, stride=2)
        self.conv4 = self.make_gen_block(32, 32, kernel_size=5, stride=3)
        if args.target == "hrtf":
            self.conv5 = self.make_gen_block(32, 440, kernel_size=5, stride=2, final_layer=True)
        elif args.target == "sht":
            self.conv5 = self.make_gen_block(32, 64, kernel_size=5, stride=2, final_layer=True)
        else:
            raise ValueError("training target not recognized")

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            if self.norm == "layer":
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(1, output_channels),
                    nn.ReLU()
                )
            else:
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(output_channels),
                    nn.ReLU()
                )
        else:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
            )

    def unsqueeze_condition(self, latent):
        return latent.view(len(latent), 1, self.condition_dim)

    def forward(self, ear_anthro, head_anthro, frequency, left_or_right):
        ear_anthro_encoding = self.ear_enc(ear_anthro)
        head_anthro_encoding = self.head_enc(head_anthro)
        frequency_encoding = self.freq_enc(frequency)
        left_or_right_enc = self.lr_enc(left_or_right)

        latent = torch.cat((ear_anthro_encoding, head_anthro_encoding,
                            frequency_encoding), dim=1)
        latent = torch.cat((latent, left_or_right_enc), dim=1)
        latent = self.unsqueeze_condition(self.fc(latent))

        latent = self.conv1(latent)
        latent = self.conv2(latent)
        latent = self.conv3(latent)
        latent = self.conv4(latent)
        out = self.conv5(latent)

        return out


class EarSCHEncoder(nn.Module):
    def __init__(self, args):
        super(EarSCHEncoder, self).__init__()
        self.sch_channel = args.ear_sch_ch
        self.sch_dim = args.ear_sch_dim
        self.norm = args.norm
        if self.norm == "batch":
            self.norm_method = nn.BatchNorm1d
        elif self.norm == "layer":
            self.norm_method = nn.GroupNorm
        elif self.norm == "instance":
            self.norm_method = nn.InstanceNorm1d
        else:
            raise ValueError("normalization method not recognized")
        self.conv1 = self.make_gen_block(3, 8, kernel_size=5, stride=2)
        self.conv2 = self.make_gen_block(8, 16, kernel_size=5, stride=3)
        self.conv3 = self.make_gen_block(16, 32, kernel_size=5, stride=3)
        self.conv4 = self.make_gen_block(32, 32, kernel_size=5, stride=2)
        self.conv5 = self.make_gen_block(32, 64, kernel_size=5, stride=2)
        self.conv6 = self.make_gen_block(64, 64, kernel_size=3, stride=2, final_layer=True)
        self.fc = nn.Sequential(
            nn.Linear(64, args.ear_sch_emb_dim),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            if self.norm == "layer":
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(1, output_channels),
                    nn.ReLU()
                )
            else:
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(output_channels),
                    nn.ReLU()
                )
        else:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, ear_sch):
        assert ear_sch.shape[1] == self.sch_channel
        assert ear_sch.shape[2] == self.sch_dim
        h = self.conv1(ear_sch)
        # print(h.shape)
        h = self.conv2(h)
        # print(h.shape)
        h = self.conv3(h)
        # print(h.shape)
        h = self.conv4(h)
        # print(h.shape)
        h = self.conv5(h)
        # print(h.shape)
        h = self.conv6(h)
        # print(h.shape)
        return self.fc(h.squeeze())
        # return torch.randn(ear_sch.shape[0], 64)


class EarSCHEncoder2(nn.Module):
    def __init__(self, args):
        super(EarSCHEncoder2, self).__init__()
        self.sch_channel = args.ear_sch_ch
        self.sch_dim = args.ear_sch_dim
        self.norm = args.norm
        if self.norm == "batch":
            self.norm_method = nn.BatchNorm1d
        elif self.norm == "layer":
            self.norm_method = nn.GroupNorm
        elif self.norm == "instance":
            self.norm_method = nn.InstanceNorm1d
        else:
            raise ValueError("normalization method not recognized")
        self.conv1 = self.make_gen_block(3, 8, kernel_size=5, stride=2)
        self.conv2 = self.make_gen_block(8, 16, kernel_size=5, stride=3)
        self.conv3 = self.make_gen_block(16, 32, kernel_size=5, stride=3)
        self.conv4 = self.make_gen_block(32, 32, kernel_size=5, stride=2)
        self.conv5 = self.make_gen_block(32, 64, kernel_size=5, stride=2)
        self.conv6 = self.make_gen_block(64, 64, kernel_size=3, stride=2, final_layer=True)
        self.fc = nn.Sequential(
            nn.Linear(64, args.ear_sch_emb_dim),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            if self.norm == "layer":
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(1, output_channels),
                    nn.ReLU()
                )
            else:
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(output_channels),
                    nn.ReLU()
                )
        else:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, ear_sch):
        assert ear_sch.shape[1] == self.sch_channel
        assert ear_sch.shape[2] == self.sch_dim
        h = self.conv1(ear_sch)
        # print(h.shape)
        h = self.conv2(h)
        # print(h.shape)
        h = self.conv3(h)
        # print(h.shape)
        h = self.conv4(h)
        # print(h.shape)
        h = self.conv5(h)
        # print(h.shape)
        h = self.conv6(h)
        # print(h.shape)
        return self.fc(h.squeeze())


class EarSCHSquareEncoder(nn.Module):
    def __init__(self, args):
        super(EarSCHSquareEncoder, self).__init__()
        self.sch_channel = args.ear_sch_ch
        self.sch_dim = args.ear_sch_dim
        self.norm = args.norm
        if self.norm == "batch":
            self.norm_method = nn.BatchNorm1d
        elif self.norm == "layer":
            self.norm_method = nn.GroupNorm
        elif self.norm == "instance":
            self.norm_method = nn.InstanceNorm1d
        else:
            raise ValueError("normalization method not recognized")
        self.conv1 = self.make_gen_block(3, 8, stride=2, padding=1)
        self.conv2 = self.make_gen_block(8, 16, stride=2)
        self.conv3 = self.make_gen_block(16, 32, stride=1, padding=0)
        self.conv4 = self.make_gen_block(32, 32, kernel_size=3, stride=1)
        self.conv5 = self.make_gen_block(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv6 = self.make_gen_block(64, 64, kernel_size=3, stride=2, final_layer=True)
        self.fc = nn.Sequential(
            nn.Linear(64, args.ear_sch_emb_dim),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1, final_layer=False):
        if not final_layer:
            if self.norm == "layer":
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                    self.norm_method(1, output_channels),
                    nn.ReLU()
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                    self.norm_method(output_channels),
                    nn.ReLU()
                )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            )

    def forward(self, ear_sch):
        assert ear_sch.shape[1] == self.sch_channel
        # print(self.sch_dim)
        # print(ear_sch.shape)
        assert ear_sch.shape[2] == np.sqrt(self.sch_dim)
        h = self.conv1(ear_sch)
        # print(h.shape)
        h = self.conv2(h)
        # print(h.shape)
        h = self.conv3(h)
        # print(h.shape)
        h = self.conv4(h)
        # print(h.shape)
        h = self.conv5(h)
        # print(h.shape)
        h = self.conv6(h)
        # print(h.shape)
        return self.fc(h.squeeze())


class ConvNNEarSch2HrtfSht(nn.Module):
    def __init__(self, args):
        super(ConvNNEarSch2HrtfSht, self).__init__()
        if args.use_schSquare:
            self.ear_enc = EarSCHSquareEncoder(args)
        else:
            self.ear_enc = EarSCHEncoder(args)
        self.head_enc = HeadMeasureEncoder(args.head_anthro_dim, args.head_emb_dim)
        self.lr_enc = nn.Embedding(2, args.lr_emb_dim)
        self.freq_enc = nn.Embedding(args.freq_bin, args.freq_emb_dim)
        self.condition_dim = args.condition_dim
        emb_concat_dim = args.ear_sch_emb_dim + args.freq_emb_dim + args.head_emb_dim
        emb_concat_dim += args.lr_emb_dim
        self.fc = nn.Linear(emb_concat_dim, args.condition_dim)
        self.norm = args.norm
        if self.norm == "batch":
            self.norm_method = nn.BatchNorm1d
        elif self.norm == "layer":
            self.norm_method = nn.GroupNorm
        elif self.norm == "instance":
            self.norm_method = nn.InstanceNorm1d
        else:
            raise ValueError("normalization method not recognized")

        self.conv1 = self.make_gen_block(1, 4, kernel_size=7, stride=3)
        self.conv2 = self.make_gen_block(4, 16, kernel_size=5, stride=2)
        self.conv3 = self.make_gen_block(16, 32, kernel_size=5, stride=2)
        self.conv4 = self.make_gen_block(32, 32, kernel_size=5, stride=3)
        if args.target == "hrtf":
            self.conv5 = self.make_gen_block(32, 440, kernel_size=5, stride=2, final_layer=True)
        elif args.target == "sht":
            self.conv5 = self.make_gen_block(32, 64, kernel_size=5, stride=2, final_layer=True)
        else:
            raise ValueError("training target not recognized")

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            if self.norm == "layer":
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(1, output_channels),
                    nn.ReLU()
                )
            else:
                return nn.Sequential(
                    nn.Conv1d(input_channels, output_channels, kernel_size, stride),
                    self.norm_method(output_channels),
                    nn.ReLU()
                )
        else:
            return nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size, stride),
            )

    def unsqueeze_condition(self, latent):
        return latent.view(len(latent), 1, self.condition_dim)

    def forward(self, ear_sch, head_anthro, frequency, left_or_right):
        ear_sch_encoding = self.ear_enc(ear_sch)
        head_anthro_encoding = self.head_enc(head_anthro)
        frequency_encoding = self.freq_enc(frequency)
        left_or_right_enc = self.lr_enc(left_or_right)

        latent = torch.cat((ear_sch_encoding, head_anthro_encoding,
                            frequency_encoding), dim=1)
        latent = torch.cat((latent, left_or_right_enc), dim=1)
        latent = self.unsqueeze_condition(self.fc(latent))

        latent = self.conv1(latent)
        latent = self.conv2(latent)
        latent = self.conv3(latent)
        latent = self.conv4(latent)
        out = self.conv5(latent)

        return out


class OnsetPrediction(nn.Module):
    def __init__(self, args):
        super(OnsetPrediction, self).__init__()
        self.ear_enc = EarMeasureEncoder(args.ear_anthro_dim, args.ear_emb_dim)
        self.head_enc = HeadMeasureEncoder(args.head_anthro_dim, args.head_emb_dim)
        self.lr_enc = nn.Embedding(2, args.lr_emb_dim)
        self.fc1 = nn.Linear(args.ear_emb_dim + args.head_emb_dim + args.lr_emb_dim,
                            args.condition_dim)
        self.fc2 = nn.Linear(args.condition_dim, 36)
        self.leakly_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, ear_anthro, head_anthro, left_or_right):
        ear_anthro_encoding = self.ear_enc(ear_anthro)
        head_anthro_encoding = self.head_enc(head_anthro)
        left_or_right_enc = self.lr_enc(left_or_right)

        latent = torch.cat((ear_anthro_encoding, head_anthro_encoding), dim=1)
        latent = torch.cat((latent, left_or_right_enc), dim=1)
        latent = self.fc1(latent)
        latent = self.leakly_relu(latent)
        latent = self.fc2(latent)
        latent = self.leakly_relu(latent)

        return latent


def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    # Dataset prepare
    parser.add_argument("--ear_sch_dim", type=int, help="ear SCH dimension", default=441)
    parser.add_argument("--ear_sch_ch", type=int, help="ear SCH channel", default=3)
    parser.add_argument("--freq_bin", type=int, help="number of frequency bin", default=41)

    # Model prepare
    parser.add_argument("--ear_sch_emb_dim", type=int, help="ear SCH embedding dimension", default=64)
    parser.add_argument("--use_schSquare", type=str2bool, nargs='?', const=True, default=True,
                        help="whether to use SCH square.")
    parser.add_argument("--lr_emb_dim", type=int, help="left_or_right embedding dimension", default=16)
    parser.add_argument("--freq_emb_dim", type=int, help="frequency embedding dimension", default=16)
    parser.add_argument("--condition_dim", type=int, default=256, help="dimension of encoded conditions")
    parser.add_argument('--norm', type=str, default='layer', choices=['batch', 'layer', 'instance'],
                        help="normalization method")
    parser.add_argument('--target', type=str, default='sht', choices=['sht', 'hrtf'])

    parser.add_argument("--gpu", type=str, help="GPU index", default="1")

    args = parser.parse_args()

    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args

if __name__ == "__main__":
    args = initParams()
    print(args)
    ear_sch = torch.randn(128, 3, 441)
    ear_encoder = EarSCHEncoder2(args.ear_sch_emb_dim)
    output = ear_encoder(ear_sch)
    print(output.shape)

    # frequency = torch.LongTensor(np.random.randint(0, 41, 128))
    # left_or_right = torch.LongTensor(np.random.randint(0, 2, 128))
    #
    # model = ConvNNEarSch2HrtfSht(args)
    # output = model(ear_sch, frequency, left_or_right)
    # print(output.shape)


