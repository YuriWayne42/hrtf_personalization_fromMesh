import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import random
import distutils.util

def plot_HRTF_72_32(subject, hrtf, frequency, right):
    # azimuth_list = [-80, -65, -55] + list(range(-45, 50, 5)) + [55, 65, 80]
    # azimuth_angle = azimuth_list[azimuth]
    frequency_lst = np.arange(0, 128) * 44100 / 256 / 1000
    frequency_Hz = frequency_lst[frequency]
    fig, ax = plt.subplots(1, 1)
    im = plt.pcolormesh(hrtf[0], cmap=plt.cm.viridis)
    # im = plt.imshow(hrtf, interpolation='nearest', cmap=plt.cm.viridis)
    # im.set_clim([0, 5.2])
    title = "Subject:%d, Frequency:%.2fk, %s ear" % (subject+1, frequency_Hz, "Right" if right else "Left")
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set(# xticks=list(np.arange(0.5, 64.5, 8)),
           # yticks=np.arange(0, 256 + 32, 32),
           # -45, 230.625 + 4.375, 4.375
           # xticklabels=['{:,.0f}'.format(x) for x in list(np.arange(-45, 230.625 + 4.375, 4.375*8))],
           # yticklabels=['{:,.2f}'.format(x) + 'k' for x in np.arange(0, 256 + 32, 32)*44100/512/1000],
           title=title,
           ylabel='Azimuth angle (degree)',
           xlabel='Elevation angle (degree)')
    return ax

def setup_seed(random_seed, cudnn_deterministic=True):
    """ set_random_seed(random_seed, cudnn_deterministic=True)

    Set the random_seed for numpy, python, and cudnn

    input
    -----
      random_seed: integer random seed
      cudnn_deterministic: for torch.backends.cudnn.deterministic

    Note: this default configuration may result in RuntimeError
    see https://pytorch.org/docs/stable/notes/randomness.html
    """

    # initialization
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = False

def compare_HRTF_72_32(subject, gen_hrtf, real_hrtf, frequency, right):
    frequency_lst = np.arange(0, 128) * 44100 / 256 / 1000
    frequency_Hz = frequency_lst[frequency]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    real_im = ax1.pcolormesh(real_hrtf[0], cmap=plt.cm.viridis)
    title = "GT, Subject:%d, Frequency:%.2fk, %s ear" % (subject + 1, frequency_Hz, "Right" if right else "Left")
    cbar = ax1.figure.colorbar(real_im, ax=ax1)
    ax1.set(# xticks=list(np.arange(0.5, 64.5, 8)),
            # yticks=np.arange(0, 256 + 32, 32),
            # xticklabels=['{:,.0f}'.format(x) for x in list(np.arange(-45, 230.625 + 4.375, 4.375 * 8))],
            # yticklabels=['{:,.2f}'.format(x) + 'k' for x in np.arange(0, 256 + 32, 32) * 44100 / 512 / 1000],
            title=title,
            ylabel='Azimuth angle (degree)',
            xlabel='Elevation angle (degree)')

    gen_im = ax2.pcolormesh(gen_hrtf[0], cmap=plt.cm.viridis)
    gen_im.set_clim([real_hrtf.min(), real_hrtf.max()])
    title = "GEN, Subject:%d, Frequency:%.2fk, %s ear" % (subject+1, frequency_Hz, "Right" if right else "Left")
    cbar = ax2.figure.colorbar(gen_im, ax=ax2)
    ax2.set(# xticks=list(np.arange(0.5, 64.5, 8)),
           # yticks=np.arange(0, 256 + 32, 32),
           # xticklabels=['{:,.0f}'.format(x) for x in list(np.arange(-45, 230.625 + 4.375, 4.375*8))],
           # yticklabels=['{:,.2f}'.format(x) + 'k' for x in np.arange(0, 256 + 32, 32)*44100/512/1000],
           title=title,
           ylabel='Azimuth angle (degree)',
           xlabel='Elevation angle (degree)')

    err_im = ax3.pcolormesh(abs(real_hrtf[0] - gen_hrtf[0]), cmap=plt.cm.viridis)
    err_im.set_clim([real_hrtf.min(), real_hrtf.max()])
    title = "ERR, Subject:%d, Frequency:%.2fk, %s ear" % (subject + 1, frequency_Hz, "Right" if right else "Left")
    cbar = ax3.figure.colorbar(err_im, ax=ax3)
    ax3.set(# xticks=list(np.arange(0.5, 64.5, 8)),
            # yticks=np.arange(0, 256 + 32, 32),
            # xticklabels=['{:,.0f}'.format(x) for x in list(np.arange(-45, 230.625 + 4.375, 4.375*8))],
            # yticklabels=['{:,.2f}'.format(x) + 'k' for x in np.arange(0, 256 + 32, 32)*44100/512/1000],
            title=title,
            ylabel='Azimuth angle (degree)',
            xlabel='Elevation angle (degree)')

    return fig

def compare_SHT_64(subject, gen_sht, real_sht, frequency, right):
    frequency_lst = np.arange(0, 128) * 44100 / 256 / 1000
    frequency_Hz = frequency_lst[frequency]
    fig, ax = plt.subplots(1,1, figsize=(15, 4))
    ax.plot(gen_sht)
    ax.plot(real_sht)
    title = "Subject:%d, Frequency:%.2fk, %s ear" % (subject + 1, frequency_Hz, "Right" if right else "Left")
    ax.set_title(title)

    return fig

def plot_loss(args, current_depth):
    log_file = os.path.join(args.out_fold, "loss_" + str(current_depth) + ".log")
    with open(log_file, "r") as log:
        x = np.array([[float(i) for i in line[:-1].split('\t')] for line in log.readlines()])
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 4))
        ax1.plot(x[:, 1])
        ax1.set_title("Discriminator Loss")
        ax2.plot(abs(x[:, 2]))
        ax2.set_title("Generator Loss")
    return fig

def SHT_recon_loss(gen_SHT, gt_hrtf, SHvec):
    recreated_HRTF = SHvec * gen_SHT
    return recreated_HRTF



def str2bool(v):
    return bool(distutils.util.strtobool(v))
