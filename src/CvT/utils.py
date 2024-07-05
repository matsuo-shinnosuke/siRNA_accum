import torch
import numpy as np
import os
import matplotlib.pyplot as plt


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


def make_folder(path):
    p = ''
    for x in path.split('/'):
        p += x+'/'
        if not os.path.exists(p):
            os.mkdir(p)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.len = self.label.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.label[idx]).long()
        return data, label


def visualize_loss_auc(history, path):
    plt.plot(history['train_auc'], label='training AUC')
    plt.plot(history['test_auc'], label='test AUC')
    plt.title('AUC')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(path+'curve_auc.png', dpi=300)
    plt.close()

    plt.plot(history['train_loss'], label='training loss')
    plt.plot(history['test_loss'], label='test loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(path+'curve_loss.png', dpi=300)
    plt.close()
