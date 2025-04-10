import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from model import TransformerClassification
from utils import Dataset, fix_seed


def main(args):
    fix_seed(seed=args.seed)
    # ---- create loader ----
    X = np.concatenate([np.load(args.dataset_path+'X_pos.npy'),
                       np.load(args.dataset_path+'X_neg.npy')])
    Y = np.concatenate([np.load(args.dataset_path+'Y_pos.npy'),
                       np.load(args.dataset_path+'Y_neg.npy')])
    X_origin = np.concatenate([np.load(args.dataset_path+'X_origin_pos.npy'),
                               np.load(args.dataset_path+'X_origin_neg.npy')])
    
    dataset = Dataset(X, Y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # ---- load model ----
    model = TransformerClassification(
        ch = X.shape[2], 
        length = X.shape[1],
        model_dim = args.model_dim,
        output_dim = 2,
        depth = args.depth,
        bin = args.bin,
        ).to(args.device)
    model.load_state_dict(torch.load(args.output_path+'model.pkl'))

    # ---- evaluation ----
    model.eval()

    gt, prob = [], []
    attention = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            data, label = batch[0], batch[1]
            data, label = data.to(args.device), label.to(args.device)

            y, w = model(data)

            gt.extend(label.cpu().detach().numpy())
            prob.extend(F.softmax(y, dim=1).cpu().detach().numpy())
            attention.extend(w.detach().cpu().numpy().astype(np.float16))

    gt, prob = np.array(gt), np.array(prob)
    attention = np.array(attention)[:, 0, 1:]

    # ---- save prediction ----
    id = np.arange(len(X))
    prediction = np.concatenate([id[:,np.newaxis], X_origin[:,np.newaxis]], axis=-1)
    prediction = np.concatenate([prediction, prob], axis=-1)
    with open(f'{args.output_path}/prediction.csv', 'w') as f:
        f.write('id, seq, prob_neg, prob_pos\n')
        np.savetxt(f, prediction, delimiter=',', fmt='%s')

    # ---- save attention ----
    attention = attention / attention.sum(axis=-1, keepdims=True)
    attention = np.round(attention, 10)

    id = np.arange(len(X))
    prediction = np.concatenate([id[:,np.newaxis], X_origin[:,np.newaxis]], axis=-1)
    prediction = np.concatenate([prediction, attention], axis=-1)
    with open(f'{args.output_path}/attention.csv', 'w') as f:
        f.write('id, seq, attention weight for class token \n')
        np.savetxt(f, prediction, delimiter=',', fmt='%s')

    return 0


if __name__ == '__main__':
    # ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42,  type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    
    parser.add_argument('--depth', default=6, type=int)
    parser.add_argument('--model_dim', default=64, type=int)
    parser.add_argument('--bin', default=1, type=int)

    parser.add_argument('--dataset_path', default='./dataset/', type=str)
    parser.add_argument('--output_path', default='./result_cvt/', type=str)
    args = parser.parse_args()

    # ----
    main(args)
