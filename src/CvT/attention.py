import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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
    model.load_state_dict(torch.load(args.output_path + 'model.pkl'))
    model.eval()

    pred_csv_path = f'{args.output_path}/prediction.csv'
    with open(pred_csv_path, 'w') as f:
        f.write('id,seq,prob_neg,prob_pos\n')

    att_csv_path = f'{args.output_path}/attention.csv'
    with open(att_csv_path, 'w') as f:
        f.write('id,seq,attention weight for class token\n')

    idx_offset = 0
    # ---- evaluation ----
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            data, label = batch[0], batch[1]
            data, label = data.to(args.device), label.to(args.device)

            y, w = model(data)
            prob_batch = F.softmax(y, dim=1).cpu().numpy()

            batch_size = data.shape[0]
            id_batch = np.arange(idx_offset, idx_offset + batch_size)
            idx_offset += batch_size

            X_origin_batch = X_origin[id_batch]

            pred_out = np.column_stack([id_batch, X_origin_batch, prob_batch])
            with open(pred_csv_path, 'a') as f:
                np.savetxt(f, pred_out, delimiter=',', fmt='%s')

            att_batch = w.detach().cpu().numpy()
            att_batch = att_batch[:, 0, 1:]

            att_batch = att_batch / att_batch.sum(axis=-1, keepdims=True)
            att_batch = np.round(att_batch, 10)

            att_out = np.column_stack([id_batch, X_origin_batch])
            att_out = np.column_stack([att_out, att_batch])

            with open(att_csv_path, 'a') as f:
                np.savetxt(f, att_out, delimiter=',', fmt='%s')

    return 0


if __name__ == '__main__':
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

    main(args)
