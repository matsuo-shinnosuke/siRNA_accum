import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
import logging

from model import TransformerClassification
from utils import Dataset, fix_seed, make_folder, visualize_loss_auc


def main(args):
    fix_seed(seed=args.seed)
    make_folder(args.output_path)

    # ---- create loger ----
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(args.output_path+'training.log')
    logging.basicConfig(level=logging.INFO, handlers=[
                        stream_handler, file_handler])
    logging.info(args)

    # ---- create loader ----
    X = np.concatenate([np.load(args.dataset_path+'X_pos.npy'),
                       np.load(args.dataset_path+'X_neg.npy')])
    Y = np.concatenate([np.load(args.dataset_path+'Y_pos.npy'),
                       np.load(args.dataset_path+'Y_neg.npy')])
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_rate, random_state=args.seed)

    train_dataset = Dataset(X_train, Y_train)
    test_dataset = Dataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # ---- create model ----
    model = TransformerClassification(
        ch = X.shape[2], 
        length = X.shape[1],
        model_dim = args.model_dim,
        output_dim = 2,
        depth = args.depth,
        bin = args.bin,
        )
    
    model = model.to(args.device)

    weight = 1 / np.eye(2)[Y].sum(axis=0)
    weight /= weight.sum()
    weight = torch.tensor(weight).float().to(args.device)
    loss_function = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {'best_auc': 0, 
               'train_loss': [], 'train_auc': [],
               'test_loss': [], 'test_auc': []}
    for epoch in range(args.num_epoch):
        model.train()

        losses = []
        train_gt, train_pred = [], []

        for batch in tqdm(train_loader, leave=False):
            data, label = batch[0], batch[1]
            data, label = data.to(args.device), label.to(args.device)

            y, _ = model(data)
            prob = F.softmax(y, dim=1)

            loss = loss_function(y, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            train_gt.extend(label.cpu().detach().numpy())
            train_pred.extend(prob[:, 1].cpu().detach().numpy())

        train_loss = np.array(losses).mean()
        train_auc = roc_auc_score(train_gt, train_pred)

        history['train_loss'].append(train_loss)
        history['train_auc'].append(train_auc)

        # ----
        model.eval()

        losses = []
        test_gt, test_pred = [], []
        test_atten_w = []

        with torch.no_grad():
            for batch in tqdm(test_loader, leave=False):
                data, label = batch[0], batch[1]
                data, label = data.to(args.device), label.to(args.device)

                y, w = model(data)
                prob = F.softmax(y, dim=1)
                loss = loss_function(y, label)

                losses.append(loss.item())
                test_gt.extend(label.cpu().detach().numpy())
                test_pred.extend(prob.cpu().detach().numpy())
                test_atten_w.extend(w.detach().cpu().numpy())

        test_loss = np.array(losses).mean()
        test_gt, test_pred = np.array(test_gt), np.array(test_pred)
        test_auc = roc_auc_score(test_gt, test_pred[:, 1])

        history['test_loss'].append(test_loss)
        history['test_auc'].append(test_auc)

        logging.info('[%d/%d]: train_auc: %.3f, train_loss: %.3f, test_auc: %.3f, test_loss: %.3f'
              % (epoch+1, args.num_epoch, train_auc, train_loss, test_auc, test_loss))

        # ---- save mdoel ----
        # visualize_loss_auc(history, args.output_path)
        if history['best_auc'] < test_auc:
            history['best_auc'] = test_auc
            torch.save(model.state_dict(), '%s/model.pkl' % args.output_path) 

    return 0


if __name__ == '__main__':
    # ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42,  type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    
    parser.add_argument('--depth', default=6, type=int)
    parser.add_argument('--model_dim', default=64, type=int)
    parser.add_argument('--bin', default=1, type=int)

    parser.add_argument('--dataset_path', default='./dataset/', type=str)
    parser.add_argument('--test_rate', default=0.3, type=float)
    parser.add_argument('--output_path', default='./result_cvt/', type=str)
    args = parser.parse_args()

    # ----
    main(args)
