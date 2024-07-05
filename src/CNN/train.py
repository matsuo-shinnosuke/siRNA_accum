import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras import optimizers
from model import CNN
from keras.models import load_model
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(args):
    # ---- load data ----
    X = np.concatenate([np.load(args.dataset_path+'X_pos.npy'),
                       np.load(args.dataset_path+'X_neg.npy')])
    Y = np.concatenate([np.load(args.dataset_path+'Y_pos.npy'),
                       np.load(args.dataset_path+'Y_neg.npy')])
    X, Y = X[:,:,np.newaxis,:], np.eye(2)[Y]

    X_origin = np.concatenate([np.load(args.dataset_path+'X_origin_pos.npy'),
                               np.load(args.dataset_path+'X_origin_neg.npy')])
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_rate, random_state=args.seed)

    # ---- calulate class weight ----
    y_integers = np.argmax(Y, axis=1)
    class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_integers), y=y_integers)
    class_weight = dict(enumerate(class_weight))

    # ---- compile model ----
    model = CNN(input_shape=X_train.shape)
    optimizer = optimizers.SGD(lr=args.lr, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizer = optimizers.Adam(lr=args.lr, beta_1=0.9, beta_2=0.999,epsilon=None,decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())

    # ---- train model ----
    history = model.fit(
        X_train, Y_train, 
        batch_size=args.batch_size, 
        epochs=args.num_epoch, 
        validation_data=(X_test, Y_test), 
        class_weight = class_weight)
    
    # ---- save mdoel ----
    model.save(args.output_path+'model.h5')
    model = load_model(args.output_path+'model.h5')

    # ---- evaluate mdoel ----
    prob = model.predict(X_test)[:,1]
    gt = np.argmax(Y_test, axis=1)
    fpr, tpr, thresholds = roc_curve(gt,prob)
    # ----
    print('roc-auc_score: %.3f' % roc_auc_score(gt, prob))
    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.savefig(args.output_path+'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- save prediction ----
    prob = model.predict(X)
    id = np.arange(len(X))
    prediction = np.concatenate([id[:,np.newaxis], X_origin[:,np.newaxis]], axis=-1)
    prediction = np.concatenate([prediction, prob], axis=-1)
    with open(f'{args.output_path}/prediction.csv', 'w') as f:
        f.write('id, seq, prob_neg, prob_pos\n')
        np.savetxt(f, prediction, delimiter=',', fmt='%s')

    return 0

if __name__ == '__main__':
    # ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42,  type=int)
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=512, type=int)

    parser.add_argument('--dataset_path', default='./dataset/', type=str)
    parser.add_argument('--test_rate', default=0.25, type=float)
    parser.add_argument('--output_path', default='./result/', type=str)
    args = parser.parse_args()

    # ----
    main(args)
