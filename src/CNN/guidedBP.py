import argparse
import numpy as np
from keras.models import load_model, Model
import innvestigate.utils as iutils
from innvestigate.analyzer import GuidedBackprop
from keras import backend as K
from keras.layers import Lambda
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"      

class _MaskedGuidedBackprop(GuidedBackprop):
    def __init__(self, 
                 model,
                 R_mask, 
                 **kwargs):
        super(_MaskedGuidedBackprop, self).__init__(model, neuron_selection_mode="all", **kwargs)
        self.initialize_r_mask(R_mask)

    def initialize_r_mask(self, R_mask):
        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R

class GBP(_MaskedGuidedBackprop):
    def __init__(self, 
                 model, 
                 target_id, 
                 relu=False,
                 **kwargs):
        self.relu=relu
        R_mask = np.zeros(model.output_shape[1])
        R_mask[target_id] = 1
        super(GBP, self).__init__(model, R_mask=R_mask, **kwargs)
    
    def analyze(self, inputs):
        if self.relu:
            return np.maximum(super(GBP, self).analyze(inputs), 0)
        else:
            return super(GBP, self).analyze(inputs)

    
def main(args):
    # ---- load data ----
    X = np.concatenate([np.load(args.dataset_path+'X_pos.npy'),
                       np.load(args.dataset_path+'X_neg.npy')])
    Y = np.concatenate([np.load(args.dataset_path+'Y_pos.npy'),
                       np.load(args.dataset_path+'Y_neg.npy')])
    X, Y = X[:,:,np.newaxis,:], np.eye(2)[Y]

    X_origin = np.concatenate([np.load(args.dataset_path+'X_origin_pos.npy'),
                               np.load(args.dataset_path+'X_origin_neg.npy')])
    
    # ---- load trained model ----
    model = load_model(args.output_path+'model.h5')
    print(model.summary())

    # ---- guidedBP ----
    partial_model = Model(
        inputs=model.inputs,
        outputs=iutils.keras.graph.pre_softmax_tensors(model.outputs),
        name=model.name,
    )
    guidedbackprop_analyzer = GBP(
        partial_model, target_id=1, relu=True,
    )

    x = X[args.example_id][np.newaxis,]
    x_origin = np.array(list(X_origin[args.example_id]))[:,np.newaxis].transpose(1,0)
    analysis_guidedbackprop = guidedbackprop_analyzer.analyze(x)[0,:,0,:].transpose(1,0)

    with open(f'{args.output_path}/guidedBP_{args.example_id}.csv', 'w') as f_handle:
        np.savetxt(f_handle, x_origin, delimiter=',', fmt='%s')
        np.savetxt(f_handle, analysis_guidedbackprop, delimiter=',', fmt='%.4f')

    return 0

if __name__ == '__main__':
    # ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42,  type=int)
    parser.add_argument('--dataset_path', default='./dataset/', type=str)
    parser.add_argument('--output_path', default='./result/', type=str)
    parser.add_argument('--example_id', default=0, type=int)
    args = parser.parse_args()

    # ----
    main(args)
