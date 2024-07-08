# (1) CNN & GuidedBP

## Setup
Create an environment using
```
conda create -n siRNA_keras python=3.6
conda activate siRNA_keras
conda install tensorflow=1.12.0=mkl_py36h69b6ba0_0 
conda install keras
pip install -r requirements.txt
```

## Training
(0) Place the two files in the `./dataset` directory.
```
./dataset
     ├── negative.txt
     └── positive.rxt
```

(1) Preprocess the data using
```
python src/preprocessing.py --dna_len=151 --path_pos='./dataset/positive.txt' --path_neg='./dataset/negative.txt'
```

(2) Train the CNN model using
```
python src/CNN/train.py --output_path='result_cnn/'
```
The binary classification results is saved at `./result_cnn/prediction.csv`.

(3) Run the guided backpropagation using
```
python src/CNN/guidedBP.py --id=0 --output_path='result_cnn/' 
```
The guided backpropagation results for the data with `id=0` is saved in `./result_cnn/guidedBP_{args.id}.csv`. For the data id, please refer to `./result_cnn/prediction.csv`.


# (2) CNN+Transformer (CvT)
## Setup
Create an environment using
```
conda create -n siRNA_pytorch python=3.9
conda activate siRNA_pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
## Training
(0) Place the two files in the `./dataset` directory.
```
./dataset
     ├── negative.txt
     └── positive.rxt
```

(1) Preprocess the data using
```
python src/preprocessing.py --dna_len=151 --path_pos='./dataset/positive.txt' --path_neg='./dataset/negative.txt'
```

(2) Train the CvT model using
```
python src/CvT/train.py --output_path='result_cvt/'
```

(3) Inference the data using
```
python src/CvT/attention.py --output_path='result_cvt/'
```
The binary classification results and the attention weight are saved at `./result_cvt/prediction.csv` and `./result_cvt/attention.csv`, respectively.