## Requirements
* python >= 3.9
* cuda && cudnn

## Setup

Clone the repository
```
git clone https://github.com/matsuo-shinnosuke/siRNA_accum
cd siRNA_accum
```

Create an environment using
```
conda create -n siRNA python=3.9
conda activate siRNA

# Pytorch install: see https://pytorch.org/get-started/locally/
# For example,
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Training
Place the pos and neg files in the `./dataset` directory.
```
./dataset
     ├── negative.txt
     └── positive.rxt
```

Preprocessing the data using
```
python src/preprocessing.py --dna_len=151 --path_pos='./dataset/positive.txt' --path_neg='./dataset/negative.txt'
```

Training the CNN+Transformer model using
```
python src/main.py
```