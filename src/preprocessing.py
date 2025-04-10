
import numpy as np
import argparse


def dna2num(dna):
    if dna.upper() == "A":
        return 0
    elif dna.upper() == "T":
        return 1
    elif dna.upper() == "G":
        return 2
    elif dna.upper() == "C":
        return 3
    elif dna.upper() == "N":
        return 4


def num2dna(num):
    if num == 0:
        return "A"
    elif num == 1:
        return "T"
    elif num == 2:
        return "G"
    elif num == 3:
        return "C"
    elif num == 4:
        return "N"


def dna2array(DNAstring):
    numarr = []
    length = len(DNAstring)
    for i in range(0, length):
        num = dna2num(DNAstring[i:i+1])
        if num >= 0:
            numarr.append(num)
    return numarr


def array2dna(numarr):
    DNAstring = []
    length = numarr.shape[0]
    for i in range(0, length):
        dna = num2dna(numarr[i].argmax())
        DNAstring.append(dna)
    DNAstring = ''.join(DNAstring)
    return DNAstring

def load_data(path_pos, path_neg, dna_len):
    X_pos, X_origin_pos = [], []
    Y_pos = []
    f = open(path_pos, "r")
    line = f.readline()
    while line:
        line2 = line.rstrip()
        X_origin_pos.append(line2)
        x = np.array(list(map(dna2array, line2))).reshape(-1)
        if x.shape[0] == dna_len:
            OneHotArr = np.eye(5)[x]
            X_pos.append(OneHotArr)
            Y_pos.append(1)
        line = f.readline()

    X_pos, X_origin_pos = np.array(X_pos), np.array(X_origin_pos)
    Y_pos = np.array(Y_pos)

    X_neg, X_origin_neg = [], []
    Y_neg = []
    f = open(path_neg, "r")
    line = f.readline()
    while line:
        line2 = line.rstrip()
        X_origin_neg.append(line2)
        x = np.array(list(map(dna2array, line2))).reshape(-1)
        if x.shape[0] == dna_len:
            OneHotArr = np.eye(5)[x]
            X_neg.append(OneHotArr)
            Y_neg.append(0)
        line = f.readline()

    X_neg, X_origin_neg = np.array(X_neg), np.array(X_origin_neg)
    Y_neg = np.array(Y_neg)

    return X_pos, X_origin_pos, Y_pos, X_neg, X_origin_neg, Y_neg


if __name__ == '__main__':
    print('preprocessing...')

    # ----
    parser = argparse.ArgumentParser()

    parser.add_argument('--dna_len', default=151, type=int)
    parser.add_argument('--path_pos', default='./dataset/positive.txt', type=str)
    parser.add_argument('--path_neg', default='./dataset/negative.txt', type=str)
    parser.add_argument('--output_path', default='./dataset/', type=str)

    args = parser.parse_args()

    # ----
    X_pos, X_origin_pos, Y_pos, X_neg, X_origin_neg, Y_neg = load_data(args.path_pos, args.path_neg, args.dna_len)
    np.save(args.output_path+'X_pos', X_pos)
    np.save(args.output_path+'X_origin_pos', X_origin_pos)
    np.save(args.output_path+'Y_pos', Y_pos)
    np.save(args.output_path+'X_neg', X_neg)
    np.save(args.output_path+'X_origin_neg', X_origin_neg)
    np.save(args.output_path+'Y_neg', Y_neg)

    print('=== X_pos ===')
    print(X_pos.shape)
    # print(X_pos)
    print('=== Y_pos ===')
    print(Y_pos.shape)
    # print(Y_pos)
    print('=== X_neg ===')
    print(X_neg.shape)
    # print(X_neg)
    print('=== Y_neg ===')
    print(Y_neg.shape)
    # print(Y_neg)
    # ----
    print('done.')
