import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import numpy as np
import argparse
import torch
import random
from os import listdir
import re
from classifiers import MADA
from dataset import DataPreparation

def train(args):
    
    args.emo_feat = args.feats_base_path + args.lang + "/emo_pca_"
    args.sent_feat = args.feats_base_path + args.lang + "/sent_pca_"
    args.wav2vec_feat = args.feats_base_path + args.lang + "/wav2vec_xlsr_pca_"

    print("=== Model Initialization ===")
    model = MADA(args)
    feat_dim_emo_pca = get_feat_dim(args.feats_base_path + args.lang, "emo_pca")
    args.emo_feat = args.emo_feat + str(feat_dim_emo_pca) + "_feats"

    feat_dim_sent_pca = get_feat_dim(args.feats_base_path + args.lang, "sent_pca")
    args.sent_feat = args.sent_feat + str(feat_dim_sent_pca) + "_feats"

    feat_dim_xlsr_pca = get_feat_dim(args.feats_base_path + args.lang, "wav2vec_xlsr_pca")
    args.wav2vec_feat = args.wav2vec_feat + str(feat_dim_xlsr_pca) + "_feats"

    feat_dim=int(feat_dim_sent_pca)+int(feat_dim_emo_pca)+ int(feat_dim_xlsr_pca)

    data_prep_obj = DataPreparation(args)
    X_train, Y_train = data_prep_obj.load_features(split = "train", dim = feat_dim)
    X_test, Y_test = data_prep_obj.load_features(split = "test", dim = feat_dim)
    Y_train=np.squeeze(Y_train)
    Y_test=np.squeeze(Y_test)

    model.train(X_train, Y_train, X_test, Y_test)


def get_feat_dim(path, flag_str):
    dim=0
    for f in listdir(path):
        m = re.search(flag_str+'_(.+?)_feats', f)        
        if m:
            dim=m.group(1)
    return dim

def set_seed(seed_val):
    np.random.seed(seed_val) 
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
        torch.cuda.manual_seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--lang', type=str, default='Hindi', choices=["Hindi", "Bengali", "Gujarati", "Kannada", "Malayalam", 
    "Punjabi", "Tamil", "Bhojpuri", "Odia", "Haryanvi"])
    parser.add_argument('--csv_path', type=str, default='./SC_abuse_detection/')
    parser.add_argument('--feats_base_path', type=str, default='')  
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    print(args)
    
    if args.mode == "train":
        train(args)
    else:
        raise Exception("Error!")
