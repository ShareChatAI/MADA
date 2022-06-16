import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

class DataPreparation:
    
    def __init__(self, args):
        self.args = args

    def load_features(self, split, dim, scale_norm = 1): 
    
        csv_path = self.args.csv_path + f"{self.args.lang}_{split}.csv"
        all_lines = pd.read_csv(csv_path)
        X_split = np.empty((len(all_lines), dim))
        y_split = np.empty((len(all_lines), 1))
        
        for idx in tqdm(range(len(all_lines))):
            filename = all_lines["filename"].iloc[idx]
            feat_name =  filename.split(".")[0] + ".feats.npy"
            
            f1=np.load(self.args.wav2vec_feat+"/"+feat_name)
            f2=np.load(self.args.emo_feat +"/"+feat_name)
            f3=np.load(self.args.sent_feat+"/"+feat_name)
            if f2.ndim > 1:
                features=np.hstack((f1,np.mean(np.squeeze(f2),axis=1),f3))
            else:
                features=np.hstack((f1,f2,f3))
            X_split[idx] = features #np.mean(np.squeeze(features),axis=1)
            # X_test[idx]= np.mean(np.squeeze(features),axis=1)
            class_YN = all_lines["label"].iloc[idx]
            if class_YN=="Yes":
                y_split[idx]=1
            else:
                y_split[idx]=0
        
        if scale_norm==1:
            X_split = sc.fit_transform(X_split)
        
        return X_split, y_split