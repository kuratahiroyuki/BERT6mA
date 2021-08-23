#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:02:24 2021

@author: kurata
"""

import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("./")

import pandas as pd
import torch
from Bio import SeqIO
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import argparse
from gensim.models import word2vec
from Bert_network import BERT
import pickle

common_path = os.path.abspath("..")

def import_fasta(filename):
    df_sets = []
    
    for record in SeqIO.parse(filename, "fasta"):
        df_sets.append([record.id, str(record.seq), record.description])
    
    return pd.DataFrame(df_sets, columns = ["id", "seq", "description"])

def pickle_save(filename, data):
    with open(filename, "wb") as fp:
        pickle.dump(data, fp)

def emb_seq(seq, w2v_model, features, num = 4):
    seq_emb = []
    for i in range(len(seq) - num + 1):
        try:
            seq_emb.append(np.array(w2v_model[seq[i:i+num]]))
        except:
            seq_emb.append(np.array(np.zeros([features, 1])))
    #seq_emb = np.array([np.array(w2v_model[seq[i:i+num]]) for i in range(len(seq) - num + 1)])
    return np.array(seq_emb)

class pv_data_sets(data.Dataset):
    #def __init__(self, data_sets):
    def __init__(self, data_sets, w2v_model, features, device):
        super().__init__()
        self.w2v_model = w2v_model
        self.seq = data_sets["seq"].values.tolist()
        self.id = data_sets["id"].values.tolist()
        self.device = device
        self.features = features

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        emb_mat = emb_seq(self.seq[idx], self.w2v_model, self.features)
        
        return torch.tensor(emb_mat).to(self.device).float(), self.id[idx]

def output_csv_pandas(filename, data):
    data.to_csv(filename, index = None)

class burt_process():
    def __init__(self, out_path, deep_model_path, batch_size = 64, features = 100, thresh = 0.5):
        self.out_path = out_path
        self.deep_model_path = deep_model_path
        self.batch_size = batch_size
        self.features = features
        self.thresh = thresh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def pre_training(self, dataset, w2v_model):
        os.makedirs(self.out_path, exist_ok = True) 
        data_all = pv_data_sets(dataset, w2v_model, self.features, self.device)
        loader = DataLoader(dataset = data_all, batch_size = self.batch_size, shuffle=False)
        
        net = BERT(n_layers = 3, d_model = self.features, n_heads = 4, d_dim = 100, d_ff = 400, time_seq = 41 - 4 + 1).to(self.device)
        net.load_state_dict(torch.load(self.deep_model_path, map_location = self.device))
            
        print("The number of data:" + str(len(dataset)))
            
        probs, pred_labels, seq_id_list, att_w_1, att_w_2, att_w_3 = [], [], [], [], [], []
            
        print("predicting...")
        net.eval()
        for i, (emb_mat, seq_id) in enumerate(loader):
            with torch.no_grad():
                outputs = net(emb_mat)
                        
            probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
            pred_labels.extend((np.array(outputs.cpu().detach().squeeze(1).numpy()) + 1 - self.thresh).astype(np.int16))        
            seq_id_list.extend(seq_id)
            
            att_w_1.extend(net.attn_list[0].cpu().detach().numpy()) 
            att_w_2.extend(net.attn_list[1].cpu().detach().numpy()) 
            att_w_3.extend(net.attn_list[2].cpu().detach().numpy()) 
            
        print("finished the prediction")

        print("saving results...")
        res = pd.DataFrame([seq_id_list, probs, pred_labels]).transpose()
        res.columns = ["id", "probability", "predictive labels"]
        output_csv_pandas(self.out_path + "/results.csv", res)
        att_weights = np.transpose(np.array([att_w_1, att_w_2, att_w_3]), (1, 0, 2, 3, 4))
        pickle_save(self.out_path + "/attention_weights.pkl", np.array(att_weights))
        print("finished all processes")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='path of input file', required=True)
    parser.add_argument('-o', '--out_dir', help='path of output directory', required=True)
    parser.add_argument("-sp", "--species", required=True, choices=['A.thaliana','C.elegans', 'C.equisetifolia', 'D.melanogaster', 'F.vesca', 'H.sapiens', 'R.chinensis', 'S.cerevisiae', 'T.thermophile', 'Ts.SUP5-1', 'Xoc.BLS256'], help="species for prediction of 6mA")
    parser.add_argument('-threshold', '--threshold', help='threshold to determine whether 6mA or non-6mA', default = 0.5)
    parser.add_argument("-batch", "--batch_size", action="store", dest='batch_size', default=64, type=int, help="batch size")

    test_path = parser.parse_args().input_file
    out_path = parser.parse_args().out_dir

    w2v_model = word2vec.Word2Vec.load(common_path + "/w2v_model/dna_w2v_100.pt")
    dataset = import_fasta(test_path)

    net = burt_process(out_path, deep_model_path = common_path + "/deep_model/6mA_" + parser.parse_args().species + "/deep_model", batch_size = parser.parse_args().batch_size, thresh = float(parser.parse_args().threshold))
    net.pre_training(dataset, w2v_model)





























