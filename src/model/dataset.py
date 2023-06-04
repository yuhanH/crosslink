import sys 
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
sys.path.append('/home/ubuntu/codebase/tf_binding/src/preprocessing/dna_seq/gen_training')

class TFBindingDataset(Dataset):

    def __init__(self, mode = 'train', split = 'random',transform=None):
        self.data_path = '/home/ubuntu/codebase/tf_binding/data/hg38/tf_seq_data_all_tf_200bp/'
        self.tf_names = self.get_all_tf_names()
        self.transform = transform
        self.tf_embeddings = self.get_tf_embeddings()
        self.data = self.load_data()
        self.mode = mode
        self.split = split
        if split == 'random':
            self.data_idx = self.data_split(self.data, mode)
        elif split == 'chr':
            self.data_idx = self.data_split_by_chr(self.data, mode)
        elif split == 'tf_cluster':
            self.data_idx = self.data_split_by_tf_cluster(self.data, mode)
        elif split == 'tf_and_chr':
            self.data_idx = self.data_split_by_tf_and_chr(self.data, mode)
        elif split == 'tf_in_domain':
            self.data_idx = self.data_split_by_tf_in_domain(self.data, mode)
        elif split == 'tf_in_domain_and_chr':
            self.data_idx = self.data_split_by_tf_in_domain_and_chr(self.data, mode)
        else:
            raise ValueError('Invalid split')

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        data = self.data.iloc[self.data_idx[idx]]
        seq = data['name']
        tf_name = data['tf_name']
        label = data['label']
        onehot_seq = self.onehot_encode(seq)
        tf_embedding = self.tf_embeddings[tf_name]
        return onehot_seq, tf_embedding, label

    def get_tf_embeddings(self):
        tf_embedding_path = '/home/ubuntu/protein_embeddings/all_tf_emb_esm2_t36_3B'
        model_id = 36
        tf_embeddings = {}
        for tf_name in self.tf_names:
            embedding = torch.transpose(torch.load(f'{tf_embedding_path}/{tf_name}.pt')['representations'][model_id], 0, 1)
            # Pad or crop to 100
            pad_margin_l = (300 - embedding.shape[1]) // 2
            pad_margin_r = 300 - embedding.shape[1] - pad_margin_l
            padded_embedding = torch.nn.functional.pad(embedding, (pad_margin_l, pad_margin_r), 'constant', 0)
            # Crop
            cropped_embedding = padded_embedding[:, 50:150]
            tf_embeddings[tf_name] = cropped_embedding
        return tf_embeddings

    def data_split(self, data, mode):
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        data_idx = np.arange(len(data))
        np.random.seed(42)
        np.random.shuffle(data_idx)
        train_idx = data_idx[:int(len(data_idx) * train_ratio)]
        val_idx = data_idx[int(len(data_idx) * train_ratio):int(len(data_idx) * (train_ratio + val_ratio))]
        test_idx = data_idx[int(len(data_idx) * (train_ratio + val_ratio)):]
        if mode == 'train':
            return train_idx
        elif mode == 'val':
            return val_idx
        elif mode == 'test':
            return test_idx
        else:
            raise ValueError('Invalid mode')

    def data_split_by_chr(self, data, mode, val_chr=['chr4'], test_chr=['chr14']):
        """return data index for train, val, test set split by chromosome"""
        data_train = data[~data['chr'].isin(val_chr + test_chr)]
        data_val = data[data['chr'].isin(val_chr)]
        data_test = data[data['chr'].isin(test_chr)]
        if mode == 'train':
            return data_train.index.values
        elif mode == 'val':
            return data_val.index.values
        elif mode == 'test':
            return data_test.index.values
        else:
            raise ValueError('Invalid mode')
        
    def data_split_by_tf_and_chr(self, data, mode, val_chr=['chr4'], test_chr=['chr14'], val_cluster=[8], test_cluster=[9]):
        """return data index for train, val, test set split by TF cluster"""
        tf_cluster = pd.read_csv('/home/ubuntu/protein_embeddings/factor_DNA_binding_emb_esm2_t36_3B/tf_cluster.kmeans10.csv')
        train_tfs = tf_cluster[~tf_cluster['cluster'].isin(val_cluster + test_cluster)]['tf'].values
        val_tfs = tf_cluster[tf_cluster['cluster'].isin(val_cluster)]['tf'].values
        test_tfs = tf_cluster[tf_cluster['cluster'].isin(test_cluster)]['tf'].values
        data_train = data[data['tf_name'].isin(train_tfs) & ~data['chr'].isin(val_chr + test_chr)]
        data_val = data[data['tf_name'].isin(val_tfs) & data['chr'].isin(val_chr)]
        data_test = data[data['tf_name'].isin(test_tfs) & data['chr'].isin(test_chr)]
        if mode == 'train':
            return data_train.index.values
        elif mode == 'val':
            return data_val.index.values
        elif mode == 'test':
            return data_test.index.values
        else:
            raise ValueError('Invalid mode')
        
    def data_split_by_tf_cluster(self, data, mode, val_cluster=[8], test_cluster=[9]):
        """return data index for train, val, test set split by TF cluster"""
        tf_cluster = pd.read_csv('/home/ubuntu/protein_embeddings/factor_DNA_binding_emb_esm2_t36_3B/tf_cluster.kmeans10.csv')
        train_tfs = tf_cluster[~tf_cluster['cluster'].isin(val_cluster + test_cluster)]['tf'].values
        val_tfs = tf_cluster[tf_cluster['cluster'].isin(val_cluster)]['tf'].values
        test_tfs = tf_cluster[tf_cluster['cluster'].isin(test_cluster)]['tf'].values
        data_train = data[data['tf_name'].isin(train_tfs)]
        data_val = data[data['tf_name'].isin(val_tfs)]
        data_test = data[data['tf_name'].isin(test_tfs)]
        if mode == 'train':
            return data_train.index.values
        elif mode == 'val':
            return data_val.index.values
        elif mode == 'test':
            return data_test.index.values
        else:
            raise ValueError('Invalid mode')
    
    def data_split_by_tf_in_domain(self, data, mode, val_num=1, test_num=1):
        """return data index for train, val, test set split by tf_name. Leave 1 TFs in each TF cluster for validation and same for test set, respectively"""
        tf_cluster = pd.read_csv('/home/ubuntu/protein_embeddings/factor_DNA_binding_emb_esm2_t36_3B/tf_cluster.kmeans10.csv')
        # first tf in each cluster is used for validation, last tf in each cluster is used for test
        val_tfs = tf_cluster.groupby('cluster').head(val_num)['tf'].values
        test_tfs = tf_cluster.groupby('cluster').tail(test_num)['tf'].values
        train_tfs = tf_cluster[~tf_cluster['tf'].isin(val_tfs + test_tfs)]['tf'].values
        data_train = data[data['tf_name'].isin(train_tfs)]
        data_val = data[data['tf_name'].isin(val_tfs)]
        data_test = data[data['tf_name'].isin(test_tfs)]
        if mode == 'train':
            return data_train.index.values
        elif mode == 'val':
            return data_val.index.values
        elif mode == 'test':
            return data_test.index.values
        else:
            raise ValueError('Invalid mode')
        
    def data_split_by_tf_in_domain_and_chr(self, data, mode, val_num=1, test_num=1, val_chr=['chr4'], test_chr=['chr14']):
        """return data index for train, val, test set split by tf_name. Leave 1 TFs in each TF cluster for validation and same for test set, respectively"""
        tf_cluster = pd.read_csv('/home/ubuntu/protein_embeddings/factor_DNA_binding_emb_esm2_t36_3B/tf_cluster.kmeans10.csv')
        # first tf in each cluster is used for validation, last tf in each cluster is used for test
        val_tfs = tf_cluster.groupby('cluster').head(val_num)['tf'].values
        test_tfs = tf_cluster.groupby('cluster').tail(test_num)['tf'].values
        train_tfs = tf_cluster[~tf_cluster['tf'].isin(val_tfs + test_tfs)]['tf'].values
        non_train_chrs = val_chr + test_chr
        data_train = data[data['tf_name'].isin(train_tfs)].query('chr not in @non_train_chrs')
        data_val = data[data['tf_name'].isin(val_tfs)].query('chr in @val_chr')
        data_test = data[data['tf_name'].isin(test_tfs)].query('chr in @test_chr')
        if mode == 'train':
            return data_train.index.values
        elif mode == 'val':
            return data_val.index.values
        elif mode == 'test':
            return data_test.index.values
        else:
            raise ValueError('Invalid mode')
        


    def load_data(self):
        processed_data_path = f'{self.data_path}/tf_binding_data.bed'
        if os.path.exists(processed_data_path):
            print('Loading processed data...')
            bed_cols = ['chr', 'start', 'end', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'sample', 'label', 'tf_name']
            data = pd.read_csv(processed_data_path, sep='\t', names=bed_cols)
        else:
            data = []
            from tqdm import tqdm
            for tf_name in tqdm(self.tf_names):
                print(f'Loading data for {tf_name}')
                bed_cols = ['chr', 'start', 'end', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'sample']
                pos_data = pd.read_csv(f'{self.data_path}/pos/{tf_name}.bed', names=bed_cols, sep = '\t')
                neg_data = pd.read_csv(f'{self.data_path}/neg/{tf_name}.bed', names=bed_cols, sep = '\t')
                pos_data['label'] = pos_data['score'].astype(float)
                neg_data['label'] = 0
                pos_data['tf_name'] = tf_name
                neg_data['tf_name'] = tf_name
                data.append(pos_data)
                data.append(neg_data)
            data = pd.concat(data)
            # Save data
            data.to_csv(processed_data_path, index=False, sep='\t', header = False)

        # Log1p transform score
        data['label'] = np.log1p(data['label'])
        return data

    def get_all_chrs(self):
        chrs = []
        for i in range(1, 23):
            chrs.append('chr' + str(i))
        chrs.append('chrX')
        return chrs

    def get_all_tf_names(self):
        tf_names = []
        for tf_name in os.listdir(self.data_path + '/pos'):
            if tf_name.endswith('.bed'):
                tf_names.append(tf_name.split('.')[0])
        return tf_names

    def onehot_encode(self, seq):
        seq_dict = {'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4}
        seq_onehot = np.zeros((5, len(seq)))
        for i, char in enumerate(seq):
            seq_onehot[seq_dict[char], i] = 1
        return seq_onehot

if __name__ == '__main__':
    dataset = TFBindingDataset()
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    breakpoint()
