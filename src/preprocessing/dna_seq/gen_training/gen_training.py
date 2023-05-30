import os
import gzip
import numpy as np
import pandas as pd
from multiprocessing import Pool

def get_all_chrs():
    chrs = []
    for i in range(1, 23):
        chrs.append('chr' + str(i))
    chrs.append('chrX')
    return chrs

def load_chr(chr_name):
    chr_path = f'/home/ubuntu/codebase/tf_binding/data/hg38/dna_sequence/{chr_name}.fa.gz'
    print(f'Reading sequence: {chr_path}')
    with gzip.open(chr_path, 'r') as f:
        seq = f.read().decode("utf-8")
    seq = seq[seq.find('\n'):]
    seq = seq.replace('\n', '').lower()
    return seq

# For multiprocessing
# Load all chromosomes
chr_names = get_all_chrs()
chr_dict = {}
for chr_name in chr_names:
    chr_dict[chr_name] = load_chr(chr_name)

def main():
    # Load all TFs
    tf_names = get_all_tf_names()

    # Save path
    save_path = '/home/ubuntu/codebase/tf_binding/data/hg38/tf_seq_data'
    os.makedirs(f'{save_path}/pos', exist_ok=True)
    os.makedirs(f'{save_path}/neg', exist_ok=True)

    with Pool(processes=28) as pool:
        df_tuple_list = pool.map(get_training_data_for_tf, tf_names)

    for data_tuple in df_tuple_list:
        tf_name, df_pos, df_neg = data_tuple
        df_pos.to_csv(os.path.join(f'{save_path}/pos/{tf_name}.bed'), sep='\t', index=False, header=False)
        df_neg.to_csv(os.path.join(f'{save_path}/neg/{tf_name}.bed'), sep='\t', index=False, header=False)

def get_training_data_for_tf(tf_name):
    seq_length = 100
    n_pos = 5000
    n_neg = 5000

    # Output bed file: chr, start, end, sequence, label, cell_type
    # Read bed file for tf
    bed_df = get_bed(tf_name)
    # Get top n peaks
    top_pos_df = bed_df.nlargest(n_pos, 'score')
    pos_df = get_seq_bed(top_pos_df, chr_dict, seq_length)
    # Get n negative samples
    neg_df = sample_negatives(chr_dict, bed_df, n_neg, seq_length)
    return tf_name, pos_df, neg_df

def get_seq_bed(bed_df, chr_dict, seq_length):
    for index, row in bed_df.iterrows():
        peak_mid = (row['start'] + row['end']) // 2
        seq = chr_dict[row['chr']][(peak_mid - seq_length // 2):(peak_mid + seq_length // 2)]
        bed_df.loc[index, 'sample'] = bed_df.loc[index, 'name']
        bed_df.loc[index, 'name'] = seq
    return bed_df

def sample_negatives(chr_dict, bed_df, n_neg, seq_length):
    sampled_negatives = []
    chr_sample_size = n_neg // len(chr_dict.keys()) + 1
    peak_margin = 1000
    for chr_name, seq in chr_dict.items():
        print('Sampling negatives for {}'.format(chr_name))
        chr_mask = np.zeros(len(seq))
        # Get all peaks for this chromosome
        chr_bed_df = bed_df[bed_df['chr'] == chr_name]
        for i, row in chr_bed_df.iterrows():
            chr_mask[row['start'] - peak_margin :row['end'] + peak_margin] = 1
        # Sample negative peaks
        # Get index of all 0s
        neg_idx = np.where(chr_mask == 0)[0][::1000] # Sample every 1000th index
        # Sample n_neg from neg_idx
        sampled_neg_idx = np.random.default_rng().choice(neg_idx, chr_sample_size, replace=False)
        # Get sequence for each sampled negative
        for idx in sampled_neg_idx:
            start = idx - seq_length // 2
            end = idx + seq_length // 2
            seq = chr_dict[chr_name][start:end]
            sampled_negatives.append([chr_name, start, end, seq, -1, '.', np.nan, np.nan, "255,0,0", 'no_sample'])
    # Remove extra negatives
    sampled_negatives = sampled_negatives[:n_neg]
    neg_df = pd.DataFrame(sampled_negatives, columns=['chr', 'start', 'end', 'sequence', 'label', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'sample'])
    return neg_df

def get_bed(tf_name):
    print(f'Reading bed file for {tf_name}')
    bed_base_path = '/home/ubuntu/shared_data/remap_tfs'
    bed_file_path = os.path.join(bed_base_path, tf_name + '.bed')
    bed_cols = ['chr', 'start', 'end', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb']
    bed_df = pd.read_csv(bed_file_path, sep='\t', header=None, names=bed_cols)
    all_chr_list = get_all_chrs()
    # Keep chrs in all_chr_list
    bed_df = bed_df[bed_df['chr'].isin(all_chr_list)]
    return bed_df

def get_all_tf_names():
    embedding_dirs = '/home/ubuntu/shared_data/factor_DNA_binding_emb_esm2_t36_3B'
    TF_dirs = '/home/ubuntu/shared_data/remap_tfs'
    embeddings = os.listdir(embedding_dirs)
    tfs = os.listdir(TF_dirs)
    emb_name = [emb.split('.')[0] for emb in embeddings]
    tf_name = [tf.split('.')[0] for tf in tfs]
    overlap = set(emb_name).intersection(set(tf_name))
    print('Overlapping TFs: ', len(overlap))
    return sorted(list(overlap))

if __name__ == '__main__':
    main()
