import gradio as gr
import plotly.graph_objects as go
import pymde
import esm
 
import torch

import sys
sys.path.append('../model')
from model import TFBindingModel
from dataset import TFBindingDataset

# Load models
def load_esm_model():
    print('Loading ESM model...')
    esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    esm_model.cuda()
    batch_converter = alphabet.get_batch_converter()
    print('Done.')
    return esm_model, batch_converter, alphabet


def load_model(model_path = '/home/ubuntu/demo_session/demo_data/model_9.pt'):
    print('Loading TF Binding model...')
    model = TFBindingModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()
    print('Done.')
    return model

# Raw Sequence Encoding
def dna_seq_encode(seq = 'agagggcggagcactcccgtgccccggggcaggagtgcagggagctcccgcgcccggaacgttgcgagcaaggcttgcgagcgtcgcaggggggcactcg'):
    import numpy as np
    seq_dict = {'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4}
    seq_onehot = np.zeros((5, len(seq)))
    for i, char in enumerate(seq):
        seq_onehot[seq_dict[char], i] = 1
    seq_tensor = torch.tensor(seq_onehot)
    return seq_tensor

def amino_acid_encode(amino_acid_seq = 'GRGRHPGKGVKSPGEKSRYETSLNLTTKRFLELLSHSADGVVDLNWAAEVLKVQKRRIYDITNVLEGIQLIAKKSKNHIQWLGS'):
    global esm_model, batch_converter, alphabet

    data = [['input_name', amino_acid_seq]]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = esm_model(batch_tokens.cuda(), repr_layers=[36], return_contacts=True)
    token_embedding = results["representations"][36][0]

    embedding = torch.transpose(token_embedding, 0, 1)
    # Pad or crop to 100
    pad_margin_l = (300 - embedding.shape[1]) // 2
    pad_margin_r = 300 - embedding.shape[1] - pad_margin_l
    padded_embedding = torch.nn.functional.pad(embedding, (pad_margin_l, pad_margin_r), 'constant', 0)
    # Crop
    cropped_embedding = padded_embedding[:, 100:200]
    return cropped_embedding


def esm_atlas(query_embedding):
    ref_embedding = torch.load('/home/ubuntu/demo_session/demo_data/all_protein_esm_matrix.pt').cuda()
    query_embedding = query_embedding.mean(1).view(1, -1)
    ref_query_emb = torch.cat((ref_embedding, query_embedding), dim=0)
    with open('/home/ubuntu/demo_session/demo_data/all_protein_metadata.json', 'r') as file:
        ref_meta = json.load(file)
    meta_data = ref_meta + [{'label': 'query'}]
    mde_emb = pymde.preserve_neighbors(ref_query_emb, embedding_dim=2, verbose=True, device="cuda", repulsive_fraction = 1.2).embed()
    mde_emb = mde_emb.detach().cpu().numpy()
    labels = [x['label'] for x in meta_data]
    colors = ['lightgrey'] * ref_embedding.shape[0] + ['red']

    scatter = go.Scatter(
    x=mde_emb[:, 0], 
    y=mde_emb[:, 1], 
    mode='markers',
    marker=dict(
        size=10,
        color=colors, #set color equal to a variable
    ),
    text=labels # set hover text
        )
    fig = go.Figure(data=scatter)
    fig.update_layout(title_text='Protein ESM visualization')
    
    return fig

def esm_visulization(amino_acid_seq):
    query_emb = amino_acid_encode( amino_acid_seq)
    fig = esm_atlas(query_emb)
    return fig, query_emb
# Prediction
def predict(amino_acid_seq, dna_seq):
    #tf_embedding = amino_acid_encode(amino_acid_seq)
    #torch.save(tf_embedding, 'tf_embedding.pt')
    tf_embedding = torch.load('tf_embedding.pt')
    onehot_seq = dna_seq_encode(dna_seq)

    pred = model_inference(tf_embedding, onehot_seq)
    pred = float(pred)
    return pred, onehot_seq, tf_embedding

def model_inference(tf_embedding, dna_sequence_onehot):
    tf_embedding = torch.tensor(tf_embedding).unsqueeze(0).cuda().float()
    dna_sequence_onehot = torch.tensor(dna_sequence_onehot).unsqueeze(0).cuda().float()
    with torch.no_grad():
        pred = model(dna_sequence_onehot, tf_embedding).squeeze().detach().cpu().numpy()
    return pred

def tf_comparison(dna_seq, current_pred):
    # Load tf embeddings ESR1, ERF, FOXP1, POU3F2
    tf_list = ['Current TF', 'ESR1', 'ERF', 'FOXP1', 'POU3F2']
    tf_root_path = '/home/ubuntu/demo_session/demo_data/factor_DNA_binding_emb_esm2_t36_3B'
    
    onehot_seq = dna_seq_encode(dna_seq)
    pred_list = [current_pred['label']]
    for tf in tf_list:
        if tf == 'Current TF': continue
        tf_embedding = torch.transpose(torch.load(f'{tf_root_path}/{tf}.pt')['representations'][36], 0, 1)
        pred = model_inference(tf_embedding, onehot_seq)
        pred_list.append(pred)

    # Plotly bar chart
    fig = go.Figure(data=[go.Bar(x=tf_list, y=pred_list)])
    fig.update_layout(title_text='TF Binding Prediction', xaxis_title='TF', yaxis_title='Prediction')
    fig.update_yaxes(range=[0, 7])
    return fig

def load_chr(chr_name):
    chr_path = f'/home/ubuntu/demo_session/demo_data/dna_sequence/{chr_name}.fa.gz'
    print(f'Reading sequence: {chr_path}')
    import gzip
    with gzip.open(chr_path, 'r') as f:
        seq = f.read().decode("utf-8")
    seq = seq[seq.find('\n'):]
    seq = seq.replace('\n', '').lower()
    return seq

def validate_local_region(chr_name, start, end, radius, tf_embedding):
    root_sequence_path = '/home/ubuntu/demo_session/demo_data/dna_sequence'
    dna_seq = load_chr(chr_name)[start:end]
    window_size = 100
    step_size = 20
    pred_list = []
    for i in range(0, len(dna_seq) - window_size, step_size):
        onehot_seq = dna_seq_encode(dna_seq[i:i+window_size])
        pred = model_inference(tf_embedding, onehot_seq)
        pred_list.append(pred)

    # Plotly bar chart
    fig = go.Figure(data=[go.Bar(x=[l - radius for l in list(range(0, len(dna_seq) - window_size, step_size))], y=pred_list)])
    fig.update_layout(title_text='TF Binding Prediction', xaxis_title='Position to Promoter (bp)', yaxis_title='Predicted Binding, log (x + 1) scaled')
    fig.update_yaxes(range=[0, 7])
    return fig

def validate_local_region_mdm2(tf_embedding):
    # MDM2
    chr_name= 'chr12'
    tss = 68807024 
    radius = 1000
    start = tss - radius
    end = tss + radius
    return validate_local_region(chr_name, start, end, radius, tf_embedding)

def validate_local_region_bax(tf_embedding):
    # MDM2
    chr_name= 'chr19'
    tss = 48954932
    radius = 1000
    start = tss - radius
    end = tss + radius
    return validate_local_region(chr_name, start, end, radius, tf_embedding)

# Analysis

esm_model, batch_converter, alphabet = load_esm_model()
model = load_model()
dataset = TFBindingDataset()
demo = gr.Blocks()

with demo:
    amino_acid_text = gr.Textbox(label = 'Amino Acid Sequence', lines = 5, placeholder = 'GRGRHPGKGVK...')
    gr.Examples(['GRGRHPGKGVKSPGEKSRYETSLNLTTKRFLELLSHSADGVVDLNWAAEVLKVQKRRIYDITNVLEGIQLIAKKSKNHIQWLGS'], inputs=amino_acid_text)
    dna_seq_text = gr.Textbox(label = 'DNA Sequence', lines = 5, placeholder = 'gcaggggggcactc...')
    gr.Examples(['agagggcggagcactcccgtgccccggggcaggagtgcagggagctcccgcgcccggaacgttgcgagcaaggcttgcgagcgtcgcaggggggcactcg'], inputs=dna_seq_text)
  

    gr.Markdown(
    """
    ESM embedding of the amino acid sequence with our pretrained proteins
    """)
    aa_plot = gr.Plot()
    tf_embedding_target = gr.State([])
    button = gr.Button('Visualize Amino Acid Embedding')
    button.click(esm_visulization, inputs = [amino_acid_text], outputs = [aa_plot, tf_embedding_target])


    output = gr.Label(label = 'Binding Affinity Prediction, log (x + 1) scaled')
    onehot_seq = gr.State([])
    tf_embedding = gr.State([])

    button = gr.Button('Predict')
    button.click(predict, inputs = [amino_acid_text, dna_seq_text], outputs = [output, onehot_seq, tf_embedding])

    tf_comparison_plot = gr.Plot()
    botton = gr.Button('Compare with other TFs')
    botton.click(tf_comparison, inputs = [dna_seq_text, output], outputs = tf_comparison_plot)

    mdm2_plot = gr.Plot()
    mdm2_button = gr.Button('Generate binding profile for MDM2 promoter')
    mdm2_button.click(validate_local_region_mdm2, inputs = [tf_embedding], outputs = mdm2_plot)

    bax_plot = gr.Plot()
    bax_button = gr.Button('Generate binding profile for BAX promoter')
    bax_button.click(validate_local_region_bax, inputs = [tf_embedding], outputs = bax_plot)

    #gr.Image(label = 'Relationship in Disease State')

demo.launch()  
