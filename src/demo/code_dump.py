import gradio as gr

import sys
sys.path.append('../model')
import torch
from model import TFBindingModel
from dataset import TFBindingDataset

def load_model(model_path = '../../runs/2023-05-30_17-24-51/models/model_9.pt'):
    model = TFBindingModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()
    return model

model = load_model()
dataset = TFBindingDataset()

def predict(idx, model):
    idx = int(idx)

    data = dataset.data.iloc[dataset.data_idx[idx]]
    seq = data['name']
    tf_name = data['tf_name']
    label = data['label']
    onehot_seq = dataset.onehot_encode(seq)
    tf_embedding = dataset.tf_embeddings[tf_name]

    pred = model_inference(onehot_seq, tf_embedding)
    pred_str = str(pred)
    return pred_str

def model_inference(tf_embedding, dna_sequence_onehot):
    dna_sequence_onehot = torch.tensor(dna_sequence_onehot).unsqueeze(0).cuda().float()
    tf_embedding = torch.tensor(tf_embedding).unsqueeze(0).cuda().float()
    pred = model(tf_embedding, dna_sequence_onehot).squeeze().detach().cpu().numpy()
    return pred

def dna_seq_to_input(dna_seq):
    number = 0
    import numpy as np
    img = np.random.rand(28, 28, 3)
    return number

def amino_acid_to_input(amino_acid):
    pass


demo = gr.Blocks()

with demo:
    dna_seq_text = gr.Textbox(label = 'DNA Sequence', lines = 5, placeholder = 'Enter DNA Sequence')
    amino_acid_text = gr.Textbox(label = 'Amino Acid Sequence', lines = 5, placeholder = 'Enter Amino Acid Sequence')
    amino_acid_embedding_img = gr.Image(label = 'Amino Acid Embedding', shape = (28, 28, 3))

    output = gr.Label()

    button = gr.Button("Submit")
    button.click(predict, inputs = dna_seq_text, outputs = output)


    with gr.Row():
        gr.Image()
        gr.Image()

    gr.Image(label = 'CDK1A Locus')
    gr.Image(label = 'MDM2 Locus')
    gr.Image(label = 'DAX Locus')

    gr.Image(label = 'Relationship in Disease State')

    #button.click(dna_seq_to_input, inputs = dna_seq_text, outputs = output)


demo.launch()  
