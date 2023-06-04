import gradio as gr
import plotly.graph_objects as go
import pymde
import esm
import json
import torch
import sys
import numpy as np
sys.path.append('../model')
from model import TFBindingCrossAttentionModel, TFBindingModel
from dataset import TFBindingDataset
torch.cuda.empty_cache()

data_dir = '/home/ubuntu/demo_session/demo_data/'
model_path = {"DNA":'/home/ubuntu/demo_session/demo_data/cross_atten_model.pt', "RNA":'/home/ubuntu/demo_session/demo_data/rbp_model_10.pt'}

# Load models
def load_esm_model():
    print('Loading ESM model...')
    esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    esm_model.cuda()
    batch_converter = alphabet.get_batch_converter()
    print('Done.')
    return esm_model, batch_converter, alphabet
class Args:
    def __init__(self, joint_cross_attn_depth):
        self.joint_cross_attn_depth = joint_cross_attn_depth

def load_model(mode = 'DNA'):
    print(f'Loading {mode} Binding model...')
    if mode == 'DNA':
      model = TFBindingCrossAttentionModel(Args(joint_cross_attn_depth=1))
    elif mode == 'RNA':
        model = TFBindingModel()

    model.load_state_dict(torch.load(model_path[mode]))
    model.eval()
    model.cuda()
    print('Done.')
    return model

# Raw Sequence Encoding
def dna_seq_encode(seq):
    import numpy as np
    seq_dict = {'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4}
    seq_onehot = np.zeros((5, len(seq)))
    for i, char in enumerate(seq):
        seq_onehot[seq_dict[char], i] = 1
    seq_tensor = torch.tensor(seq_onehot)
    return seq_tensor

def amino_acid_encode(amino_acid_seq):
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
    cropped_embedding = padded_embedding[:, 50:250]
    return cropped_embedding


def amino_acid_mut_encode(amino_acid_seq):
    return amino_acid_encode(amino_acid_seq), amino_acid_seq

def esm_atlas(query_embedding):
    ref_embedding = torch.load(data_dir + '/all_protein_esm_matrix.pt').cuda()
    query_embedding = query_embedding.mean(1).view(1, -1)
    ref_query_emb = torch.cat((ref_embedding, query_embedding), dim=0)
    with open(data_dir + '/all_protein_metadata.json', 'r') as file:
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
def predict(amino_acid_seq, dna_seq, mode = 'DNA'):
    global model
    model = load_model(mode)
    tf_embedding = amino_acid_encode(amino_acid_seq)
    #torch.save(tf_embedding, 'tf_embedding.pt')
    #tf_embedding = torch.load('tf_embedding.pt')
    onehot_seq = dna_seq_encode(dna_seq)

    pred = model_inference(tf_embedding, onehot_seq)
    if mode == 'RNA':
        pred = float(np.mean(pred)) 
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
    tf_root_path = data_dir + '/factor_DNA_binding_emb_esm2_t36_3B'
    
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
    chr_path = f'{data_dir}/dna_sequence/{chr_name}.fa.gz'
    print(f'Reading sequence: {chr_path}')
    import gzip
    with gzip.open(chr_path, 'r') as f:
        seq = f.read().decode("utf-8")
    seq = seq[seq.find('\n'):]
    seq = seq.replace('\n', '').lower()
    return seq


def compare_wt_mut_tf(chr_name, start, end, radius, wt_tf, mut_tf):
    fig_wt, pred_list_wt = validate_local_region(chr_name, start, end, radius, wt_tf)
    fig_mut, pred_list_mut = validate_local_region(chr_name, start, end, radius, mut_tf, tf_type = 'Mutated')

    # Plot difference
    window_size = 100
    step_size = 20
    import numpy as np
    pred_diff = np.array(pred_list_mut) - np.array(pred_list_wt)
    fig_diff = go.Figure(data=[go.Bar(x=[l - radius for l in list(range(0, len(pred_diff) * step_size, step_size))], y=pred_diff)])
    fig_diff.update_layout(title_text='TF Binding Prediction Difference', xaxis_title='Position to Promoter (bp)', yaxis_title='Predicted Binding Difference')
    #fig_diff.update_yaxes(range=[-1, 1])
    return fig_wt, fig_mut, fig_diff

def compare_wt_mut_tf_mdm2(wt_tf, mut_tf):
    # MDM2
    chr_name= 'chr12'
    tss = 68807024 
    radius = 1000
    start = tss - radius
    end = tss + radius
    return compare_wt_mut_tf(chr_name, start, end, radius, wt_tf, mut_tf)

def compare_wt_mut_tf_bax(wt_tf, mut_tf):
    # BAX
    chr_name= 'chr19'
    tss = 48954932
    radius = 1000
    start = tss - radius
    end = tss + radius
    return compare_wt_mut_tf(chr_name, start, end, radius, wt_tf, mut_tf)

def compare_wt_mut_tf_kmt2a(wt_tf, mut_tf):
    # KMT2A
    chr_name= 'chr11'
    tss = 118436755
    radius = 1000
    start = tss - radius
    end = tss + radius
    return compare_wt_mut_tf(chr_name, start, end, radius, wt_tf, mut_tf)

def validate_local_region(chr_name, start, end, radius, tf_embedding, tf_type = 'Wild Type'):
    root_sequence_path = data_dir + '/dna_sequence'
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
    fig.update_layout(title_text=f'TF Binding Prediction ({tf_type})', xaxis_title='Position to Promoter (bp)', yaxis_title='Predicted Binding, log (x + 1) scaled')
    fig.update_yaxes(range=[0, 7])
    return fig, pred_list

def validate_local_region_mdm2(tf_embedding):
    # MDM2
    chr_name= 'chr12'
    tss = 68807024 
    radius = 1000
    start = tss - radius
    end = tss + radius
    return validate_local_region(chr_name, start, end, radius, tf_embedding)[0]

def validate_local_region_bax(tf_embedding):
    # BAX
    chr_name= 'chr19'
    tss = 48954932
    radius = 1000
    start = tss - radius
    end = tss + radius
    return validate_local_region(chr_name, start, end, radius, tf_embedding)[0]

def validate_local_region_kmt2a(tf_embedding):
    # KMT2A
    chr_name= 'chr11'
    tss = 118436755
    radius = 1000
    start = tss - radius
    end = tss + radius
    return validate_local_region(chr_name, start, end, radius, tf_embedding)[0]

def change_model(choice = "DNA"):
    model = load_model(choice)
    if choice == "RNA":
        return gr.update(lines=5, visible=True, label="RNA input sequences: ", placeholder = 'ccggatggtgcact...'), choice
    else:
        return gr.update(lines=5, visible=True, label="DNA input sequences: ", placeholder = 'gcaggggggcactc...'), choice



# Analysis

esm_model, batch_converter, alphabet = load_esm_model()
model = None
dataset = TFBindingDataset()
demo = gr.Blocks()

with demo:
    with gr.Tab("Introduction"):
        gr.Markdown(
    """
    # Introduction
    Crosslink is a deep learning tool designed to predict binding events based on nucleic acid sequences (DNA/RNA) and protein structure. This model enables identifying potential changes in binding sites due to specific protein mutations. Crosslink operates by utilizing separate encoders for protein and nucleic acid sequences which are subsequently fused via an attention network to predict binding occurrences. This computation ChIP-seq/eCLIP-seq tool offers a unique approach to understanding protein - DNA/RNA interactions and their potential alterations. For further details, please visit our [GitHub repository](https://github.com/crosslink-bioml/crosslink).
    """)
        gr.Markdown(
    """
    ## Background
    Mutational events that alter protein functions and structures are the most prominent molecular drivers of cancer progression. Unfortunately, the vast diversity of these events complicates therapeutic strategies. Large-scale mutations, including indels, structural variations, and alternative-splicing-induced alterations, are abundant in cancers but are challenging to study due to a lack of appropriate assays. However, nucleic acid-binding proteins (NBPs), encompassing transcription factors and RNA-binding proteins, are potentially ideal subjects for understanding the impacts of such mutations due to their integral role as master regulators of cell identities and long-term community efforts in comprehensively mapping their molecular targets. Recent progress in deep learning has facilitated precise modeling of biophysical and genomic mechanisms, including protein folding. Despite this, no prior study has successfully integrated all elements of the DNA/RNA-NBP binding process into a singular model. 
    Our project aims to fill this gap by developing a unimodal end-to-end deep learning model to predict how genomic variations influence protein structure and subsequent DNA/RNA binding affinities. Our sequence-structure-binding model is capable of processing mutated NBPs originating from oncogenic processes, and quantifying mutation-caused differential binding affinities. This approach will provide the scientific community with a unique opportunity to examine gene regulation from a novel and deep mechanistic perspective, and is the first step towards identifying novel oncogenic targets created through remodeled NBP binding in cancer.
    We envision that this model could be further refined to engineer proteins that specifically target these NBPs on an individual basis, serving as the foundation of a new generation of precision therapeutics. Appreciating the importance of seamless integration into existing workflows in the biotech community, we aim to develop a user-friendly, intuitive interface for our deep learning model, which will facilitate its application across various sectors of the industry.  
    ![background](file/schema_crosslink.png)
    
    ## Modeling DNA/RNA binding dynamics through deep learning
    ### Aim 1. Identify DNA/RNA binding domains based on NBP amino acid sequences
    The most relevant entities in the binding process are the nucleic-acid-binding protein domains and their target sequences. While the latter can be identified using high-throughput sequencing assays at scale, pinpointing the former requires extensive biochemical efforts thus not always completely annotated in canonical databases. Here we propose an automatic annotation pipeline to identify these domains using pretrained protein structure model.

    ### Aim 2: Predict DNA/RNA binding affinity based on protein structure embeddings
    The structure modality provides a more mechanistic understanding of the nature of protein-DNA/RNA interactions, while the sequence modality is indicative of functional importance and is usually the most readily utilizable modality in clinical practices. Thus, we aim to build a multimodal model to bridge the structure and sequence information from both protein and nucleic acids. 

    ### Aim 3: Investigate the functional consequences of NBP-binding-altering large-scale mutational events
    Our NBP-affinity-scoring model serves as the central infrastructure to a streamlined pipeline that interrogates the phenotypic importance of NBP-binding-perturbing large-scale cancer mutations present in real-world human patients.

    ![workflow](file/workflow.png)
    ## Deep learning model archtecture
    Our model uses a combination of convolutional and attention mechanisms to process DNA/RNA sequence and protein structure-based features. Here is a description of its architecture:
    The model is composed of several components:ConvBlock, ESMEncoder, SeqEncoder, NBP-BindingCrossAttentionModel, and NBP-BindingModel.

    Overall, this architecture is designed to extract complex features from both sequence and structural data and combine them to predict if protein and DNA/RNA can bind together. It makes use of both convolutional and attention mechanisms, which allow it to model both local and global interactions in the data.

    ![deep learning model archtecture](file/deep_learing_model_archtecture.png)
    """)
        
        
    with gr.Tab("Analysis"):
        gr.Markdown(
        """
        # Data input
        ## Protein
        In the below sections, you can enter the amino acid sequences for both wild type and mutant proteins. Our default example features the wild type TP53, a tumor suppressor protein, alongside its most common mutation, R248Q. However, you are welcome to input your own protein sequences to cater the analysis to your specific needs.

        """)
    
        amino_acid_text = gr.Textbox(label = 'Amino Acid Sequence', lines = 5, placeholder = 'GRGRHPGKGVK...')
        gr.Examples(['TYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRK', # TP53
                    'GRGRHPGKGVKSPGEKSRYETSLNLTTKRFLELLSHSADGVVDLNWAAEVLKVQKRRIYDITNVLEGIQLIAKKSKNHIQWLGS'], inputs=amino_acid_text)

        amino_acid_mut_text = gr.Textbox(label = 'Amino Acid Sequence', lines = 5, placeholder = 'GRGRHPGKGVK...')
        gr.Examples(['TYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVIVCACPGRDRRTEEENLRK', # TP53 Mut R273I
                    'TYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVKVCACPGRDRRTEEENLRK', # TP53 Mut R273K
                    ], inputs=amino_acid_mut_text)

        gr.Markdown(
        """
        ## DNA/RNA
        In the below sections, you can also input your own sequences to cater the analysis to your specific needs in the FASTA format.  The default DNA input sequences are some cell cycle related genes (such as MDM2, BAX2). 
        """)

        CHOICE = gr.Radio(
            ["DNA", "RNA"], label="Essay Length to Write?"
        )

        dna_seq_text = gr.Textbox(lines=2, interactive=True)
        model_choice = gr.State([])
        CHOICE.change(fn=change_model, inputs=CHOICE, outputs=[dna_seq_text,model_choice])
        gr.Examples(['agagggcggagcactcccgtgccccggggcaggagtgcagggagctcccgcgcccggaacgttgcgagcaaggcttgcgagcgtcgcaggggggcactcg'], inputs=dna_seq_text)
    

        gr.Markdown(
        """
        ## ESM2 visualization of your input proteins and our pretrained proteins
        In the following section, you can visualize the ESM2 embeddings derived from the esm2_t36_3B_UR50D model for both your input proteins and our pre-trained proteins. Each protein is represented in 2560 dimensions. We have applied pyMDE to generate two-dimensional visualization of these embeddings. If you find that your proteins significantly differ from our pre-trained protein set, the accuracy of the prediction might be compromised. This discrepancy could occur due to deviations outside the trained protein distribution.
        """)
        aa_plot = gr.Plot()
        tf_embedding_target = gr.State([])
        button = gr.Button('Visualize Amino Acid Embedding')
        button.click(esm_visulization, inputs = [amino_acid_text], outputs = [aa_plot, tf_embedding_target])
        gr.Markdown(
        """
        # Crosslink Prediction
        ## Binding affinity prediction
        """)
        output = gr.Label(label = 'Binding Affinity Prediction, log (x + 1) scaled')
        onehot_seq = gr.State([])
        tf_embedding = gr.State([])
        mut_tf_embedding = gr.State([])
    
        button = gr.Button('Predict')
        button.click(predict, inputs = [amino_acid_text, dna_seq_text, model_choice], outputs = [output, onehot_seq, tf_embedding])

        mutation_label = gr.Label(label = 'Mutation Sequence Processed')
        button = gr.Button('Process Mutation')
        button.click(amino_acid_mut_encode, inputs = [amino_acid_mut_text], outputs = [mut_tf_embedding, mutation_label])

        gr.Markdown(
        """
        ## Comparison with other TFs binding affinity
        """)
        tf_comparison_plot = gr.Plot()
        botton = gr.Button('Compare with other TFs')
        botton.click(tf_comparison, inputs = [dna_seq_text, output], outputs = [tf_comparison_plot])
        gr.Markdown(
        """
        ## Promoter binding difference between wild type and mutant of protein
        """)
        gr.Markdown(
        """
        # MDM2 promoter 
        """)
        mdm2_plot = gr.Plot()
        mdm2_button = gr.Button('Generate binding profile for MDM2 promoter')
        mdm2_button.click(validate_local_region_mdm2, inputs = [tf_embedding], outputs = mdm2_plot)

        mdm2_plot_wt = gr.Plot()
        mdm2_plot_mut = gr.Plot()
        mdm2_plot_diff = gr.Plot()
        mdm2_mut_button = gr.Button('Compare mutated on MDM2 promoter region')
        mdm2_mut_button.click(compare_wt_mut_tf_mdm2, inputs = [tf_embedding, mut_tf_embedding], outputs = [mdm2_plot_wt, mdm2_plot_mut, mdm2_plot_diff])
        gr.Markdown(
        """
        # BAX promoter 
        """)
        bax_plot = gr.Plot()
        bax_button = gr.Button('Generate binding profile for BAX promoter')
        bax_button.click(validate_local_region_bax, inputs = [tf_embedding], outputs = bax_plot)

        bax_plot_wt = gr.Plot()
        bax_plot_mut = gr.Plot()
        bax_plot_diff = gr.Plot()
        bax_mut_button = gr.Button('Compare mutated on BAX promoter region')
        bax_mut_button.click(compare_wt_mut_tf_bax, inputs = [tf_embedding, mut_tf_embedding], outputs = [bax_plot_wt, bax_plot_mut, bax_plot_diff])


        gr.Markdown(
        """
        # KMT2A promoter
        KMT2A is a gene that encodes a histone methyltransferase. KMT2A is most notorious for its role in acute leukemia. Mutations and changes in its expression have also been found in solid tumors, such as lung, colorectal, and gastric cancers.
        """)
        kmt2a_plot = gr.Plot()
        kmt2a_button = gr.Button('Generate binding profile for KMT2A promoter')
        kmt2a_button.click(validate_local_region_kmt2a, inputs = [tf_embedding], outputs = kmt2a_plot)

        kmt2a_plot_wt = gr.Plot()
        kmt2a_plot_mut = gr.Plot()
        kmt2a_plot_diff = gr.Plot()
        kmt2a_mut_button = gr.Button('Compare mutated on KMT2A promoter region')
        kmt2a_mut_button.click(compare_wt_mut_tf_kmt2a, inputs = [tf_embedding, mut_tf_embedding], outputs = [kmt2a_plot_wt, kmt2a_plot_mut, kmt2a_plot_diff])

demo.launch(share=True)  
