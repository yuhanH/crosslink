import torch
import esm

esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
esm_model.cuda()
batch_converter = alphabet.get_batch_converter()

def encode_amino_acid(amino_acid_seq):
    amino_acid_seq = 'GRGRHPGKGVKSPGEKSRYETSLNLTTKRFLELLSHSADGVVDLNWAAEVLKVQKRRIYDITNVLEGIQLIAKKSKNHIQWLGS'
    data = [['input_name', amino_acid_seq]]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to('cuda')
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[36], return_contacts=True)
    token_embedding = results["representations"][36][0]

    embedding = torch.transpose(token_embedding, 0, 1)
    # Pad or crop to 100
    pad_margin_l = (300 - embedding.shape[1]) // 2
    pad_margin_r = 300 - embedding.shape[1] - pad_margin_l
    padded_embedding = torch.nn.functional.pad(embedding, (pad_margin_l, pad_margin_r), 'constant', 0)
    # Crop
    cropped_embedding = padded_embedding[:, 100:200].unsqueeze(0)
    return cropped_embedding

embedding = encode_amino_acid('0')
breakpoint()
