# Cross-Link: Unimodal end-to-end deep learning model to predict how genomic variations influence protein structure + DNA/RNA binding affinities
Authors: Jimin Tan, Jingye Wang, Shentong Mo, Xi Fu, Yikai Luo, Yuhan Hao

![Cross-link]([[http://url/to/img.png](https://github.com/crosslink-bioml/crosslink/blob/main/data/schema_crosslink.png?raw=true)](https://github.com/crosslink-bioml/crosslink/blob/main/data/schema_crosslink.png?raw=true))
We present an innovative strategy aimed at elucidating the impact of oncogenic mutations on nucleic acid-binding proteins (NBPs)s by developing a end-to-end deep learning model. This model predicts how coding mutation influence nucleic acid-binding proteins (NBPs), which are master regulators of cell identity, and their subsequent DNA/RNA binding affinities. We propose three key aims: (1) the identification of DNA/RNA binding domains in NBP sequences using a protein language model (ESM2), (2) the prediction of DNA/RNA binding affinity based on protein structure embeddings and sequence information, and (3) an exploration of the functional outcomes of mutational events that alter NBP binding. We have employed a wide variety of datasets, such as the Cancer Genome Atlas and the International Cancer Genome Consortium, to provide comprehensive insights. This integrative approach facilitates the understanding of the gene regulation mechanisms in cancer, potentially identifying novel oncogenic targets, and providing a basis for the development of next-generation precision therapeutics. For seamless integration into existing biotech workflows, we created an intuitive user interface for our model, promoting widespread application across the industry.

## Training
```bash
git clone git@github.com:crosslink-bioml/crosslink.git
cd crosslink
conda env create -f environment. yml
crosslink/src/model/run.sh
```

## Demo (Gradio app)
```bash
cd crosslink/src/demo
gradio gradioapp.py
```

## Citation
https://github.com/lucidrains/bidirectional-cross-attention for inspiration on joint cross attention.
