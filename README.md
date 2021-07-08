# Tackling Attribute Fine-grainedness in Cross-modal Fashion Search with Multi-level Features

Hello welcome to the repository for the paper: Tackling Attribute Fine-grainedness in Cross-modal Fashion Search with Multi-level Features. 

A quick outline of the different folders:
- fashion_bert: FashionBERT model 
- SimCLR_pre: pre_trained simCLR used for generating features 
- comb: main model based on SCAN 
- data: data folder
- laenen: implementation of laenen cross-modal paper
- vilbert_beta: vilbert model 
- visualize: scripts to visualize attention of SCAN-model


# Requierements
We performed this work in a conda environment. In the repository you can find a txt-file (requirements.txt) with all the packages. In this way the environment can be replicated. 

# How to Run
python comb/exp_scan.py --vocab_path PATH --data_path PATH

# Data
The data folders for the Fashion200K and Fashion-Gen dataset should be placed inside the data folder. Afterwards run the following scripts to create features:
Fashion200K: comb/util/generate_tsv_ken.py  
Fashion-Gen: comb/util/generate_fashion_gen.py
