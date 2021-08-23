# BERT6mA
This package is used for the prediction of 6mA sites.  

# features
・BERT6mA provides predictions of 6mA sites for 11 species.  
・In addition to the predictions, BERT6mA serves the attention weights generated during the prediction.  

# environment
    python   : 3.7.9
    anaconda : 4.9.2
※We recommend creating a virtual environment by using anaconda.  

# preparation and installation
## 0. Preparation of a virtual environment
0-1. Creating a virtual environment.  
    `$ conda create -n [virtual environment name] python==3.7.9`  
    ex)  
    `$ conda create -n 6mAbert_ve python==3.7.9`  
    
0-2. Activating the virtual environment  
    `$ conda activate [virtual environment name]`  
    ex)  
    `$ conda activate 6mAbert_ve`  
    
## 1. Installing the BERT6mA package
Execute the following command in the directory where the package is located.  
`$ pip install ./BERT6mA/dist/BERT6mA-0.0.1.tar.gz`  

## 2. Prediction
`$ bert6mA -i [Predictive file path (fasta format)] -o [output dir path] -sp [species] -threshold [threshold to  determine whether 6mA or non-6mA] -batch [batch size]`  

ex)  
`$ bert6mA -i ~/BERT6mA/data/sample.fasta -o ~/BERT6mA/results -sp A.thaliana -threshold 0.5 -batch 128`  

※The length of sequences must be 41 long.  

Specify the species from among the following:  
    A.thaliana  
    C.elegans  
    C.equisetifolia  
    D.melanogaster  
    F.vesca  
    H.sapiens  
    R.chinensis  
    S.cerevisiae  
    T.thermophile  
    Ts.SUP5-1  
    Xoc.BLS256  
    
## 3. results
CSV and pickle files will be output to the specified directory.  
File name: results.csv, attention_weights.pkl  

results.csv: Predictive results are saved.  
    id: sequence       id
    probability:       predictive probabilities  
    predictive labels: predictive labels  
    
attention_weights.pkl: Attention weights are saved.  
    The shape of the data is (sample, layer, head, query (sequence), key (sequence))  

              














