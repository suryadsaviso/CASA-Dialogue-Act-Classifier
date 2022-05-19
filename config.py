import torch
import os

config = {
    
    # data 
    # "data_dir":os.path.join('//mnt/d/Programs/NLP/utils/CASA-Dialogue-Act-Classifier-main', 'data'),#os.getcwd(), 'data'),
    "data_dir":os.path.join(os.getcwd(), 'data'),
    "dataset":"switchboard",
    #"text_field":"clean_text",
    #"label_field":"act_label_1",
    "text_field":"Text",
    "label_field":"DamslActTag",

    "max_len":256,
    "batch_size":64,
    "num_workers":48,
    
    # model
    "model_name":"roberta-base", #roberta-base
    "hidden_size":768,
    "num_classes":43, # there are 43 classes in switchboard corpus
    
    # training
    # "save_dir":os.path.join('//mnt/d/Programs/NLP/utils/CASA-Dialogue-Act-Classifier-main', 'output'),
    "save_dir":os.path.join(os.getcwd(), 'output'),
    "project":"dialogue-act-classification",
    "run_name":"context-aware-attention-dac",
    "lr":1e-5,
    "monitor":"val_accuracy",
    "min_delta":0.001,
    "filepath":"./checkpoints/{epoch}-{val_accuracy:4f}",
    "precision":32,
    "average":"micro",
    "epochs":100,
    "device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "restart":False,
    "restart_checkpoint":"./checkpoints/epoch=10-val_accuracy=0.720291.ckpt" 
}