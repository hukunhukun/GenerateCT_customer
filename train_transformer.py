from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer
from transformer_maskgit.videotextdataset import VideoTextDataset
from transformer_maskgit.train_transformer import TransformerTrainer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
def train():
    # set up distributed training

    ctvit = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 128,
        patch_size = 16,
        temporal_patch_size = 2,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )


    # Load the pre-trained weights

    pretrained_ctvit_path = 'pretrained_models/ctvit_pretrained.pt'
    ctvit.load(pretrained_ctvit_path)

    maskgit = MaskGit(
        num_tokens=8192,
        max_seq_len=10000,
        dim=512,
        dim_context=768,
        depth=6,
    )
   
    transformer_model = MaskGITTransformer(
        ctvit=ctvit,
        maskgit=maskgit
    )
    print_trainable_parameters(transformer_model)
    batch_size=1
    transformer_model.load('pretrained_models/transformer_pretrained.pt')

    # initialize DDP
    trainer = TransformerTrainer(
        transformer_model,
        num_train_steps=100000000,
        batch_size=1,
        pretrained_ctvit_path='pretrained_models/ctvit_pretrained.pt',
        results_folder="transformer_train",
        log_folder="transformer_train/",
    )


    trainer.train()

if __name__ == '__main__':
    # set up multiprocessing
    train()
