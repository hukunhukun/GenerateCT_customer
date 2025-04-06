from transformer_maskgit import CTViT, MaskGit, MaskGITTransformer
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os
from torch.utils.data import DataLoader
from transformer_maskgit.data import tensor_to_nifti
from pathlib import Path
import glob
import pandas as pd
import random
import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
def cycle(dl):
    while True:
        for data in dl:
            yield data

def scan_folder(directory):
    # Use a set to automatically eliminate duplicates
    txt_content = set()

    # Use glob to recursively find all .txt files
    for filename in glob.glob(directory + '/**/*.txt', recursive=True):
        with open(filename, 'r') as f:
            content = f.readline().strip()
            txt_content.add(content)

    return list(txt_content)



def infer():
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

    ctvit.eval()

    # Load the pre-trained weights

    pretrained_ctvit_path = 'mymodel/ctvit_pretrained.pt'
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
    ).cuda()
    batch_size=1
    transformer_model.load('mymodel/maskgittransformer.85999.pt')
    transformer_model.eval()

    csv_file = 'mydata/valid_data/Pelvic_test.csv'
    labels_data = pd.read_csv(csv_file)

    # 构建一个字典，将 image 对应的 text 映射到一起
    texts_dict = dict(zip(labels_data['image'], labels_data['text']))

    print(texts_dict)
    #for i in range(len(texts)):
    print("Inference for the transformer model")
    for i, (input_name, text) in tqdm.tqdm(enumerate(texts_dict.items())):
        for k in range(1):
            out=transformer_model.sample(texts =text, num_frames = 129, cond_scale = 5.)
            path_test=Path("transformer_inference")
            sampled_videos_path = path_test / f'samples_{str(i)}'
            (sampled_videos_path).mkdir(parents = True, exist_ok = True)
            for tensor in out.unbind(dim = 0):
                tensor_to_nifti(tensor, str(sampled_videos_path / f'{input_name}_{k}.nii.gz'))
                img = nib.load(str(sampled_videos_path / f'{input_name}_{k}.nii.gz'))
                data = img.get_fdata()
                slice_idx = data.shape[2] // 2
                slice_data = data[:, :, slice_idx]
                plt.imsave(str(sampled_videos_path / f'{input_name}_{k}.png'), slice_data, cmap='gray')
                # Save the text to a file
                filename = str(sampled_videos_path / f'{input_name}_{k}.txt')
                with open(filename, "w", encoding="utf-8") as file:
                    file.write(text)

if __name__ == '__main__':
    # set up multiprocessing
    infer()
