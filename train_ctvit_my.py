import torch
from transformer_maskgit import CTViT, CTViTTrainer

cvivit = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 128,
    patch_size = 16,
    temporal_patch_size = 2,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8,
)
# pretrained_ctvit_path = 'pretrained_models/vae.19750.pt'
# cvivit.load(pretrained_ctvit_path)

trainer = CTViTTrainer(
    cvivit,
    folder = 'mydata/valid_data',
    batch_size = 4,
    results_folder="ctvit",
    grad_accum_every = 3,
    train_on_images = False,  # you can train on images first, before fine tuning on video, for sample efficiency
    use_ema = False,          # recommended to be turned on (keeps exponential moving averaged cvivit) unless if you don't have enough resources
    num_train_steps = 10000,
    num_frames=2,
    valid_frac = 0.05,
    log_folder="ctvit/"
)

trainer.train()               # reconstructions and checkpoints will be saved periodically to ./results
