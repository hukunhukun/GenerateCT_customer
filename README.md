# GenerateCT 修改
原文章
```
@article{hamamci2023generatect,
  title={GenerateCT: Text-Conditional Generation of 3D Chest CT Volumes},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Sekuboyina, Anjany and Simsar, Enis and Tezcan, Alperen and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Almas, Furkan and Dogan, Irem and Dasdelen, Muhammed Furkan and others},
  journal={arXiv preprint arXiv:2305.16037},
  year={2023}
}
```
- 只保留 (128,128) 分辨率图像生成

- 个人数据训练并推理

- gradient-checkpointing 显存节省 ($\approx$ 24 GB)

## Requirements


```setup
# 在 'transformer_maskgit' 目录下安装相关库
cd transformer_maskgit
pip install -e .

# Return to the root directory
cd ..
```

## Training

Train the CT-ViT model 

```train
accelerate launch train_ctvit.py
```
To train the MaskGIT Transformer model, 

```train
accelerate launch train_transformer.py
```


## Inference


```eval
python inference_ctvit.py
```
推理出图

```eval
python inference_transformer.py
```




## Pretrained Models

预训练模型下载

- [t5-v1_1-base](https://huggingface.co/google/t5-v1_1-base)

- [mymodel/maskgittransformer.85999.pt](https://huggingface.co/kunkunhu/GenerateCT-customer)

- [mymodel/vae.9750.pt](https://huggingface.co/kunkunhu/GenerateCT-customer)

