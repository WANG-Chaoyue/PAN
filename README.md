# PAN
[Perceptual Adversarial Networks (PAN) for Image-to-Image Transformation](https://arxiv.org/abs/1706.09138).

## Getting started

- Clone this repo:
```bash
git clone https://github.com/WANG-Chaoyue/PAN.git
cd PAN
```

- Install [Theano 1.0.0+](http://deeplearning.net/software/theano/install.html), [lassagne 0.2+](https://lasagne.readthedocs.io/en/latest/user/installation.html).

### Datasets
The training datasets were similar [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Therefore, we borrowed the code of dataset download from them. 
- Download facades dataset:
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

### Training
- Train a model
```bash
python train_facades.py
```


### Test
- Test a model
```bash
python test_facades.py
```

## Citation
If you use this code for your research, please cite our paper.
```
@article{wang2018perceptual,
  title={Perceptual adversarial networks for image-to-image transformation},
  author={Wang, Chaoyue and Xu, Chang and Wang, Chaohui and Tao, Dacheng},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={8},
  pages={4066--4079},
  year={2018},
  publisher={IEEE}
}
```
