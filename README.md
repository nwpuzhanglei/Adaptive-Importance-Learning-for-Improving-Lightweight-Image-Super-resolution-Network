# PyTorch VDSR
Official implementation of IJCV2019 paper: "Adaptive Importance Learning for Improving Lightweight Image Super-resolution Network" in PyTorch

## Dependencies
  - Python 2.7
  - Pytorch >= 1.0.1
  - numpy
  - tqdm
  - h5py
  - Python 2.7

### Training
```
usage: main_ail.py [-h] [--round ROUND] [--width WIDTH] [--scale SCALE] [--batchSize BATCHSIZE] 
               [--nEpochs NEPOCHS] [--lr LR] [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--tea TEACHER] [--premodel PREMODEL] [--dataset DATASET]
               
optional arguments:
  -h, --help            Show this help message and exit
  --round               Round of adaptive importance learning (AIL)
  --width               Width of lightweight network, num of feature maps
  --scale               SR scales, e.g., 2,3,4
  --batchSize           Training batch size
  --nEpochs             Number of epochs to train for
  --lr                  Learning rate. Default=0.01
  --step                Learning rate decay, Default: n=10 epochs
  --cuda                Use cuda
  --resume              Path to checkpoint
  --clip                Clipping Gradients. Default=0.4
  --threads             Number of threads for data loader to use Default=4
  --momentum            Momentum, Default: 0.9
  --weight-decay        Weight decay, Default: 1e-4
  --tea                 path to the teacher model
  --premodel            path to pre-trained lightweight model with the traditional learning scheme
  --dataset             path to training data in .h5 file for the corresponding SR scale e.g., scale=2
```

### Test
```
usage: test.py [-h] [--cuda] [--model MODEL] [--fpath PATH] [--dataset SET] [--image IMAGE] [--scale SCALE]
               
optional arguments:
  -h, --help            Show this help message and exit
  --cuda                Use cuda
  --model               Model path. Default='model/model/ours_ail_r9_f13_s2.pth'
  --premodel            Model path. Default='model/pre_vdsr_f13.pth'
  --tea                 Model path. Default='model/model/tea_vdsr.pth'
  --fpath               Path to the test dataset. Default="./"
  --dataset             Test set name. Default="Set5"
  --image               Image name. Default="woman_GT"
  --scale               Scale factor, Default: 2
```
We use PIL for image convertion, for best PSNR performance and SSIM, please use Matlab

### Prepare Training dataset
  - We provide a simple hdf5 format training sample in data folder with 'data' and 'label' keys, the training data is generated with Matlab Bicubic Interplotation, please refer [Code for Data Generation](https://github.com/twtygqyy/pytorch-vdsr/tree/master/data) for creating training files.
### Run steps
 1. Train the teacher vdsr model (e.g., 64 feature maps per layer) with 'main_tea.py' in the traiditional learning way
 2. Pre-train the lightweight vdsr model (e.g., 13 feature maps per layer) with 'main_pre.py' in the traiditional learning way
 3. Train the lightweight vdsr model (e.g., 13 feature maps per layer) with 'main_ail.py' in the proposed AIL learning way
### Performance
  - Following [VDSR](https://cv.snu.ac.kr/research/VDSR/), we train the teacher model and pretrain the lightweight model on the training pairs from all scales, e.g., 2,3,4
  - According to the eay-to-hard learning strategy, we train a separate model for each scale using the proposed AIL learning scheme.
  - The proposed AIL learning scheme can utilized to train a model from scratch without pre-training learning. With VDSR network, pre-training makes the training process more stable. While for the more complex [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch), pre-traning does not make difference.
  - The proposed AIL performs better with a much more lightweight network baseline.
  - In this implementation, we obtain the lightweight network by slimming the baseline network, i.e., reducing the number of feature maps per layer.
  - No bias is used in this implementation
  
 ### Reference
If you find our work useful in your research or publication, please cite our work:<br>
[1] Lei Zhang, Peng Wang, Chunhua Shen, Lingqiao Liu, Wei Wei, Yanning Zhang, and Anton van den Hengel. "Adaptive importance learning for improving lightweight image super-resolution network." International Journal of Computer Vision 128, 479–499 (2020).</i>[[PDF](https://doi.org/10.1007/s11263-019-01253-6)]
```
@article{zhang2019adaptive,
  title={Adaptive importance learning for improving lightweight image super-resolution network},
  author={Zhang, Lei and Wang, Peng and Shen, Chunhua and Liu, Lingqiao and Wei, Wei and Zhang, Yanning and van den Hengel, Anton},
  journal={International Journal of Computer Vision},
  pages={1--21},
  year={2019},
  publisher={Springer}
}
```

