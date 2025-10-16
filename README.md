# Progressive Test Time Energy Adaptation for Medical Image Segmentation

## [[Paper]](https://arxiv.org/abs/2312.00837) [[Project page]](https://voldemort108x.github.io/AdaCS/) [[Poster]](assets/ECCV%202024%20Poster.pdf) [[Slides]](assets/ECCV%202024%20Oral%20Talk.pdf)

## Motivation
<img src="assets/motivation.png" width="350" style="float:left; margin-right:20px;">
<ul style="width: 750px;">
<li>Covariate shifts caused by nuisances such as heteroscedastic noise and inconsistent imaging protocols limit the fidelity of medical image segmentation models.</li>
<li>Without assuming access to a pre-collected target dataset, which is often impractical, test-time adaptation (TTA) offers a practical solution to calibrate models on-the-fly during inference.</li>
<li>Assuming a segmentation model is solely trained on source dataset, our goal is to adapt the model to target data without access to the entire target dataset.</li>
</ul>

## Framework
<img src="assets/main_framework.png" width="1000">

## Installation
```bash
conda env create -f environment.yml
conda activate pttea_env
```

## Default directory structure
    ├── Dataset                   
    |   ├── ACDC       # Place the downloaded dataset here
    |   |   ├── train
    |   |   ├── val
    |   |   ├── test
    |   ├── LVQuant
    |   |   ├── train
    |   |   ├── ...
    ├── Code
    |   ├── pttea_seg
    |   |   ├── train_energy.py
    |   |   ├── run_pttea.py
    |   |   ├── ...


## Train
```bash
python train_energy.py --dataset YOUR_DATASET_NAME --batch-size BATCH_SIZE --model-dir MODEL_SAVE_PATH  --image-loss 'bce' --epochs 150 --load-seg-model YOUR_PATH_TO_SEG_MODEL --transform_type 'adv' --numclasses 3 --augmentation 'spatial_aug' --wandb-name YOUR_EXP_NAME --model_seg_type 'unet' 
```

## Test
```bash
!python run_pttea.py --dataset YOUR_DATASET_NAME --model_seg_dir YOUR_PATH_TO_SEG_MODEL --model_energy_dir YOUR_PATH_TO_ENERGY_MODEL --numclasses 3 --batch_size BATCH_SIZE --num_iterations 10 --result_dir RESULTS_SAVE_PATH --model_type 'unet'
```

## Demo
Please check out [demo.ipynb](./demo.ipynb) for progressive update visualization.

## Citation
```
@article{zhang2025progressive,
        title={Progressive Test Time Energy Adaptation for Medical Image Segmentation},
        author={Zhang, Xiaoran and Hong, Byung-Woo and Park, Hyoungseob and Pak, Daniel H and Rickmann, Anne-Marie and Staib, Lawrence H and Duncan, James S and Wong, Alex},
        journal={arXiv preprint arXiv:2503.16616},
        year={2025}
      }
```