# Monodepth2 - UPB

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **Digging into Self-Supervised Monocular Depth Prediction**
>
> [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/), [Oisin Mac Aodha](http://vision.caltech.edu/~macaodha/), [Michael Firman](http://www.michaelfirman.co.uk) and [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/)  
>
> [ICCV 2019](https://arxiv.org/abs/1806.01260)

<p align="center">
  <img src="sample/sample.gif" alt="example input output gif" width="1024" />
</p>

```
@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
```


## Create dataset

```shell
mkdir raw_dataset
```

* Download the UBP dataset into the "raw_dataset" directory. A sample of the UPB dataset is available <a href="https://drive.google.com/drive/folders/1p_2-_Xo-Wd9MCnkYqPfGyKs2BnbeApqn?usp=sharing">here</a>. Those video are 3FPS. Consider downloding the original dataset and downsample to 10FPS.

```shell
mkdir scene_splits
```

* Download the scene splits into the "scene_splits" directory. The train-validation split is available <a href="https://github.com/RobertSamoilescu/UPB-Dataset-Split">here</a>.
In the "scene_splits" directory you should have: "train_scenes.txt" and "test_scenes.txt".


```shell
# script to create the dataset
python3 scripts/create_dataset.py \
  --src_dir raw_dataset \
  --dst_dir ./dataset \
  --split_dir scene_splits
```

## Train model - example

* Downloading the pretrained model to fine-tune

```shell
# script to download pretrained model
python3 download.py
````

* Fine-tune existing model

```shell
# script to train the model
python3 train.py \
  --model_name finetuned_mono \
  --load_weights_folder ./models/mono_640x192 \
  --data_path ./dataset\
  --log_dir ./logs \
  --height 256 \
  --width 512 \
  --num_workers 4 \
  --split upb \
  --dataset upb \
  --learning_rate 1e-6 \
  --batch_size 12 \
  --num_epochs 5 \
  --disparity_smoothness 1e-3\
```
Conisder playing with "disparity_smoothness".

## Test model
* Copy trained model
```shell
cp -r logs/finetuned_mono/models/weights_4 models/monodepth
```

* Get samples
```shell
# script to get some sample results
python3 scripts/results.py \
  --model_name monodepth\
  --models_dir ./models\
  --split_dir ./splits/upb\
  --dataset_dir ./dataset\
  --results_dir ./results

```

## Pre-trained model
A pre-trained model (512x256 - 10FPS) is available <a href='https://drive.google.com/drive/folders/18kTR4PaRlQIeEFJ2gNkiXYnFcTfyrRNH?usp=sharing'>here</a>.
