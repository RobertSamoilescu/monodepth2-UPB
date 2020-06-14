#!/bin/bash
ROOT_DIR=/home/robert/PycharmProjects

# # script to create the dataset
# python3 create_dataset.py \
# 	--src_dir $ROOT_DIR/upb_dataset\
# 	--dst_dir $ROOT_DIR/disertatie/monodepth2/dataset\
# 	--split_dir $ROOT_DIR/disertatie/scenes_split


# script to get some sample results
python3 results.py \
	--model_name monodepth\
	--models_dir $ROOT_DIR/disertatie/monodepth2/models\
	--dataset_dir $ROOT_DIR/disertatie/monodepth2/dataset\
	--results_dir $ROOT_DIR/disertatie/monodepth2/results



