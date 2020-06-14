#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import PIL.Image as pil
import glob
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, help='source directory containing all the videos')
parser.add_argument('--dst_dir', type=str, help='destination directory for the dataset')
parser.add_argument('--split_dir', type=str, help='directory containign the split of the dataset')
parser.add_argument('--camera_idx', action='store_true', help='if the scenes have camera index')
args = parser.parse_args()


def read_video(file: str, src_folder: str, dst_folder: str, verbose: bool = False):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    src_path = os.path.join(src_folder, file)
    dst_path = os.path.join(dst_folder, file[:-4])
    cap = cv2.VideoCapture(src_path)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    # make destination folder
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        
    # frame index    
    frame_idx = 0    
    
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            frame = frame[:320, ...]
            
            # Display the resulting frame
            if verbose:
                cv2.imshow('Frame',frame)
            
            # save frame image
            img_path = os.path.join(dst_path, str(frame_idx) + ".png")
            frame_idx += 1
            img = pil.fromarray(frame[..., ::-1])
            img.save(img_path, 'png')

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()


def optical_flow(one, two):
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(
        one_g, two_g, flow=None,
        pyr_scale=0.5, levels=1, 
        winsize=15, iterations=2,
        poly_n=5, poly_sigma=1.1, flags=0)
    
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)



if __name__ == "__main__":
    videos = os.listdir(args.src_dir)
    videos = [v for v in videos if v.endswith('.mov')][:10]

    for v in tqdm(videos):
        read_video(v, args.src_dir, args.dst_dir, False)

    # get the path for all frames
    frames_path = set(glob.iglob(os.path.join(args.dst_dir, '**','*.png'), recursive=True))
    moving_frames = []

    for frame_path in tqdm(frames_path):
        # get a line
        line = frame_path.split("/")
        
        # split line in folder, scene and frame idx
        folder = "/".join(line[:-2])
        scene = line[-2]
        frame_idx = int(line[-1].split(".")[0])
        
        # check if the next second frame is in the list
        snd_path = os.path.join(folder, scene, str(frame_idx + 2) + ".png")
        if snd_path not in frames_path:
            continue
            
        # get the next frame path
        fst_path = os.path.join(folder, scene, str(frame_idx + 1) + ".png")
        
        # compute optical flow (line, fst_line), (fst_line, snd_line)
        img0 = cv2.imread(frame_path)
        img1 = cv2.imread(fst_path)
        img2 = cv2.imread(snd_path)
        
        of1 = optical_flow(img0, img1)
        if of1 < 1:
            continue
            
        of2 = optical_flow(img1, img2)
        if of2 < 1:
            continue
        
        # append fst_path to moving frames
        moving_frames.append(fst_path)

    formated_frames = []
    for frame_path in moving_frames:
        # get a line
        line = frame_path.split("/")
        
        # split line in folder, scene and frame idx
        folder = "/".join(line[:-2])
        scene = line[-2]
        frame_idx = line[-1].split(".")[0]
        
        # add extra f for compatibility with dataloader
        new_line = " ".join([scene, frame_idx, "f"])
        formated_frames.append(new_line)
        
    formated_frames = sorted(formated_frames)


    # split in train and test
    with open(os.path.join(args.split_dir, "train_scenes.txt"), "rt") as fin:
        train_scenes = fin.read()
        train_scenes = train_scenes.split('\n')
        train_scenes = set(train_scenes)

    with open(os.path.join(args.split_dir, "test_scenes.txt"), "rt") as fin:
        test_scenes = fin.read()
        test_scenes = test_scenes.split("\n")
        test_scenes = set(test_scenes)

    train_files = []
    test_files = []

    for ff in formated_frames:
        scene, _, _ = ff.split(" ")
        if args.camera_idx:
            scene = scene[:-2]
        
        if scene in train_scenes:
            train_files.append(ff)
            
        if scene in test_scenes:
            test_files.append(ff)
        
    train_files = "\n".join(train_files)
    test_files = "\n".join(test_files)

    if not os.path.exists('splits'):
        os.makedirs('splits')

    if not os.path.exists('splits/upb'):
        os.makedirs('splits/upb')

    with open(os.path.join('splits', 'upb', 'train_files.txt'), 'wt') as fout:
        fout.write(train_files)

    with open(os.path.join('splits', 'upb', 'test_files.txt'), 'wt') as fout:
        fout.write(test_files)
