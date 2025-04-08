# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed MegaDepth
# dataset at https://www.cs.cornell.edu/projects/megadepth/
# See datasets_preprocess/preprocess_megadepth.py
# --------------------------------------------------------
import os.path as osp
import numpy as np
from PIL import Image
import numpy as np
import torch
import cv2
import PIL
import re

    
import sys
import os
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

    
class Doppelgangers(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, trainon=['dg'], teston=['dg'], **kwargs):
        self.ROOT = ROOT 
        self.train_metadatas = []
        self.test_metadatas = []
        for ds in trainon:
            if ds == 'dg':
                self.train_metadatas += ['train_pairs_megadepth.npy', 'train_pairs_flip.npy', 'train_pairs_noflip.npy']
            if ds == 'visym':
                self.train_metadatas += ['train_pairs_visym.npy']
                
        for ds in teston:
            if ds == 'dg':
                self.test_metadatas += ['test_pairs.npy']
            if ds == 'visym':
                self.test_metadatas += ['test_pairs_visym.npy']
            if ds == 'mapillary':
                self.test_metadatas += ['test_pairs_msls.npy']
            
       
        super().__init__(*args, **kwargs)
        self.split = split
        self.loaded_data = self._load_data(self.split)


    def _load_data(self, split):
        meta_to_image = {
            'train_pairs_megadepth.npy': 'train_megadepth', 
            'train_pairs_flip.npy': 'train_set_flip', 
            'train_pairs_noflip.npy': 'train_set_noflip', 
            'test_pairs.npy': 'test_set',
            'train_pairs_visym.npy': '',
        }
        if split == 'train':
            self.all_pairs = []
            for metadata in self.train_metadatas:
                pairs = np.load(osp.join(self.ROOT, 'pairs_metadata', metadata), allow_pickle=True)
                for pair in pairs:
                    if 'gif' in pair[0] or 'gif' in pair[1]:
                        continue
                    
                    if 'visym' in metadata:
                        self.all_pairs.append([
                            osp.join(self.ROOT, 'visymscenes', pair[0]),
                            osp.join(self.ROOT, 'visymscenes', pair[1]),
                            int(pair[2])
                        ])
                    else: #if 'doppelgangers' in metadata:
                        self.all_pairs.append([
                            osp.join(self.ROOT, 'doppelgangers', 'images', meta_to_image[metadata], pair[0]), 
                            osp.join(self.ROOT, 'doppelgangers', 'images', meta_to_image[metadata], pair[1]),
                            pair[2]
                        ])  
                    #else:
                    #    pass 

            np.random.shuffle(self.all_pairs)
        else:
            self.all_pairs = []
            for metadata in self.test_metadatas:
                pairs = np.load(osp.join(self.ROOT, 'pairs_metadata', metadata), allow_pickle=True)
                for pair in pairs:
                    if 'gif' in pair[0] or 'gif' in pair[1]:
                        continue

                    if 'visym' in metadata:
                        self.all_pairs.append([
                            osp.join(self.ROOT, 'visymscenes', pair[0]),
                            osp.join(self.ROOT, 'visymscenes', pair[1]),
                            int(pair[2])
                        ])
                    else: #'doppelgangers' in metadata:
                        self.all_pairs.append([
                            osp.join(self.ROOT, 'doppelgangers', 'images', meta_to_image[metadata], pair[0]), 
                            osp.join(self.ROOT, 'doppelgangers', 'images', meta_to_image[metadata], pair[1]),
                            pair[2]
                        ])
                    #else:
                    #    pass
                        
            np.random.shuffle(self.all_pairs)

        

    def __len__(self):
        return len(self.all_pairs)

    def get_stats(self):
        return f'{len(self.all_pairs)} pairs'

    def _get_views(self, pair_idx, resolution, rng):
        pair_info = self.all_pairs[pair_idx]
        im1_path, im2_path, dopp_label = pair_info[0], pair_info[1], pair_info[2]

        views = []
        for im_path in [im1_path, im2_path]:
            try:
                image = imread_cv2(im_path)
                image = PIL.Image.fromarray(image)
                # dummy depthmap
                depthmap = np.random.uniform(1.0, 10.0, size=image.size[::-1]).astype(np.float32)
                
            except Exception as e:
                raise OSError(f'cannot load {im_path}, got exception {e}')

            # dummy intrinsics
            intrinsics = np.array([
                [1000, 0, image.size[0] // 2], 
                [0, 1000, image.size[1] // 2], 
                [0, 0, 1]], dtype=np.float32)

            camera_pose = np.eye(4, dtype=np.float32)
            
            image, depthmap, intrinsics  = self._pad_resize_if_necessary(image, depthmap, intrinsics, resolution, rng, info=im_path)
         
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  
                camera_intrinsics=intrinsics,
                dataset='Doppelgangers',
                instance=im_path,
                label=osp.realpath(im_path).replace(osp.realpath(self.ROOT), ''),
                dopp_label=dopp_label)) 
            
        return views
