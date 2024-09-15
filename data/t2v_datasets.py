import os
import re
import json
import torch
import decord
import torchvision
import requests
import numpy as np
import boto3
from einops import rearrange
from typing import Dict, List, Tuple
import random
import traceback
import cv2

CACHE_DIR = "/mnt/petrelfs/tianjie/projects/Datasets/video_cache"

class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        
    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str

class Vimeo(torch.utils.data.Dataset):
    """Load the Vimeo video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        import pandas as pd
        df = pd.read_csv(self.data_path)
        df = pd.read_csv(configs.data_path)
        shuffled_df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        self.video_lists = shuffled_df
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        # !!private data
        access_key = "H5WUE6Q2E230X71GWSK1"
        secret_key = "8CcxH91RlLHXFzxfxNRxHJhWUkAKV0xTwJJBwdzc"
        endpoint_url='http://10.135.7.249:80'
        # !!private date
        self.s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
        )
    def __getitem__(self, index):
        try:
            index = index % len(self.video_lists)
            s3_path = self.video_lists['url'][index]
            s3_bucket, s3_key = 'vimeo_seg_new', s3_path
            # download video
            local_cache_dir = CACHE_DIR
            if not os.path.exists(local_cache_dir):
                os.mkdir(local_cache_dir)
            local_video_path = os.path.join(local_cache_dir, os.path.basename(s3_key))
            if not os.path.exists(local_video_path):
                self.s3_client.download_file(s3_bucket, s3_key, local_video_path)
            prompt = self.video_lists['caption'][index]
            vframes, _, info = torchvision.io.read_video(filename=local_video_path, pts_unit='sec', end_pts=60*2, output_format='TCHW')
            total_frames = len(vframes)
            os.remove(local_video_path)
            assert total_frames > self.target_video_len , f'Vimeo: total frames is only {total_frames}, less than requried {self.target_video_len}'
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
            # print(f'{s3_path=}, {frame_indice=}')
            video = vframes[frame_indice] #
            video = self.transform(video) # T C H W
            return {'video': video, 'prompt': str(prompt)}
        except Exception as e:
            print(f'read exception data with error {e}')
            # print(traceback.format_exc())
            return None

    def __len__(self):
        return len(self.video_lists)


class Vimeo2(torch.utils.data.Dataset):
    """Load the Vimeo video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = '/mnt/petrelfs/share_data/cgj/vimeo2M.csv'
        import pandas as pd
        df = pd.read_csv(self.data_path)
        self.video_lists = df
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        # !!private data
        access_key = "MB5P67LAWHT86X04ZYOX"
        secret_key = "5IgAUltFfZbQDEWfF92dk6EZelBI7Tgv2C7kU6Xi"
        endpoint_url='http://p-ceph-norm-inside.pjlab.org.cn'
        # !!private date
        self.s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
        )
    def __getitem__(self, index):
        try:
            index = index % len(self.video_lists)
            s3_path = self.video_lists['id'][index].replace('s3://', '')
            #vimeo_seg/100052624/100052624-00;03;21.827-00;03;35.048.mp4
            s3_key = 'vimeo/videos/'+s3_path.split('/')[1]+'.mp4'
            s3_bucket = 'llm-crawl-video'
            # download video
            local_cache_dir = CACHE_DIR
            if not os.path.exists(local_cache_dir):
                os.mkdir(local_cache_dir)
            local_video_path = os.path.join(local_cache_dir, os.path.basename(s3_key))
            vimeo_seg_cache_path = os.path.join(CACHE_DIR, 'vimeo_seg', os.path.basename(s3_key))
            prompt = self.video_lists['caption'][index]
            if not os.path.exists(vimeo_seg_cache_path):
                self.s3_client.download_file(s3_bucket, s3_key, local_video_path)
                ### seg this video
                range_str = s3_path.split('/')[-1]
                # 100052624-00;03;21.827-00;03;35.048.mp4
                _range = range_str.replace('.mp4','').split('-')[1:3]
                start_time_ls = [ float(t) for t in _range[0].split(';')]
                end_time_ls = [float(t) for t in _range[1].split(';')]
                start_time, end_time = 0, 0
                for st, et in zip(start_time_ls, end_time_ls):
                    start_time *= 60
                    start_time += st
                    end_time *= 60
                    end_time += et
                seg_video, _, info = torchvision.io.read_video(filename=local_video_path, pts_unit='sec',start_pts=start_time, end_pts=end_time, output_format='TCHW')
                os.remove(local_video_path)
            else:
                seg_video, _, info = torchvision.io.read_video(filename=vimeo_seg_cache_path, pts_unit='sec', output_format='TCHW')
            
            total_frames = len(seg_video)-5
                
            assert total_frames > self.target_video_len * 4 , f'Vimeo: total frames is only {total_frames}, less than requried {self.target_video_len}*4'
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
            # print(f'{s3_path=}, {frame_indice=}')
            video = seg_video[frame_indice] #
            video = self.transform(video) # T C H W
            
            
            
            if not os.path.exists(vimeo_seg_cache_path):
                seg_video = seg_video.permute( 0, 2, 3, 1)
                torchvision.io.write_video(vimeo_seg_cache_path, seg_video, info['video_fps'])
            return {'video': video, 'prompt': str(prompt)}
        except Exception as e:
            print(f'read exception data with error {e}')
            print(traceback.format_exc())
            return None

    
    def __len__(self):
        return len(self.video_lists)

class Vimeo3(torch.utils.data.Dataset):
    """Load the Vimeo video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = '/mnt/petrelfs/share_data/cgj/vimeo2M.csv'
        import pandas as pd
        df = pd.read_csv(self.data_path)
        self.video_lists = df
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        # !!private data
        access_key = "MB5P67LAWHT86X04ZYOX"
        secret_key = "5IgAUltFfZbQDEWfF92dk6EZelBI7Tgv2C7kU6Xi"
        endpoint_url='http://p-ceph-norm-inside.pjlab.org.cn'
        # !!private date
        self.s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
        )
    def __getitem__(self, index):
        try:
            index = index % len(self.video_lists)
            s3_path = self.video_lists['id'][index].replace('s3://', '')
            #vimeo_seg/100052624/100052624-00;03;21.827-00;03;35.048.mp4
            s3_key = 'vimeo/videos/'+s3_path.split('/')[1]+'.mp4'
            s3_bucket = 'llm-crawl-video'
            # download video
            local_cache_dir = CACHE_DIR
            if not os.path.exists(local_cache_dir):
                os.mkdir(local_cache_dir)
            local_video_path = os.path.join(local_cache_dir, os.path.basename(s3_key))
            prompt = self.video_lists['caption'][index]
            
                
            self.s3_client.download_file(s3_bucket, s3_key, local_video_path)
            ### seg this video
            range_str = s3_path.split('/')[-1]
            # 100052624-00;03;21.827-00;03;35.048.mp4
            _range = range_str.replace('.mp4','').split('-')[1:3]
            start_time_ls = [ float(t) for t in _range[0].split(';')]
            end_time_ls = [float(t) for t in _range[1].split(';')]
            start_time, end_time = 0, 0
            for st, et in zip(start_time_ls, end_time_ls):
                start_time *= 60
                start_time += st
                end_time *= 60
                end_time += et
            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                raise Exception("Error: Could not open video.")
            fps = cap.get(cv2.CAP_PROP_FPS)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)-1
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            while cap.isOpened():
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if current_frame > end_frame:
                    break
                if len(frames) > 200:
                    break
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if not ret:
                    break
            frames = np.array(frames)
            frames = torch.tensor(frames)
            seg_video = frames.permute(0,3,1,2)
            os.remove(local_video_path)
            total_frames = len(seg_video)
            assert total_frames > self.target_video_len * 4 , f'Vimeo: total frames is only {total_frames}, less than requried {self.target_video_len}*4'
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
            # print(f'{s3_path=}, {frame_indice=}')
            video = seg_video[frame_indice] #
            video = self.transform(video) # T C H W
            return {'video': video, 'prompt': str(prompt)}
        except Exception as e:
            print(f'read exception data with error {e}')
            print(traceback.format_exc())
            return None

    
    def __len__(self):
        return len(self.video_lists)


class WebVid(torch.utils.data.Dataset):
    """Load the Webvid video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = "/mnt/petrelfs/share_data/cgj/webvid2M.csv"
        import pandas as pd
        df = pd.read_csv(self.data_path)
        self.video_lists = df
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        # !!private data
        access_key = "H5WUE6Q2E230X71GWSK1"
        secret_key = "8CcxH91RlLHXFzxfxNRxHJhWUkAKV0xTwJJBwdzc"
        endpoint_url='http://10.135.7.249:80'
        # !!private date
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
            )
    
    def __getitem__(self, index):
        try:
            index = index % len(self.video_lists)
            s3_path = self.video_lists['id'][index]
            s3_bucket, s3_key = 'WebVid10M', s3_path.replace('s3://WebVid10M/','')
            # download video
            local_cache_dir = CACHE_DIR
            if not os.path.exists(local_cache_dir):
                os.mkdir(local_cache_dir)
            local_video_path = os.path.join(local_cache_dir, os.path.basename(s3_key))
            if not os.path.exists(local_video_path):
                self.s3_client.download_file(s3_bucket, s3_key, local_video_path)
            prompt = self.video_lists['caption'][index]
            vframes, _, info = torchvision.io.read_video(filename=local_video_path, pts_unit='sec', end_pts=2*60, output_format='TCHW', )
            vframes = vframes[:10000,:,:]
            total_frames = len(vframes)
            os.remove(local_video_path)
            assert total_frames > self.target_video_len , f'Webvid: total frames is only {total_frames}, less than requried {self.target_video_len}*4'
            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
            # print(f'{s3_path=}, {prompt=}')
            video = vframes[frame_indice] #
            video = self.transform(video) # T C H W
            return {'video': video, 'prompt': str(prompt)}
        except Exception as e:
            print(f'read exception data with error {e}')
            # print(traceback.format_exc())
            return None

    def __len__(self):
        return len(self.video_lists)



class T2V_Videos(torch.utils.data.Dataset):
    """Load the UCF101 video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        # !!private data
        access_key = "H5WUE6Q2E230X71GWSK1"
        secret_key = "8CcxH91RlLHXFzxfxNRxHJhWUkAKV0xTwJJBwdzc"
        endpoint_url='http://10.135.7.249:80'
        # !!private date
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
            )
        self.webvid = WebVid(configs=configs, transform=transform, temporal_sample=temporal_sample)
        self.vimeo = Vimeo3(configs=configs, transform=transform, temporal_sample=temporal_sample)
    
    def __getitem__(self, index):
        try:
            if index % 2 == 0:
                return self.webvid[index//2]
            else:
                return self.vimeo[index//2]
            # return self.vimeo[index]
        except Exception as e:
            print(f'read exception data with error {e}')
            return None
        
    def __len__(self):
        return min(len(self.vimeo), len(self.webvid))


if __name__ == '__main__':

    import argparse
    import torch.utils.data as Data
    import torchvision.transforms as transforms
    from torchvision.transforms import functional as F

    class TemporalRandomCrop:
        def __init__(self, size):
            self.size = size

        def __call__(self, clip_length):
            if not isinstance(clip_length, int):
                raise TypeError("Expected clip_length to be an integer, but got {}".format(type(clip_length)))
            
            if clip_length <= self.size:
                start_frame_ind = 0
            else:
                start_frame_ind = torch.randint(0, clip_length - self.size + 1, (1,)).item()
            
            end_frame_ind = start_frame_ind + self.size
            return start_frame_ind, end_frame_ind

    class ToTensorVideo:
        def __call__(self, clip):
            if isinstance(clip, torch.Tensor):
                return clip  # If clip is already a tensor, return it as is
            else:
                return torch.stack([F.to_tensor(img) for img in clip])  # Otherwise, con

    class RandomHorizontalFlipVideo:
        def __init__(self, p=0.5):
            self.p = p

        def __call__(self, clip):
            if torch.rand(1).item() < self.p:
                return torch.stack([F.hflip(img) for img in clip])
            return clip

    class UCFCenterCropVideo:
        def __init__(self, size):
            self.size = size

        def __call__(self, clip):
            return torch.stack([F.center_crop(img, self.size) for img in clip])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    # parser.add_argument("--data-path", type=str, default="/nvme/share_data/datasets/UCF101/videos")
    parser.add_argument("--data-path", type=str, default="/path/to/datasets/UCF101/videos/")
    config = parser.parse_args()


    temporal_sample = TemporalRandomCrop(config.num_frames * config.frame_interval)

    transform_ucf101 = transforms.Compose([
        ToTensorVideo(), # TCHW
        RandomHorizontalFlipVideo(),
        UCFCenterCropVideo(256),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])


    ffs_dataset = T2V_Videos(config, transform=transform_ucf101, temporal_sample=temporal_sample)
    ffs_dataloader = Data.DataLoader(dataset=ffs_dataset, batch_size=6, shuffle=False, num_workers=16)

    # for i, video_data in enumerate(ffs_dataloader):
    for video_data in ffs_dataloader:
        print(type(video_data))
        video = video_data['video']
        print(video.shape)
        exit()