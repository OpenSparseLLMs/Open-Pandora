import os
import random
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
import boto3
import shutil

from torch.multiprocessing import Lock
delete_lock = Lock()

class WebVid(Dataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """
    def __init__(self,
                 meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 rand_cond_frame=True,
                 processor=None,
                 **kwargs,
                 ):
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.rand_cond_frame = rand_cond_frame

        self.image_processor = processor['image_processor']
        self.diffusion_image_processor = processor['diffusion_image_processor']
        self.tokenizer = processor['tokenizer']

        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_metadata()

        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution),antialias=True),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution, antialias=True)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def dynamic_resize(self, img):
        '''resize frames'''
        width, height = img.size
        t_width, t_height = 512, 320
        k = min(t_width/width, t_height/height)
        new_width, new_height = int(width*k), int(height*k)
        pad = (t_width-new_width)//2, (t_height-new_height)//2, (t_width-new_width+1)//2, (t_height-new_height+1)//2, 
        trans = transforms.Compose([transforms.Resize((new_height, new_width),antialias=True),
                                    transforms.Pad(pad)])
        return trans(img)
                
    def process_img(self, image):
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values # normalize change axis etc.
        diffusion_pixel_values = self.diffusion_image_processor(self.dynamic_resize(image).convert('RGB')).unsqueeze(1) # [C, 1, H, W]
        diffusion_cond_image = diffusion_pixel_values.unsqueeze(0)[:, :, 0] #[1, C, H, W]
        return {'pixel_values':pixel_values.bfloat16(), 'diffusion_pixel_values':diffusion_pixel_values.bfloat16(), 'diffusion_cond_image':diffusion_cond_image.bfloat16()}
        
    def _load_metadata(self):
        if(os.path.isdir(self.meta_path)):
            dataframes = []
            for filename in reversed(os.listdir(self.meta_path)):
                if filename.endswith('.csv'):
                    # 构建完整的文件路径
                    file_path = os.path.join(self.meta_path, filename)
                    # 读取 CSV 文件并将其添加到数据帧列表中
                    df = pd.read_csv(file_path, encoding='ISO-8859-1')
                    dataframes.append(df)
            metadata = pd.concat(dataframes)
        else:
            metadata = pd.read_csv(self.meta_path, dtype=str)
        print(f'>>> {len(metadata)} data samples loaded.')
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
   
        self.metadata = metadata
        self.metadata.dropna(inplace=True)



    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp


    def __getitem__(self, index):

        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
            
        ## get frames until success
        while True:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            try:
                video_path = self._get_video_path(sample)
                ## video_path should be in the format of "....../WebVid/videos/$page_dir/$videoid.mp4"
                caption = sample['caption']
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {sample}")
                continue

            
            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length-1) + 1

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0

            ## calculate frame indices
            frame_indices = [start_idx + frame_stride*i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue

        ## process data
        batch = self.process_data(caption, frames, frame_stride, fps_ori)

        self.delete_files_in_folder()

        return batch
    
    def process_data(self, caption, frames, frame_stride, fps_ori):
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        # 1. process text
        text = self.tokenizer.bos_token + "<image>" + caption + "[IMG_P]" * 64
        batch = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        batch = {k:v[0] for k,v in batch.items()}
        # 2. process image
        image = frames.asnumpy()[0]
        image = Image.fromarray(image)
        batch.update(self.process_img(image))

        # 4. process video
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'

        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        video_batch = {'video': frames, 'caption': caption, 'fps': fps_clip, 'frame_stride': frame_stride}

        batch.update(video_batch)

        return batch
    
    def __len__(self):
        return len(self.metadata)

def resize_image(image):
    # 将图像转换为PIL图像对象
    pil_image = Image.fromarray(image)

    # 调整图像大小为（224，224）
    resized_image = pil_image.resize((512, 320))

    # 如果原图像有alpha通道，则需要去除alpha通道
    if resized_image.mode == 'RGBA':
        resized_image = resized_image.convert('RGB')

    # 将图像转换为NumPy数组
    resized_image_np = np.array(resized_image)

    return resized_image_np


if __name__== "__main__":
    meta_path = "" ## path to the meta file
    data_dir = "" ## path to the data directory
    save_dir = "" ## path to the save directory
    dataset = WebVid(meta_path,
                 data_dir,
                 subsample=None,
                 video_length=16,
                 resolution=[256,448],
                 frame_stride=4,
                 spatial_transform="resize_center_crop",
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True
                 )
    dataloader = DataLoader(dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=False)

    
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    from utils.save_video import tensor_to_mp4
    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        video = batch['video']
        name = batch['path'][0].split('videos/')[-1].replace('/','_')
        tensor_to_mp4(video, save_dir+'/'+name, fps=8)

