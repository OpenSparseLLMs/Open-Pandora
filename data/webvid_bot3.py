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
import cv2
import torchvision
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
                 subsample=None,
                 video_length=16,
                 resolution=[320, 512],
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
        self.max_prompt_len = 256
        self.img_p_token = '[IMG_P]'
        self.image_processor = processor['image_processor']
        self.diffusion_image_processor = processor['diffusion_image_processor']
        self.tokenizer = processor['tokenizer']

        self.local_cache_dir = '/mnt/petrelfs/tianjie/projects/Datasets/video_cache/'
        access_key = 'VVEGYBP0A4FFFPZDIUIC'
        secret_key = 'XO3aoVDg4z2sJ8i8TDgDfnwReCJfdawLBgWPzwld'
        # vimeo
        endpoint_url='http://10.135.7.249:80'
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_metadata()
        os.makedirs(self.local_cache_dir,exist_ok=True)
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
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



    def _get_video_path(self, sample, s3_bucket='WebVid10M'):
        s3_path = sample['id']
        mp4_file_key = s3_path.replace(f"s3://{s3_bucket}/", "")

        local_video_path = os.path.join(self.local_cache_dir, os.path.basename(mp4_file_key))
        self.s3_client.download_file(s3_bucket, mp4_file_key, local_video_path)

        return local_video_path
    
    def delete_files_in_folder(self):
        """
        删除指定文件夹下的所有文件

        参数：
        folder_path：要删除文件的文件夹路径
        """
        # 获取文件夹中的所有文件名
        with delete_lock:
            file_names = os.listdir(self.local_cache_dir)
            if len(file_names)>1e3:
                # 遍历所有文件并删除
                try:
                    shutil.rmtree(self.local_cache_dir)
                except OSError as e:
                    print(f"Error: {e.strerror}")

                try:
                    os.makedirs(self.local_cache_dir)
                except OSError as e:
                    print(f"Error: {e.strerror}")

    def __getitem__(self, index):
        max_retries = 10
        retries = 0
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
            
        ## get frames until success
        while retries < max_retries:
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
                    retries += 1
                    continue
                else:
                    pass
            except:
                index += 1
                retries += 1
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
                    retries += 1 
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
                frames = video_reader.get_batch(frame_indices).asnumpy()
                break
            except:
                print(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                retries += 1 
                continue
            
        assert retries < max_retries, "Max retries reached. Unable to load video."
        ## process data
        batch = self.process_data(caption, frames, frame_stride, fps_ori)

        self.delete_files_in_folder()

        return batch
    
    def process_data(self, caption, frames, frame_stride, fps_ori):
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        # 1. process text
        batch = self.tokenize_and_truncate(caption)
        # 2. process image
        image = frames[0]
        image = Image.fromarray(image)
        batch.update(self.process_img(image))

        # 4. process video
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
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

    def tokenize_and_truncate(self, caption):
        
        text = self.tokenizer.bos_token + "<image>" + caption
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        truncated_tokens = tokens[:self.max_prompt_len - 64]

        img_p_tokens = [self.tokenizer.encode(self.img_p_token, add_special_tokens=False)] * 64

        final_tokens = truncated_tokens + sum(img_p_tokens, [])

        final_tokens_tensor = torch.tensor(final_tokens)
        batch = {"input_ids": final_tokens_tensor, "attention_mask": torch.ones_like(final_tokens_tensor)}

        return batch


class Vimeo(WebVid):
    def __init__(self, **args):
        super(Vimeo, self).__init__(**args)
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
        max_retries = 10
        retries = 0
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride
        while retries < max_retries:
            index = index % len(self.metadata)
            sample = self.metadata.iloc[index]
            try:
                s3_path = sample['id'].replace('s3://', '')
                #vimeo_seg/100052624/100052624-00;03;21.827-00;03;35.048.mp4
                s3_key = 'vimeo/videos/'+s3_path.split('/')[1]+'.mp4'
                s3_bucket = 'llm-crawl-video'
                # download video
                if not os.path.exists(self.local_cache_dir):
                    os.mkdir(self.local_cache_dir)
                local_video_path = os.path.join(self.local_cache_dir, os.path.basename(s3_key))
                prompt = sample['caption']
                
                    
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
                seg_video = np.transpose(frames, (0, 3, 1, 2))
                os.remove(local_video_path)
                total_frames = len(seg_video)

            except:
                index += 1
                retries += 1
                print(f"Load video failed! path = {sample}")
                continue

            fps_ori = fps
            if self.fixed_fps is not None:
                frame_stride = int(frame_stride * (1.0 * fps_ori / self.fixed_fps))

            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)
            
            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length-1) + 1
            frame_num = total_frames
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    retries += 1
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
                frames = np.transpose(seg_video[frame_indices],(0,2,3,1))
                break
            except:
                print(f"Get frames failed! path = {sample}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                retries += 1
                continue

            ## process data
        assert retries < max_retries, "Max retries reached. Unable to load video."
        batch = self.process_data(prompt, frames, frame_stride, fps_ori)

        self.delete_files_in_folder()

        return batch

    def process_data(self, caption, frames, frame_stride, fps_ori):
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        # 1. process text
        batch = self.tokenize_and_truncate(caption)
        # 2. process image
        # image = frames[0]
        # image = Image.fromarray(image)
        batch.update(self.process_img(frames[:5]))
        frames = frames[4:]

        # 4. process video
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float() # [t,h,w,c] -> [c,t,h,w]
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
    
    def process_img(self, images):
        images = [Image.fromarray(img) for img in images]
        image = images[4]
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values # normalize change axis etc.
        cond_image_values = [self.diffusion_image_processor(self.dynamic_resize(img).convert('RGB')) for img in images[:5]]
        if random.random() > 0.4 :
            diffusion_pixel_values = cond_image_values[-1].unsqueeze(1)
        else :
            diffusion_pixel_values = torch.stack(cond_image_values[:4], dim=1)
        diffusion_cond_image = cond_image_values[-1].unsqueeze(0) #[1, C, H, W]
        return {'pixel_values':pixel_values.bfloat16(), 'diffusion_pixel_values':diffusion_pixel_values.bfloat16(), 'diffusion_cond_image':diffusion_cond_image.bfloat16()}
                

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

def tensor_to_mp4(video, savepath, fps, rescale=True, nrow=None):
    """
    video: torch.Tensor, b,c,t,h,w, 0-1
    if -1~1, enable rescale=True
    """
    n = video.shape[0]
    video = video.permute(1, 0, 2, 3, 4) # t,b,c,h,w
    nrow = int(np.sqrt(n)) if nrow is None else nrow
    frame_grids = [torchvision.utils.make_grid(framesheet, nrow=nrow, padding=0) for framesheet in video] # [3, grid_h, grid_w]
    grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [T, 3, grid_h, grid_w]
    # grid = torch.clamp(grid.float(), -1., 1.)
    # if rescale:
    #     grid = (grid + 1.0) / 2.0
    # grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, 3, grid_h, grid_w] -> [T, grid_h, grid_w, 3]
    torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

if __name__== "__main__":
    meta_path = "" ## path to the meta file
    data_dir = "" ## path to the data directory
    save_dir = "." ## path to the save directory
    # dataset = WebVid(meta_path,
    #              data_dir,
    #              subsample=None,
    #              video_length=16,
    #              resolution=[256,448],
    #              frame_stride=4,
    #              spatial_transform="resize_center_crop",
    #              crop_resolution=None,
    #              fps_max=None,
    #              load_raw_resolution=True
    #              )
    dataset = Vimeo(
        meta_path = '/mnt/petrelfs/share_data/cgj/vimeo2M.csv',

    )
    dataloader = DataLoader(dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=True)

    
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    # from utils.save_video import tensor_to_mp4
    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        video = batch['video']
        name = batch['caption']
        print(name)
        tensor_to_mp4(video, save_dir+'/'+str(i)+'.mp4', fps=8)
        if i>4:
            break
