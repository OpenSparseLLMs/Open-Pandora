
import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer
import numpy as np
from PIL import Image
import torch
import torchvision
import gradio as gr
from argparse import ArgumentParser
import os, re
import uuid
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    ''''input parameters'''
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=False,
        default='../ckpt'
    )
    parser.add_argument(
        "--debug",
        action='store_true'
    )
    args = parser.parse_args()
    return args

def dynamic_resize(img):
    '''resize frames'''
    width, height = img.size
    t_width, t_height = 512, 320
    k = min(t_width/width, t_height/height)
    new_width, new_height = int(width*k), int(height*k)
    pad = (t_width-new_width)//2, (t_height-new_height)//2, (t_width-new_width+1)//2, (t_height-new_height+1)//2, 
    trans = transforms.Compose([transforms.Resize((new_height, new_width)),
                                transforms.Pad(pad)])
    return trans(img)

class ChatWM:
    def __init__(self, model, processor):
        self.model = model
        self.image_processor = processor['image_processor']
        self.diffusion_image_processor = processor['diffusion_image_processor']
        self.tokenizer = processor['tokenizer']
        self.generate_kwargs =  {
            "unconditional_guidance_scale": 4,
            "ddim_steps": 50,
            "ddim_eta": 1.0,
            "fs": 15,
            "timestep_spacing": "uniform_trailing",
            "n_samples": 4,
        }
        self.cat_videos = []
        self.text = ''
        self.pixel_values = None
        self.diffusion_cond_image = None
        self.current_round = 0
        self.video_path = [f'./video_output/video_output_gradio_round{i}_{uuid.uuid4()}.mp4' for i in range(10)]
        self.text_list = []

        

    def generate_video(self, image, text_input, ddim_steps, fs, n_samples,
                       unconditional_guidance_scale, ddim_eta,
                       progress=gr.Progress()):
        self.generate_kwargs['ddim_steps'] = ddim_steps
        self.generate_kwargs['fs'] = fs
        self.generate_kwargs['n_samples'] = n_samples
        self.generate_kwargs['unconditional_guidance_scale'] =unconditional_guidance_scale
        self.generate_kwargs['ddim_eta'] = ddim_eta 
        self.generate_kwargs['gr_progress_bar'] = progress
        self.generate_kwargs['round_info'] = [1,1]
        self.current_round = 1
        if self.model == None: # debug mode
            return self.video_path[0]
         
        self.text = self.tokenizer.bos_token + "<image>" + text_input + "[IMG_P]" * 64

        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        batch = self.tokenizer(self.text, return_tensors="pt", add_special_tokens=False)
        batch.update(self.process_img(image))
        batch = {k: v.to(torch_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        videos = self.model.generate(**batch,
                            tokenizer=self.tokenizer,
                            **self.generate_kwargs)
        self.cat_videos = [videos]
        self.text_list = [self.text]
        
        self.pixel_values = batch['pixel_values']
        self.diffusion_cond_image = batch['diffusion_cond_image']
        self.process_generated_video(videos, fps=8, video_path=self.video_path[1])

        return self.video_path[1], self.video_path[1], gr.update(interactive=True, value='🔄 Re-do Action 1'), gr.update(interactive=True),  gr.update(interactive=False) 

    def generate_video_next_round(self, text_input, ddim_steps, fs, n_samples,
                       unconditional_guidance_scale, ddim_eta,
                       progress=gr.Progress()):
        self.generate_kwargs['ddim_steps'] = ddim_steps
        self.generate_kwargs['fs'] = fs
        self.generate_kwargs['n_samples'] = n_samples
        self.generate_kwargs['unconditional_guidance_scale'] =unconditional_guidance_scale
        self.generate_kwargs['ddim_eta'] = ddim_eta 
        self.generate_kwargs['gr_progress_bar'] = progress
        self.generate_kwargs['round_info'] = [1,1]
        
        if self.model == None: # debug mode
            return self.video_path[0]

        self.cat_videos = self.cat_videos[:self.current_round -1]
        self.text_list = self.text_list[:self.current_round -1]
        self.text = ''.join(self.text_list) +  "<image>" * 16 + text_input + "[IMG_P]" * 64

        batch = self.tokenizer(self.text, return_tensors="pt", add_special_tokens=False)
        batch.update(self.process_img_from_output(self.cat_videos[-1], self.pixel_values))
        batch['diffusion_cond_image'] = self.diffusion_cond_image
        batch = {k: v.to(torch_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        videos = self.model.generate(**batch,
                            tokenizer=self.tokenizer,
                            **self.generate_kwargs)
        self.cat_videos.append(videos)
        self.pixel_values = batch['pixel_values']
        self.process_generated_video(videos, fps=8, video_path=self.video_path[self.current_round])
        self.process_generated_video_multi(self.cat_videos,fps=8, video_path=self.video_path[0],num_round=len(self.cat_videos))
        return self.video_path[0], self.video_path[self.current_round], gr.update(interactive=True, value=f'🔄 Re-do Action {self.current_round}'), gr.update(interactive=True) # ,  self.video_path[0]

    def generate_video_next_round2(self,text_input, ddim_steps, fs, n_samples,
                       unconditional_guidance_scale, ddim_eta,
                       progress=gr.Progress()):
        self.current_round = 2
        return self.generate_video_next_round(text_input, ddim_steps, fs, n_samples, unconditional_guidance_scale, ddim_eta, progress)

    def generate_video_next_round3(self,text_input, ddim_steps, fs, n_samples,
                       unconditional_guidance_scale, ddim_eta,
                       progress=gr.Progress()):
        self.current_round = 3
        return self.generate_video_next_round(text_input, ddim_steps, fs, n_samples, unconditional_guidance_scale, ddim_eta, progress)

    def generate_video_next_round4(self,text_input, ddim_steps, fs, n_samples,
                       unconditional_guidance_scale, ddim_eta,
                       progress=gr.Progress()):
        self.current_round = 4
        return self.generate_video_next_round(text_input, ddim_steps, fs, n_samples, unconditional_guidance_scale, ddim_eta, progress)

    def generate_video_next_round5(self,text_input, ddim_steps, fs, n_samples,
                       unconditional_guidance_scale, ddim_eta,
                       progress=gr.Progress()):
        self.current_round = 5
        return self.generate_video_next_round(text_input, ddim_steps, fs, n_samples, unconditional_guidance_scale, ddim_eta, progress)

    def generate_video_mutliround(self, image, text_input, ddim_steps, fs, n_samples,
                       unconditional_guidance_scale, ddim_eta,num_round=2, video_path=f'./video_output/video_output_gradio_multiturn_{uuid.uuid4()}.mp4',
                       progress=gr.Progress()):
        self.generate_kwargs['ddim_steps'] = ddim_steps
        self.generate_kwargs['fs'] = fs
        self.generate_kwargs['n_samples'] = n_samples
        self.generate_kwargs['unconditional_guidance_scale'] =unconditional_guidance_scale
        self.generate_kwargs['ddim_eta'] = ddim_eta 
        self.generate_kwargs['gr_progress_bar'] = progress
        self.generate_kwargs['round_info'] = [1,num_round]
        if self.model == None: # debug mode
            return video_path
         
        text = self.tokenizer.bos_token + "<image> " + text_input + "[IMG_P]" * 64
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        batch = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        batch.update(self.process_img(image))
        batch = {k: v.to(torch_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        videos = self.model.generate(**batch,
                            tokenizer=self.tokenizer,
                            **self.generate_kwargs)

        cat_videos = [videos]
        for _ in range(1, num_round):
            self.generate_kwargs['round_info'][0] += 1
            text += "<image>" * 16 + text_input + "[IMG_P]" * 64
            batch.update(self.tokenizer(text, return_tensors="pt", add_special_tokens=False))
            batch.update(self.process_img_from_output(videos, batch['pixel_values']))
            batch = {k: v.to(torch_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            videos = self.model.generate(**batch,
                        tokenizer=self.tokenizer,
                        **self.generate_kwargs)
            cat_videos.append(videos)
        self.process_generated_video_multi(cat_videos,fps=8, video_path=video_path,num_round=num_round)
        return video_path, gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False),gr.update(interactive=False)

    def generate_video_mutliround_separate(self, image, text_input, ddim_steps, fs, n_samples,
                       unconditional_guidance_scale, ddim_eta,num_round=2,
                       progress=gr.Progress()):
        self.generate_kwargs['ddim_steps'] = ddim_steps
        self.generate_kwargs['fs'] = fs
        self.generate_kwargs['n_samples'] = n_samples
        self.generate_kwargs['unconditional_guidance_scale'] =unconditional_guidance_scale
        self.generate_kwargs['ddim_eta'] = ddim_eta 
        self.generate_kwargs['gr_progress_bar'] = progress
        self.generate_kwargs['round_info'] = [1,num_round]
        # video_path='./video_output/video_output_gradio.mp4',
        video_path_list = [f'./video_output/video_output_gradio_{i}.mp4' for i in range(num_round+1)]
        if self.model == None: # debug mode
            return video_path_list
         
        text = self.tokenizer.bos_token + "<image> " + text_input + "[IMG_P]" * 64
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        batch = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        batch.update(self.process_img(image))
        batch = {k: v.to(torch_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        videos = self.model.generate(**batch,
                            tokenizer=self.tokenizer,
                            **self.generate_kwargs)
        self.process_generated_video(videos, fps=8, video_path=video_path_list[1])
        cat_videos = [videos]
        for j in range(1, num_round):
            self.generate_kwargs['round_info'][0] += 1
            text += "<image>" * 16 + text_input + "[IMG_P]" * 64
            batch.update(self.tokenizer(text, return_tensors="pt", add_special_tokens=False))
            batch.update(self.process_img_from_output(videos, batch['pixel_values']))
            batch = {k: v.to(torch_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            videos = self.model.generate(**batch,
                        tokenizer=self.tokenizer,
                        **self.generate_kwargs)
            self.process_generated_video(videos, fps=8, video_path=video_path_list[j])
            cat_videos.append(videos)
        self.process_generated_video_multi(cat_videos,fps=8, video_path=video_path_list[0],num_round=num_round)
        return video_path_list

    
    def process_img(self, image):
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.to(torch_device)
        diffusion_pixel_values = self.diffusion_image_processor(dynamic_resize(image).convert('RGB')).unsqueeze(1)
        diffusion_cond_image = diffusion_pixel_values.unsqueeze(0)[:, :, 0]
        return {'pixel_values':pixel_values.bfloat16(), 'diffusion_pixel_values':diffusion_pixel_values.bfloat16(), 'diffusion_cond_image':diffusion_cond_image.bfloat16()}
    
    def process_img_from_output(self, videos, pixel_values):
        new_images = videos.squeeze(0)[0].detach().permute((1, 0, 2, 3)).clamp(-1., 1.).to(torch.float32)
        new_images = (new_images + 1.) / 2.
        new_pil_images = [transforms.functional.to_pil_image(new_image, mode='RGB') for new_image in new_images]
        new_pixel_values = self.image_processor(images=new_pil_images, return_tensors="pt").pixel_values.to(torch_device)
        pixel_values = torch.cat((pixel_values, new_pixel_values), dim=0)
        diffusion_pixel_values = [self.diffusion_image_processor(dynamic_resize(new_image).convert('RGB')) for new_image in new_pil_images[-4:]]
        diffusion_pixel_values = torch.stack(diffusion_pixel_values, dim=1)
        return {'pixel_values':pixel_values.bfloat16(), 'diffusion_pixel_values':diffusion_pixel_values.bfloat16()}

            
            
    def process_generated_video(self, videos, fps=8, video_path='video_output.mp4'):
        video = videos.squeeze(0).detach().cpu().to(torch.float32).clamp(-1., 1.)
        video = video.permute(2, 0, 1, 3, 4)
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=2, padding=0) for framesheet in video]
        grid = torch.stack(frame_grids, dim=0)
        grid = ((grid + 1.) / 2. * 255.).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(video_path, grid, fps=fps, video_codec='h264', options={'crf': '10'})
        
    def process_generated_video_multi(self,cat_videos, fps=8, video_path='video_output.mp4',num_round=2):
        video_list = [list(range(0,12))]
        for i in range(1,num_round):
            if i == num_round - 1:
                video_list.append(list(range(i*16, (i+1)*16)))
            else:
                video_list.append(list(range(i*16,(i+1)*16-4)))
        video = torch.cat(cat_videos, dim=3).squeeze(0).squeeze(0).detach().cpu().clamp(-1., 1.)
        video = ((video + 1.) / 2. * 255.).permute((1, 2, 3, 0))
        
        video = torch.cat( [video[video_l] for video_l in video_list], dim=0)
        # video = torch.cat((video[0:12], video[16:32]), dim=0)
        torchvision.io.write_video(video_path, video, fps=fps, video_codec='h264', options={'crf': '10'})
        





def load_wm(repo_id,model=None):
    '''load model, image processor and tokenizer'''

    from model_demo import WorldModel, WorldModelConfig
    ckpt_name = repo_id.split('/')[-1]
    print(f"Start to load model, current ckpt is: {ckpt_name}")
    config = WorldModelConfig.from_pretrained(repo_id)
    # config.reset_training_args(do_alignment=False,
    #                        dynamicrafter='./DynamiCrafter/configs/inference_512_v1.0.yaml',
    #                        )
    if model == None:
        model = WorldModel.from_pretrained(repo_id, config=config, ignore_mismatched_sizes=True)
        # model = model.to(device=torch_device, dtype=torch.bfloat16).eval()
    # model loaded
    
    # load image processors
    image_processor = model.video_model.get_vision_tower().image_processor
    diffusion_image_processor= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    # load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.add_tokens(["<img_s>"], special_tokens=True)
    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["[IMG_P]"], special_tokens=True)
    tokenizer.image_start_token_id = tokenizer.convert_tokens_to_ids("<img_s>")
    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    tokenizer.image_prefix_token_id = tokenizer.convert_tokens_to_ids("[IMG_P]")
    processor = {
        'image_processor':image_processor,
        'diffusion_image_processor':diffusion_image_processor,
        'tokenizer':tokenizer
    }
    return model, processor



def init_sliders():
    fs = gr.Slider(
        minimum=1,
        maximum=30,
        value=15,
        step=1,
        interactive=True,
        label="FPS",
    )
    n_samples = gr.Slider(
        minimum=1,
        maximum=9,
        value=1,
        step=1,
        interactive=True,
        label="Number of generated samples",
    )
    unconditional_guidance_scale = gr.Slider(
        minimum=1,
        maximum=20,
        value=4,
        step=0.5,
        interactive=True,
        label="Unconditional guidance scale",
    )
    ddim_steps = gr.Slider(
        minimum=10,
        maximum=200,
        value=50,
        step=10,
        interactive=True,
        label="DDIM steps",
    )     
    ddim_eta = gr.Slider(
        minimum=0.0,
        maximum=5.0,
        value=1.0,
        step=0.2,
        interactive=True,
        label="DDIM eta",
    )
    num_round = gr.Slider(
        minimum=1,
        maximum=5,
        value=2,
        step=1,
        interactive=True,
        label="round",
    )
    return fs, n_samples, unconditional_guidance_scale, ddim_steps, ddim_eta, num_round

def gradio_reset():
    return (
        gr.update(interactive=True, value='💭 Action 1'), #button
        gr.update(interactive=False, value='💭 Action 2'),
        gr.update(interactive=False,value='💭 Action 3'),
        gr.update(interactive=False,value='💭 Action 4'),
        gr.update(interactive=False,value='💭 Action 5'),
        gr.update(interactive=True),

        gr.update(value=None), # video
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),

        gr.update(value=None), # text
        gr.update(value=None), # image

    )
    




def reset_button():
    return gr.update(interactive=True), 

