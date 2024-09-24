
import os
import json
import torch
from model import ChatWM, load_wm
from PIL import Image
import torchvision
import numpy as np
from argparse import ArgumentParser
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


missiing = ['a basket of french fries in a fryer']

class Chat(ChatWM):
    def __init__(self, model, processor):
        super(Chat, self).__init__(model, processor)


    def process_generated_video(self, videos, fps=8, video_path='video_output.mp4'):
        video_dir = os.path.dirname(video_path)
        video_base = os.path.basename(video_path).rsplit('.', 1)[0]
        video_path_prefix = os.path.join(video_dir, video_base) if video_dir else video_base

        videos = videos.squeeze(0).detach().cpu().to(torch.float32).clamp(-1., 1.)
        
        for sample_idx in range(videos.shape[0]):
            video = videos[sample_idx].permute(1, 0, 2, 3)
            frame_grids = [torchvision.utils.make_grid(framesheet, nrow=2, padding=0) for framesheet in video]
            grid = torch.stack(frame_grids, dim=0)
            grid = ((grid + 1.) / 2. * 255.).to(torch.uint8).permute(0, 2, 3, 1)
            
            output_path = f"{video_path_prefix}-{sample_idx}.mp4"
            torchvision.io.write_video(output_path, grid, fps=fps, video_codec='h264', options={'crf': '10'})
            
    def generate_video(self, image, text_input, ddim_steps, fs, n_samples,
                       unconditional_guidance_scale, ddim_eta,
                       progress):
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
         
        self.text = self.tokenizer.bos_token + "<image> " + text_input + "[IMG_P]" * 64

        video_dir = os.path.dirname(self.video_path)
        video_base = os.path.basename(self.video_path).rsplit('.', 1)[0]

        video_path_prefix = os.path.join(video_dir, video_base) if video_dir else video_base
        if os.path.exists(f"{video_path_prefix}-4.mp4"):
            print(f"{video_path_prefix} exits!")
            return
        # if type(image) == np.ndarray:
        #     image = Image.fromarray(image)
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
        self.process_generated_video(videos, fps=8, video_path=self.video_path)

        return 

def main():
    ''''input parameters'''
    parser = ArgumentParser()
    parser.add_argument('--vbench_data',type=str,required=True,help='Point to test data dir of vbench.')
    parser.add_argument('--img_crop',type=str,default='8-5',help='The crop ratio of img.')
    parser.add_argument('--save_dir',type=str,required=True,help='The folder in which the final results are stored.')
    parser.add_argument("--ckpt_path",type=str,required=False,default='../ckpt')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM steps.')
    parser.add_argument('--fs', type=int, default=8, help='Some fs parameter.')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of samples to generate.')
    parser.add_argument('--unconditional_guidance_scale', type=float, default=7.5, help='Scale for unconditional guidance.')
    parser.add_argument('--ddim_eta', type=float, default=1.0, help='DDIM eta parameter.')
    parser.add_argument('--start_id', type=int, default=0, help='DDIM eta parameter.')
    parser.add_argument('--end_id', type=int, default=int(1e9), help='DDIM eta parameter.')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    img_crop_dir = os.path.join(args.vbench_data, 'crop', args.img_crop)
    meta_file = os.path.join(args.vbench_data, 'vbench2_i2v_full_info.json')
    A_meta_file = os.path.join(args.vbench_data, 'i2v-bench-info.json')
    prompt_set = set([ i['caption'] for i in json.load(open(A_meta_file))])
    meta = json.load(open(meta_file))[args.start_id:args.end_id]

    model, processor = load_wm(repo_id =args.ckpt_path)
    model = model.to(device=torch_device, dtype=torch.bfloat16).eval()

    chatwm = Chat(model,processor)

    for m in meta:
        img_name = m['image_name']
        prompt = m['prompt_en']

        img_path = os.path.join(img_crop_dir, img_name)
        chatwm.video_path = os.path.join(args.save_dir, f"{prompt}.mp4")
        img = np.array(Image.open(img_path))
        chatwm.generate_video(img, prompt, args.ddim_steps, args.fs, args.n_samples,
                        args.unconditional_guidance_scale, args.ddim_eta, None)

if __name__ == '__main__':
    main()