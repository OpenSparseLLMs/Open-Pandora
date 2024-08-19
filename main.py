
    
import torch
from demo_utils import *
import transformers
import pytorch_lightning as pl
from collections import OrderedDict
from einops import repeat, rearrange, repeat
from typing import Optional, Tuple, Union, List, Dict
from DynamiCrafter.lvdm.models.samplers.ddim import DDIMSampler
import logging
mainlogger = logging.getLogger('mainlogger')

class Pandora(pl.LightningModule):
    def __init__(self, model_path, lightning_config):
        super(Pandora, self).__init__()
        self.model, processor = load_wm(repo_id =model_path)
        self.lr = lr
        self.save_hyperparameters(ignore=["video_model", "diffusion_model"])

    def get_input(self, x):
        '''
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        '''
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    
    def get_batch_input(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        diffusion_pixel_values: torch.FloatTensor = None,
        diffusion_cond_image: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        random_uncond=False,
        cfg_img=None,
        noise_shape=None,
        multiple_cond_cfg=False,
        unconditional_guidance_scale=1.0,
        **generate_kwargs, # max_new_tokens, guidance_scale
    ):
        ## x: b c t h w
        '''
        'image': (1,336,596,3)
        'video': (1, 3, 16, 320, 512) #normalnize and reshape
        'caption': list[str]
        'path': list[str]
        'fps': tensor([5.], device='cuda:0', dtype=torch.float16)
        'frame_stride': tensor([5], device='cuda:0')
        '''
        assert input_ids.size(0) == 1, "Currently only support batch size 1"
        assert input_ids[0][-1] == tokenizer.image_prefix_token_id

        if noise_shape==None:
            noise_shape=[1, 4, self.diffusion_model.temporal_length, h//8, w//8],
        past_key_values = None
        output_sequence = input_ids
        gen_images = []
        
        img_feat_num = self.config.image_prefix_length if self.config.use_image_prefix else self.config.num_query_tokens

        diffusion_conditioning = self.get_diffusion_conditioning(input_ids, pixel_values, attention_mask, True, None, None)
        cond_emb = diffusion_conditioning[-1:] # Only generate last video

        h, w = diffusion_pixel_values.shape[-2:]

        ## x: b c t h w
        x = self.get_input(diffusion_cond_image)
        ## encode video frames x to z via a 2D encoder        
        z = self.encode_first_stage(x)
        
        ## get caption condition
        ddim_sampler = DDIMSampler(self.diffusion_model) if not multiple_cond_cfg else DDIMSampler_multicond(self.diffusion_model)
        batch_size = noise_shape[0]
        fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=self.diffusion_model.device)

        # img = videos[:,:,0] #bchw
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)

        img = x[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = z[:,:,cond_frame_index,:,:]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

        out = [z, cond]

        ## get conditioning frame

        out = [z, cond, fs]
        
        ''' 
        z:(1,4,64,40,64) 
        cond:
            c_crossattn: (1, 356+77, 1024)
            c_concat: (1,4,64,40,64) 
        fs: 5
        '''
        return out
    
    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self.diffusion_model(x, c, **kwargs)
        
        return loss, loss_dict    
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.diffusion_model.classifier_free_guidance)
        ## sync_dist | rank_zero_only 
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        #self.log("epoch/global_step", self.global_step.float(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        '''
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        '''
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
