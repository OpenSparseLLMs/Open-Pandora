import torch
from torch import nn
import transformers
from transformers.utils import logging
from transformers import PreTrainedModel, Blip2QFormerModel, AutoModelForCausalLM
from transformers import  CLIPTextConfig, CLIPTextModel
from transformers.modeling_outputs import  BaseModelOutputWithPooling

from transformers.models.clip.modeling_clip import CLIPTextTransformer
import sys
from einops import rearrange
from configuration import WorldModelConfig
from typing import Optional, Tuple, Union, List, Dict


from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, CLIPTokenizer

from ChatUniVi.constants import DEFAULT_IMAGE_TOKEN
from ChatUniVi.model import ChatUniViLlamaForCausalLM, ChatUniViConfig

sys.path.append('./DynamiCrafter')
from DynamiCrafter.scripts.evaluation.inference import load_model_checkpoint, instantiate_from_config
from DynamiCrafter.lvdm.models.samplers.ddim import DDIMSampler
from DynamiCrafter.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from omegaconf import OmegaConf
from einops import repeat
from transformers import logging
logging.set_verbosity_error()
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
_make_causal_mask = AttentionMaskConverter._make_causal_mask

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
IMAGE_PREFIX_TOKEN = "[IMG_P]"


logger = logging.get_logger(__name__)

def freeze_sub_models(function):
    def wrapper(*args, **kwargs):
        model = function(*args, **kwargs)
        if model.config.freeze_video_model:
            for param in model.video_model.parameters():
                param.requires_grad = False

        if model.config.use_diffusion_text_encoder and model.config.freeze_diffusion_text_encoder:
            for param in model.diffusion_text_encoder.parameters():
                param.requires_grad = False
        if model.config.use_image_callbacks:
            for param in model.diffusion_original_text_encoder.parameters():
                param.requires_grad = False
        if model.config.freeze_diffusion_qformer:
            for param in model.diffusion_qformer.parameters():
                param.requires_grad = False
            for param in model.diffusion_qformer_proj.parameters():
                param.requires_grad = False
            model.diffusion_query_tokens.requires_grad = False

            for param in model.diffusion_proj.parameters():
                param.requires_grad = False

        return model
    return wrapper



class WorldModel(PreTrainedModel):
    config_class = WorldModelConfig
    sub_models = ['video_model']
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)

        if config.use_image_prefix:
            self.image_prefix = nn.Linear(config.video_model_config.hidden_size, config.image_prefix_length, bias=False)

        if config.use_flash_attn:
            video_model_config = self._check_and_enable_flash_attn_2(config=config.video_model_config)
        else:
            video_model_config = config.video_model_config
            video_model_config._flash_attn_2_enabled = False

        if config.use_image_tokenizer:
            self.image_embeddings = nn.Embedding(config.image_vocab_size, config.video_model_config.hidden_size)
        self.diffusion_qformer_proj = nn.Linear(config.video_model_config.hidden_size, config.diffusion_qformer_config.hidden_size)
        self.diffusion_qformer = Blip2QFormerModel(config.diffusion_qformer_config)
        
        self.diffusion_query_tokens = nn.Parameter(torch.zeros(config.diffusion_text_encoder_config.max_position_embeddings, config.diffusion_qformer_config.hidden_size))
        
        self.diffusion_proj = nn.Linear(config.diffusion_qformer_config.hidden_size, config.diffusion_proj_out_dim)

        if config.use_image_callbacks:
            self.diffusion_original_text_encoder = CLIPTextModel.from_pretrained(config.diffusion_model_name_or_path, subfolder="text_encoder")
            self.diffusion_tokenizer = CLIPTokenizer.from_pretrained(config.diffusion_model_name_or_path,subfolder='tokenizer')
        if config.use_diffusion_text_encoder:
            self.diffusion_text_encoder = CLIPTextEmbeddingModel(config.diffusion_text_encoder_config)

        self.post_init()
        
        self.video_model = AutoModelForCausalLM.from_pretrained(config.video_model_name_or_path, config=video_model_config)
        for module in self.video_model.modules():
            module._is_hf_initialized = True

        if not config.do_alignment:
            model_config = OmegaConf.load(config.dynamicrafter)
            model_config = model_config.pop("model", OmegaConf.create())
            model_config['params']['unet_config']['params']['use_checkpoint'] = False
            self.diffusion_model = instantiate_from_config(model_config)
            self.diffusion_model.perframe_ae = True
            # load_model_checkpoint(self.diffusion_model, config.dynamicrafter_ckpt)
            for module in self.diffusion_model.modules():
                module._is_hf_initialized = True

        if config.use_image_tokenizer:
            self.image_embeddings.weight.data.normal_(mean=0.0, std=0.5)



    @classmethod
    @freeze_sub_models
    def from_pretrained(cls, *args, **kwargs):
        return super(WorldModel, cls).from_pretrained(*args, **kwargs)


    def get_diffusion_conditioning(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        diffusion_tgt_mask: Optional[torch.LongTensor] = None,
    ):
        # print(f'11 normal {pixel_values.shape=}', flush=True)
        # Copy and modify from ChatUniVi forward function ------------------------------------------------------------------------------------------------------------------
        video_model = self.video_model
        output_attentions = output_attentions if output_attentions is not None else video_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else video_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else video_model.config.use_return_dict

        # Use labels to keep track of the location of image prefix tokens since image features will change the length of the sequence
        image_prefix_token_id = self.video_model.config.vocab_size + 1 # ugly hardcode here
        labels = input_ids.clone()
        input_ids[input_ids.eq(image_prefix_token_id)] = 0
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = video_model.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, None, labels, pixel_values)
        
        # print('11 normal', flush=True)
        if self.config.use_image_prefix:
            bs, seq_len = labels.shape
            labels = labels.reshape(-1)
            image_prefix_mask = labels.eq(image_prefix_token_id)
            inputs_embeds = inputs_embeds.reshape(bs * seq_len, -1)
            
 
            image_num = image_prefix_mask.sum().item() / self.config.image_prefix_length
            assert int(image_num) == image_num
            image_prefix_embeddings = self.image_prefix.weight.repeat(int(image_num), 1)
            
            inputs_embeds[image_prefix_mask] = image_prefix_embeddings
            inputs_embeds = inputs_embeds.reshape(bs, seq_len, -1)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        # print(f'12 normal {inputs_embeds.shape=}', flush=True)
        outputs = video_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # print(f'13 normal {outputs[0].shape}', flush=True)

        output_hidden_states = outputs[0]
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        output_hidden_states = output_hidden_states.reshape(bs * seq_len, -1)
        image_outputs_embeds = output_hidden_states[image_prefix_mask][diffusion_tgt_mask]
        diffusion_loss = None
        img_feat_num = self.config.image_prefix_length # if self.config.use_image_prefix else self.config.num_query_tokens
        diffusion_conditioning = image_outputs_embeds.view(-1, img_feat_num, self.config.video_model_config.hidden_size)
        diffusion_conditioning = self.diffusion_qformer_proj(diffusion_conditioning)
        
        diffusion_query_tokens = self.diffusion_query_tokens.expand(diffusion_conditioning.shape[0], -1, -1)
        diffusion_conditioning = self.diffusion_qformer(
            query_embeds=diffusion_query_tokens,
            encoder_hidden_states=diffusion_conditioning,
        )[0]
        
        diffusion_conditioning = self.diffusion_proj(diffusion_conditioning)
        return diffusion_conditioning


    @staticmethod
    def get_latent_z(model, videos):
        b, c, t, h, w = videos.shape
        x = rearrange(videos, 'b c t h w -> (b t) c h w')
        z = model.encode_first_stage(x)
        z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
        if t == 1:
            zero_pad = repeat(torch.zeros_like(z), 'b c t h w -> b c (repeat t) h w', repeat=3)
            z = torch.cat([z, zero_pad], dim=2)
        z = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=4)
        return z

    def image_guided_synthesis(self, diffusion_conditioning, videos, diffusion_cond_image, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                               unconditional_guidance_scale=1.0, cfg_img=None, fs=None, multiple_cond_cfg=False, loop=False, gfi=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
        ddim_sampler = DDIMSampler(self.diffusion_model) if not multiple_cond_cfg else DDIMSampler_multicond(self.diffusion_model)
        batch_size = noise_shape[0]
        fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=self.diffusion_model.device)

        # img = videos[:,:,0] #bchw
        img = diffusion_cond_image
        img_emb = self.diffusion_model.embedder(img) ## blc
        img_emb = self.diffusion_model.image_proj_model(img_emb)

        # cond_emb = self.diffusion_model.get_learned_conditioning(prompts)
        cond_emb = diffusion_conditioning
        cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
        if self.diffusion_model.model.conditioning_key == 'hybrid':
            z = self.get_latent_z(self.diffusion_model, videos) # b c t h w
            img_cat_cond = z
            cond["c_concat"] = [img_cat_cond] # b c 1 h w
        
        if unconditional_guidance_scale != 1.0:
            if self.diffusion_model.uncond_type == "empty_seq":
                prompts = batch_size * [""]
                uc_emb = self.diffusion_model.get_learned_conditioning(prompts)
            elif self.diffusion_model.uncond_type == "zero_embed":
                uc_emb = torch.zeros_like(cond_emb)
            uc_img_emb = self.diffusion_model.embedder(torch.zeros_like(img)) ## b l c
            uc_img_emb = self.diffusion_model.image_proj_model(uc_img_emb)
            uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
            if self.diffusion_model.model.conditioning_key == 'hybrid':
                uc["c_concat"] = [img_cat_cond]
        else:
            uc = None

        ## we need one more unconditioning image=yes, text=""
        if multiple_cond_cfg and cfg_img != 1.0:
            uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
            if self.diffusion_model.model.conditioning_key == 'hybrid':
                uc_2["c_concat"] = [img_cat_cond]
            kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
        else:
            kwargs.update({"unconditional_conditioning_img_nonetext": None})

        z0 = None
        cond_mask = None

        batch_variants = []
        for _ in range(n_samples):

            if z0 is not None:
                cond_z0 = z0.clone()
                kwargs.update({"clean_cond": True})
            else:
                cond_z0 = None
            if ddim_sampler is not None:

                samples, _ = ddim_sampler.sample(S=ddim_steps,
                                                 conditioning=cond,
                                                 batch_size=batch_size,
                                                 shape=noise_shape[1:],
                                                 verbose=True,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 cfg_img=cfg_img, 
                                                 mask=cond_mask,
                                                 x0=cond_z0,
                                                 fs=fs,
                                                 timestep_spacing=timestep_spacing,
                                                 guidance_rescale=guidance_rescale,
                                                 precision=diffusion_conditioning.dtype,
                                                 **kwargs
                                                 )

            ## reconstruct from latent to pixel space
            batch_images = self.diffusion_model.decode_first_stage(samples)
            batch_variants.append(batch_images)
        ## variants, batch, c, t, h, w
        batch_variants = torch.stack(batch_variants)
        return batch_variants.permute(1, 0, 2, 3, 4, 5)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        diffusion_pixel_values: torch.FloatTensor = None,
        diffusion_cond_image: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        **generate_kwargs, # max_new_tokens, guidance_scale
    ):
        assert input_ids.size(0) == 1, "Currently only support batch size 1"
        past_key_values = None
        output_sequence = input_ids
        gen_images = []
        
        img_feat_num = self.config.image_prefix_length if self.config.use_image_prefix else self.config.num_query_tokens

        assert input_ids[0][-1] == tokenizer.image_prefix_token_id
            
        diffusion_conditioning = self.get_diffusion_conditioning(input_ids, pixel_values, attention_mask, True, None, None)
        diffusion_conditioning = diffusion_conditioning[-1:] # Only generate last video

        h, w = diffusion_pixel_values.shape[-2:]
        samples = self.image_guided_synthesis(diffusion_conditioning=diffusion_conditioning,
                                              videos=diffusion_pixel_values[None, ...],
                                              diffusion_cond_image=diffusion_cond_image,
                                              noise_shape=[1, 4, self.diffusion_model.temporal_length, h//8, w//8],
                                              **generate_kwargs)
        return samples
        

class CLIPTextEmbeddingTransformer(CLIPTextTransformer):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            # raise ValueError("You have to specify input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        else:
            assert inputs_embeds is not None

            input_shape = inputs_embeds.size()[:-1]
            hidden_states = inputs_embeds

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)


        if not return_dict:
            return (last_hidden_state) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPTextEmbeddingModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextEmbeddingTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

import torch
import uuid
import numpy as np
import torchvision
from PIL import Image
from torch import nn
import transformers
from transformers.utils import logging
from transformers import PreTrainedModel, Blip2QFormerModel, AutoModelForCausalLM
from transformers import  CLIPTextConfig, CLIPTextModel
from transformers.modeling_outputs import  BaseModelOutputWithPooling

from transformers.models.clip.modeling_clip import CLIPTextTransformer
import sys
from einops import rearrange
from configuration import WorldModelConfig
from typing import Optional, Tuple, Union, List, Dict
from transformers import AutoTokenizer, AutoConfig
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, CLIPTokenizer
from pytorch_lightning.utilities import grad_norm
from ChatUniVi.constants import DEFAULT_IMAGE_TOKEN
from ChatUniVi.model import ChatUniViLlamaForCausalLM, ChatUniViConfig
import pytorch_lightning as pl  
import gradio as gr
import torchvision.transforms as transforms

sys.path.append('./DynamiCrafter')
from DynamiCrafter.scripts.evaluation.inference import load_model_checkpoint, instantiate_from_config
from DynamiCrafter.lvdm.models.samplers.ddim import DDIMSampler
from DynamiCrafter.lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from omegaconf import OmegaConf
from einops import repeat
from transformers import logging
logging.set_verbosity_error()
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from functools import partial
ckpt = torch.utils.checkpoint.checkpoint
_make_causal_mask = AttentionMaskConverter._make_causal_mask

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
IMAGE_PREFIX_TOKEN = "[IMG_P]"

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

def load_wm(repo_id,training_args=None, model=None):
    '''load model, image processor and tokenizer'''
    
    ckpt_name = repo_id.split('/')[-1]
    print(f"Start to load model, current ckpt is: {ckpt_name}")
    config = WorldModelConfig.from_pretrained(repo_id)
    
    if training_args is not None:
        config.reset_training_args(
            do_alignment=training_args.do_alignment,
            learning_rate=training_args.learning_rate
            )
    else:
        config.reset_training_args(
            do_alignment=False,
            )
        
    if model == None:
        model = WorldModel.from_pretrained(repo_id, config=config, ignore_mismatched_sizes=True)

    # load image processors
    image_processor = model.video_model.get_vision_tower().image_processor
    diffusion_image_processor= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    # load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.image_start_token_id = tokenizer.convert_tokens_to_ids("<img_s>")
    tokenizer.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    tokenizer.image_prefix_token_id = tokenizer.convert_tokens_to_ids("[IMG_P]")
    processor = {
        'image_processor':image_processor,
        'diffusion_image_processor':diffusion_image_processor,
        'tokenizer':tokenizer
    }
    return model, processor


def dynamic_resize(img):
    '''resize frames'''
    trans = transforms.Compose([
            transforms.Resize(min((576, 1024))),
            transforms.CenterCrop((576, 1024))
            ])
    return trans(img)



def freeze_sub_models(function):
    def wrapper(*args, **kwargs):
        model = function(*args, **kwargs)
        model.train()
        if model.config.freeze_video_model:
            # model.video_model.eval()

            for param in model.video_model.parameters():
                param.requires_grad = False

        if model.config.use_diffusion_text_encoder and model.config.freeze_diffusion_text_encoder:
            model.diffusion_text_encoder.eval()

            for param in model.diffusion_text_encoder.parameters():
                param.requires_grad = False

        if model.config.use_image_callbacks:
            for param in model.diffusion_original_text_encoder.parameters():
                param.requires_grad = False

        if model.config.freeze_diffusion_qformer:
            model.diffusion_qformer.eval()
            model.diffusion_qformer_proj.eval()
            model.diffusion_query_tokens.eval()
            model.diffusion_proj.eval()

            for param in model.diffusion_qformer.parameters():
                param.requires_grad = False
            for param in model.diffusion_qformer_proj.parameters():
                param.requires_grad = False
            model.diffusion_query_tokens.requires_grad = False

            for param in model.diffusion_proj.parameters():
                param.requires_grad = False

        return model
    return wrapper



class WorldModel(PreTrainedModel, pl.LightningModule):
    config_class = WorldModelConfig
    sub_models = ['video_model']
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: WorldModelConfig):
        super().__init__(config)

        if config.use_image_prefix:
            self.image_prefix = nn.Linear(config.video_model_config.hidden_size, config.image_prefix_length, bias=False)

        if config.use_flash_attn:
            video_model_config = self._check_and_enable_flash_attn_2(config=config.video_model_config)
        else:
            video_model_config = config.video_model_config
            video_model_config._flash_attn_2_enabled = False

        if config.use_image_tokenizer:
            self.image_embeddings = nn.Embedding(config.image_vocab_size, config.video_model_config.hidden_size)
        self.diffusion_qformer_proj = nn.Linear(config.video_model_config.hidden_size, config.diffusion_qformer_config.hidden_size)
        self.diffusion_qformer = Blip2QFormerModel(config.diffusion_qformer_config)
        
        self.diffusion_query_tokens = nn.Parameter(torch.zeros(config.diffusion_text_encoder_config.max_position_embeddings, config.diffusion_qformer_config.hidden_size))
        
        self.diffusion_proj = nn.Linear(config.diffusion_qformer_config.hidden_size, config.diffusion_proj_out_dim)

        if config.use_image_callbacks:
            self.diffusion_original_text_encoder = CLIPTextModel.from_pretrained(config.diffusion_model_name_or_path, subfolder="text_encoder")
            self.diffusion_tokenizer = CLIPTokenizer.from_pretrained(config.diffusion_model_name_or_path,subfolder='tokenizer')
        if config.use_diffusion_text_encoder:
            self.diffusion_text_encoder = CLIPTextEmbeddingModel(config.diffusion_text_encoder_config)

        self.post_init()
        
        self.video_model = AutoModelForCausalLM.from_pretrained(config.video_model_name_or_path, config=video_model_config)
        for module in self.video_model.modules():
            module._is_hf_initialized = True

        if not config.do_alignment:
            model_config = OmegaConf.load(config.dynamicrafter)
            model_config = model_config.pop("model", OmegaConf.create())
            self.diffusion_model = instantiate_from_config(model_config)
            self.diffusion_model.perframe_ae = True
            # load_model_checkpoint(self.diffusion_model, config.dynamicrafter_ckpt)
            for module in self.diffusion_model.modules():
                module._is_hf_initialized = True

        if config.use_image_tokenizer:
            self.image_embeddings.weight.data.normal_(mean=0.0, std=0.5)



    @classmethod
    @freeze_sub_models
    def from_pretrained(cls, *args, **kwargs):
        return super(WorldModel, cls).from_pretrained(*args, **kwargs)


    def get_diffusion_conditioning(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        diffusion_tgt_mask: Optional[torch.LongTensor] = None,
    ):
        # print(f'11 normal {pixel_values.shape=}', flush=True)
        # Copy and modify from ChatUniVi forward function ------------------------------------------------------------------------------------------------------------------
        video_model = self.video_model
        output_attentions = output_attentions if output_attentions is not None else video_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else video_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else video_model.config.use_return_dict

        # Use labels to keep track of the location of image prefix tokens since image features will change the length of the sequence
        image_prefix_token_id = self.video_model.config.vocab_size + 1 # ugly hardcode here
        labels = input_ids.clone()
        input_ids[input_ids.eq(image_prefix_token_id)] = 0
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = video_model.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, None, labels, pixel_values)
        
        # print('11 normal', flush=True)
        if self.config.use_image_prefix:
            bs, seq_len = labels.shape
            labels = labels.reshape(-1)
            image_prefix_mask = labels.eq(image_prefix_token_id)
            inputs_embeds = inputs_embeds.reshape(bs * seq_len, -1)
            
 
            image_num = image_prefix_mask.sum().item() / self.config.image_prefix_length
            assert int(image_num) == image_num
            image_prefix_embeddings = self.image_prefix.weight.repeat(int(image_num), 1)
            
            inputs_embeds[image_prefix_mask] = image_prefix_embeddings
            inputs_embeds = inputs_embeds.reshape(bs, seq_len, -1)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        # print(f'12 normal {inputs_embeds.shape=}', flush=True)
        outputs = video_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # print(f'13 normal {outputs[0].shape}', flush=True)

        output_hidden_states = outputs[0]
        #--------------------------------------------------------------------------------------------------------------------------------------------------------------------
        output_hidden_states = output_hidden_states.reshape(bs * seq_len, -1)
        image_outputs_embeds = output_hidden_states[image_prefix_mask][diffusion_tgt_mask]
        diffusion_loss = None
        img_feat_num = self.config.image_prefix_length # if self.config.use_image_prefix else self.config.num_query_tokens
        diffusion_conditioning = image_outputs_embeds.view(-1, img_feat_num, self.config.video_model_config.hidden_size)
        diffusion_conditioning = self.diffusion_qformer_proj(diffusion_conditioning)
        
        diffusion_query_tokens = self.diffusion_query_tokens.expand(diffusion_conditioning.shape[0], -1, -1)
        diffusion_conditioning = self.diffusion_qformer(
            query_embeds=diffusion_query_tokens,
            encoder_hidden_states=diffusion_conditioning,
        )[0]
        
        diffusion_conditioning = self.diffusion_proj(diffusion_conditioning)
        return diffusion_conditioning


    @staticmethod
    def get_latent_z(model, videos):
        b, c, t, h, w = videos.shape
        x = rearrange(videos, 'b c t h w -> (b t) c h w')
        z = model.encode_first_stage(x)
        z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
        if t == 1:
            # zero_pad = repeat(torch.zeros_like(z), 'b c t h w -> b c (repeat t) h w', repeat=3)
            # z = torch.cat([z, zero_pad], dim=2)
            z = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=4)
        z = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=4)
        return z

    def image_guided_synthesis(self, diffusion_conditioning, videos, diffusion_cond_image, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                               unconditional_guidance_scale=1.0, cfg_img=None, fs=None, multiple_cond_cfg=False, loop=False, gfi=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
        ddim_sampler = DDIMSampler(self.diffusion_model) if not multiple_cond_cfg else DDIMSampler_multicond(self.diffusion_model)
        batch_size = noise_shape[0]
        fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=self.diffusion_model.device)

        # img = videos[:,:,0] #bchw
        img = diffusion_cond_image
        img_emb = self.diffusion_model.embedder(img) ## blc
        img_emb = self.diffusion_model.image_proj_model(img_emb)

        # cond_emb = self.diffusion_model.get_learned_conditioning(prompts)
        cond_emb = diffusion_conditioning
        cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
        if self.diffusion_model.model.conditioning_key == 'hybrid':
            z = self.get_latent_z(self.diffusion_model, videos) # b c t h w
            img_cat_cond = z
            cond["c_concat"] = [img_cat_cond] # b c 1 h w
        
        if unconditional_guidance_scale != 1.0:
            if self.diffusion_model.uncond_type == "empty_seq":
                prompts = batch_size * [""]
                uc_emb = self.diffusion_model.get_learned_conditioning(prompts)
            elif self.diffusion_model.uncond_type == "zero_embed":
                uc_emb = torch.zeros_like(cond_emb)
            uc_img_emb = self.diffusion_model.embedder(torch.zeros_like(img)) ## b l c
            uc_img_emb = self.diffusion_model.image_proj_model(uc_img_emb)
            uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
            if self.diffusion_model.model.conditioning_key == 'hybrid':
                uc["c_concat"] = [img_cat_cond]
        else:
            uc = None

        ## we need one more unconditioning image=yes, text=""
        if multiple_cond_cfg and cfg_img != 1.0:
            uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
            if self.diffusion_model.model.conditioning_key == 'hybrid':
                uc_2["c_concat"] = [img_cat_cond]
            kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
        else:
            kwargs.update({"unconditional_conditioning_img_nonetext": None})

        z0 = None
        cond_mask = None

        batch_variants = []
        for _ in range(n_samples):

            if z0 is not None:
                cond_z0 = z0.clone()
                kwargs.update({"clean_cond": True})
            else:
                cond_z0 = None
            if ddim_sampler is not None:

                samples, _ = ddim_sampler.sample(S=ddim_steps,
                                                 conditioning=cond,
                                                 batch_size=batch_size,
                                                 shape=noise_shape[1:],
                                                 verbose=True,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta,
                                                 cfg_img=cfg_img, 
                                                 mask=cond_mask,
                                                 x0=cond_z0,
                                                 fs=fs,
                                                 timestep_spacing=timestep_spacing,
                                                 guidance_rescale=guidance_rescale,
                                                 precision=diffusion_conditioning.dtype,
                                                 **kwargs
                                                 )

            ## reconstruct from latent to pixel space
            batch_images = self.diffusion_model.decode_first_stage(samples)
            batch_variants.append(batch_images)
        ## variants, batch, c, t, h, w
        batch_variants = torch.stack(batch_variants)
        return batch_variants.permute(1, 0, 2, 3, 4, 5)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        diffusion_pixel_values: torch.FloatTensor = None,
        diffusion_cond_image: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        **generate_kwargs, # max_new_tokens, guidance_scale
    ):
        assert input_ids.size(0) == 1, "Currently only support batch size 1"
        past_key_values = None
        output_sequence = input_ids
        gen_images = []
        
        img_feat_num = self.config.image_prefix_length if self.config.use_image_prefix else self.config.num_query_tokens

        assert input_ids[0][-1] == tokenizer.image_prefix_token_id

        # print("use FrozenOpenCLIPEmbedder")
        # caption = tokenizer.decode(input_ids[0],skip_special_tokens=True)
        # diffusion_conditioning = self.diffusion_model.cond_stage_model(caption) 

        diffusion_conditioning = self.get_diffusion_conditioning(input_ids, pixel_values, attention_mask, True, None, None)
        diffusion_conditioning = diffusion_conditioning[-1:] # Only generate last video

        h, w = diffusion_pixel_values.shape[-2:]
        samples = self.image_guided_synthesis(diffusion_conditioning=diffusion_conditioning,
                                              videos=diffusion_pixel_values[None, ...],
                                              diffusion_cond_image=diffusion_cond_image,
                                              noise_shape=[1, 4, self.diffusion_model.temporal_length, h//8, w//8],
                                              **generate_kwargs)
        return samples


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
        video,
        input_ids: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        diffusion_pixel_values: torch.FloatTensor = None,
        diffusion_cond_image: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        fps=None,
        frame_stride=None,
        random_uncond=True,
        **kwargs
    ):
        ## x: b c t h w
        assert input_ids.size(0) == 1, "Currently only support batch size 1"
        
        pixel_values = pixel_values[0]
        diffusion_cond_image = diffusion_cond_image[0]
        
        try:
            assert input_ids[0][-1] == 32001
        except AssertionError:
            print("Assertion failed. The value of input_ids[0] is:")
            print(input_ids[0])
        ## x: b c t h w
        x = self.get_input(video)
        z = self.diffusion_model.encode_first_stage(x)
        diffusion_conditioning = self.get_diffusion_conditioning(input_ids, pixel_values, attention_mask, True, None, None)
        cond_emb = diffusion_conditioning[-1:]

        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.diffusion_model.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.diffusion_model.uncond_prob).float() * (random_num < 3 * self.diffusion_model.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.diffusion_model.get_learned_conditioning([""])
        prompt_emb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        img = diffusion_cond_image[0]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.diffusion_model.embedder(img)
        img_emb = self.diffusion_model.image_proj_model(img_emb)

        if self.diffusion_model.model.conditioning_key == 'hybrid':
            ## encode video frames x to z via a 2D encoder        
            img_cat_cond = self.get_latent_z(self.diffusion_model, diffusion_pixel_values)
            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [torch.cat([prompt_emb, img_emb], dim=1)] ## concat in the seq_len dim
        
        out = [z, cond, fps]
        
        return out
    
    def alignment_forward(
        self,
        video,
        caption,
        input_ids: torch.FloatTensor,
        pixel_values: torch.FloatTensor = None,
        diffusion_pixel_values: torch.FloatTensor = None,
        diffusion_cond_image: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        frame_stride=None,
        random_uncond=True,
        **kwargs
    ):
        ## x: b c t h w
        assert input_ids.size(0) == 1, "Currently only support batch size 1"
        
        pixel_values = pixel_values[0]
        diffusion_cond_image = diffusion_cond_image[0]
        caption = caption[0]
        assert input_ids[0][-1] == 32001
        ## x: b c t h w
        diffusion_conditioning = self.get_diffusion_conditioning(input_ids, pixel_values, attention_mask, True, None, None)
        llm_cond_emb = diffusion_conditioning[-1:]
        clip_cond_emb = self.diffusion_model.cond_stage_model(caption)

        # KLDivLoss
        # kl_loss = nn.KLDivLoss(reduction='batchmean')

        # log_llm_cond_emb = torch.log(llm_cond_emb)
        # loss = kl_loss(log_llm_cond_emb, clip_cond_emb)

        # MSELoss
        mse_loss = torch.nn.MSELoss()
        
        loss = mse_loss(llm_cond_emb, clip_cond_emb)

        return loss, {"loss":loss}
  
    
    def training_step(self, batch, batch_idx):
        if not self.config.do_alignment:

            x, c, fs = self.get_batch_input(**batch, random_uncond=False)

            kwargs= {"fs": fs.long()}
            loss, loss_dict = self.diffusion_model(x, c, **kwargs)

        else:
            loss, loss_dict = self.alignment_forward(**batch)
        ## sync_dist | rank_zero_only 
        # loss_dict.update({"global_step": int(self.global_step)})
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)

        return loss
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.diffusion_model.model.diffusion_model, norm_type=2)

        self.log_dict(norms)

    def configure_optimizers(self):
        
        lr = self.config.learning_rate
        params = list()
        if not self.config.do_alignment:
            params = list(self.diffusion_model.model.parameters())

        params.append(self.diffusion_query_tokens)
        params.extend(self.image_prefix.parameters())
        params.extend(list(self.diffusion_proj.parameters()))
        params.extend(list(self.diffusion_qformer.parameters()))
        params.extend(list(self.diffusion_qformer_proj.parameters()))
        
        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        # if self.use_scheduler:
        #     logger.info("Setting up scheduler...")
        #     lr_scheduler = self.configure_schedulers(optimizer)
        #     return [optimizer], [lr_scheduler]
        
        return optimizer
    

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=50, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, mask=None, **kwargs):
        """ log images for LatentVisualDiffusion """
        return {"image":batch["pixel_values"]}

class ChatWM():
    def __init__(self, model, processor, training_args=None, video_path=None):
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
        self.config = training_args
        

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
        self.text = self.tokenizer.bos_token + "<image> " + text_input + "[IMG_P]" * 64
        
        # if type(image) == np.ndarray:
        #     image = Image.fromarray(image) # <class 'numpy.ndarray'> -> <class 'PIL.Image.Image'>
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
        resized_img = dynamic_resize(Image.fromarray(image)) #(4400, 2750, 3) -> 576x1024
        diffusion_pixel_values = self.diffusion_image_processor(resized_img).unsqueeze(1)
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
        

class CLIPTextEmbeddingTransformer(CLIPTextTransformer):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            # raise ValueError("You have to specify input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        else:
            assert inputs_embeds is not None

            input_shape = inputs_embeds.size()[:-1]
            hidden_states = inputs_embeds

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)


        if not return_dict:
            return (last_hidden_state) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CLIPTextEmbeddingModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextEmbeddingTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.text_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
