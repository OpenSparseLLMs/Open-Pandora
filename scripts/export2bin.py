import torch

pretrained_ckpt = '/mnt/petrelfs/tianjie/projects/WorldModel/Pandora/output/ckp/checkpoints/epoch=1-step=8000.ckpt/checkpoint/mp_rank_00_model_states.pt'
model_state = torch.load(pretrained_ckpt)['module']
model_state = {k.replace('_forward_module.',''):v for k,v in model_state.items()}
torch.save(model_state, '/mnt/petrelfs/tianjie/projects/WorldModel/Pandora/models/stage2/pytorch_model.bin')