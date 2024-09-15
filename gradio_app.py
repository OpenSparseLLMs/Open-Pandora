import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import gradio as gr
from model import ChatWM, load_wm

from argparse import ArgumentParser
# import debugpy
# debugpy.listen(address=('0.0.0.0',7678))
# debugpy.wait_for_client()
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


args = parse_args()

if args.ckpt_path:
    repo_id = args.ckpt_path
else:
    repo_id = find_latest_checkpoint()
ckpt_name = repo_id.split('/')[-1]

if args.debug:
    model = None
    processor = {
        'image_processor': None,
        'diffusion_image_processor': None,
        'tokenizer': None
    }
else:
    model, processor = load_wm(repo_id =repo_id)
    model = model.to(device=torch_device, dtype=torch.bfloat16).eval()
    pretrained_ckpt = '/mnt/petrelfs/tianjie/projects/Pandora/output/finetune/checkpoints/epoch=3-step=3000.ckpt/checkpoint/mp_rank_00_model_states.pt'
    model_state = torch.load(pretrained_ckpt, weights_only=False)['module']
    model_state = {k.replace('_forward_module.',''):v for k,v in model_state.items()}
    model.load_state_dict(model_state, strict=True)
    del model_state
    # import pdb;pdb.set_trace()
    # torch.save(model.state_dict(), '/mnt/petrelfs/tianjie/projects/Pandora/Open-Pandora/pytorch_model.bin')
    # import pdb;pdb.set_trace()
chatwm = ChatWM(model,processor)


# title = """<h1 align="center"><a href="https://maitrix.org/"><img src="https://maitrix.org/assets/img/logo-title.png" alt="World Model" border="0" style="margin: 0 auto; height: 50px;" /></a> </h1>"""
description = (
    """<br><a href='http://71.142.245.226:8583/'>
    # Pandora
    <img src='https://img.shields.io/badge/Github-Code-blue'></a><p>
    - Upload An Image
    - Press Generate
    """
)

demo = gr.Blocks(theme=gr.themes.Soft(primary_hue="slate",))
with demo:
    # gr.Markdown(title)
    gr.Markdown(description)
    if args.debug:
        gr.Markdown("***Debug Mode, No Model loaded***")

    gr.Markdown(f"Current checkpoint: {ckpt_name}")
    with gr.Tabs():
        with gr.Row():
            with gr.Column(visible=True, scale=0.6)  as input_raws:
                image_input = gr.Image(label='Current State',height=320,width=1024)
                text_input = gr.Textbox(label='Text Control Action')
                with gr.Row():
                    round1_button = gr.Button("💭 Action 1",visible=True, interactive=True,variant="primary")
                    round2_button = gr.Button("💭 Action 2",visible=True, interactive=False,variant="primary")
                    round3_button = gr.Button("💭 Action 3",visible=True, interactive=False,variant="primary")
                with gr.Row():
                    round4_button = gr.Button("💭 Action 4",visible=True, interactive=False,variant="primary")
                    round5_button = gr.Button("💭 Action 5",visible=True, interactive=False,variant="primary")
                    multi_button = gr.Button("💭 Multi-Action",visible=True, interactive=True,variant="primary")

                with gr.Row():
                    clear_button = gr.Button("Clear",visible=True, interactive=True)
                gr.Markdown(" ")
            with gr.Column(visible=True, scale=0.4)  as input_raws:
                fs, n_samples, unconditional_guidance_scale, ddim_steps, ddim_eta, num_round = init_sliders()
                gr.Markdown(" ")
        with gr.Row():
            video_output_0 = gr.Video(width=512,height=320,label='Final Output')
            video_output_1 = gr.Video(width=512,height=320, label='Action 1')
            video_output_2 = gr.Video(width=512,height=320, label='Action 2')
        with gr.Row():
            video_output_3 = gr.Video(width=512,height=320, label='Action 3')
            video_output_4 = gr.Video(width=512,height=320, label='Action 4')
            video_output_5 = gr.Video(width=512,height=320, label='Action 5')
        with gr.Row():
            examples = gr.Examples(
                examples=[
                    ['examples/car.png', 'The car moves forward.'],
                    ['examples/bench.png', 'Wind flows the leaves.'],
                    ['examples/red_car.png', 'Explosion happens.'],
                ],
                inputs=[image_input, text_input, ddim_steps, fs, 
                        n_samples,unconditional_guidance_scale, ddim_eta]
            )

    video_output = [video_output_0, video_output_1, video_output_2, video_output_3, video_output_4, video_output_5]
    button_output = [round1_button,round2_button,round3_button,round4_button,round5_button, multi_button]
    text_image_output = [image_input, text_input]
    total_output = button_output + video_output + text_image_output

    round1_button.click(chatwm.generate_video, inputs=[image_input, text_input, ddim_steps, fs, 
                                                      n_samples,unconditional_guidance_scale, ddim_eta], outputs=[video_output_0, video_output_1, round1_button, round2_button, multi_button])
    round2_button.click(chatwm.generate_video_next_round2, inputs=[text_input, ddim_steps, fs, 
                                                      n_samples,unconditional_guidance_scale, ddim_eta], outputs=[video_output_0, video_output_2,round2_button, round3_button])
    round3_button.click(chatwm.generate_video_next_round3, inputs=[text_input, ddim_steps, fs, 
                                                      n_samples,unconditional_guidance_scale, ddim_eta], outputs=[video_output_0, video_output_3,round3_button, round4_button])
    round4_button.click(chatwm.generate_video_next_round4, inputs=[text_input, ddim_steps, fs, 
                                                        n_samples,unconditional_guidance_scale, ddim_eta], outputs=[video_output_0, video_output_4,round4_button, round5_button])
    round5_button.click(chatwm.generate_video_next_round5, inputs=[text_input, ddim_steps, fs, 
                                                        n_samples,unconditional_guidance_scale, ddim_eta], outputs=[video_output_0, video_output_5, round5_button, round1_button])
    multi_button.click(chatwm.generate_video_mutliround, inputs=[image_input, text_input, ddim_steps, fs, 
                                                      n_samples,unconditional_guidance_scale, ddim_eta, num_round], outputs=[video_output_0,round2_button,round3_button,round4_button,round5_button])
    clear_button.click(gradio_reset,outputs=total_output)
demo.queue()
demo.launch(share=False, server_name='0.0.0.0', server_port=10040)
