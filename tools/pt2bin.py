import os
import argparse
import torch

def process_checkpoint(input_folder, output_folder):
    # 构造预训练检查点文件的完整路径
    pretrained_ckpt = os.path.join(input_folder,'mp_rank_00_model_states.pt')

    # 加载预训练检查点
    model_state = torch.load(pretrained_ckpt)['module']

    # 处理模型状态字典，移除 '_forward_module.' 前缀
    model_state = {k.replace('_forward_module.', ''): v for k, v in model_state.items()}

    # 保存处理后的模型状态字典
    torch.save(model_state, os.path.join(output_folder, 'pytorch_model.bin'))

def main():
    parser = argparse.ArgumentParser(description='Process a checkpoint file and save the model state.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='The directory containing the .ckpt checkpoint file.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='The path where the processed model will be saved.')

    args = parser.parse_args()

    # 检查输入目录是否存在
    if not os.path.exists(args.input_folder) or not os.path.isdir(args.input_folder):
        print(f"Error: The input folder '{args.input_folder}' does not exist or is not a directory.")
        return

    # 确保输出文件的父目录存在
    output_dir = os.path.dirname(args.output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_checkpoint(args.input_folder, args.output_folder)

if __name__ == '__main__':
    main()