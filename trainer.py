import argparse, os, sys, datetime
from omegaconf import OmegaConf
from transformers import logging as transf_logging
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
import torch
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.append('./DynamiCrafter')
from DynamiCrafter.scripts.evaluation.inference import load_model_checkpoint, instantiate_from_config
from utils.utils_train import get_trainer_callbacks, get_trainer_logger, get_trainer_strategy
from utils.utils_train import set_logger, init_workspace, load_checkpoints
from utils.utils_data import DataModuleFromConfig
from data.webvid_bot3 import Vimeo, WebVid
from data.openvid_s3 import OpenVid
from data.panda import Panda
from model import load_wm
import debugpy
debugpy.listen(address=('0.0.0.0',7678))
debugpy.wait_for_client()

torch.set_float32_matmul_precision('medium')

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--seed", "-s", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--name", "-n", type=str, default="", help="experiment name, as saving folder")
    parser.add_argument("--base", "-b", nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. "
                            "Parameters can be overwritten or added with command-line options of the form `--key value`.", default=list())
    parser.add_argument("--model_path", "-m", type=str, default="", help="pretrained model")
    parser.add_argument("--train", "-t", action='store_true', default=False, help='train')
    parser.add_argument("--val", "-v", action='store_true', default=False, help='val')
    parser.add_argument("--test", action='store_true', default=False, help='test')
    parser.add_argument("--logdir", "-l", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--auto_resume", action='store_true', default=False, help="resume from full-info checkpoint")
    parser.add_argument("--auto_resume_weight_only", action='store_true', default=False, help="resume from weight-only checkpoint")
    parser.add_argument("--debug", "-d", action='store_true', default=False, help="enable post-mortem debugging")
    parser.add_argument("--do_alignment", action='store_true', default=False, help="whether or not you do alignment training")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    return parser

def setup_environment(args):
    transf_logging.set_verbosity_error()
    seed_everything(args.seed)
    workdir, ckptdir, cfgdir, loginfo = init_workspace(args.name, args.logdir, OmegaConf.create(), {}, rank=0)
    logger = set_logger(logfile=os.path.join(loginfo, f'log_0:{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.txt'))
    return workdir, ckptdir, cfgdir, loginfo, logger

def configure_trainer(args, lightning_config, workdir, ckptdir, logger):
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    callback_cfg = get_trainer_callbacks(lightning_config, OmegaConf.create(), workdir, ckptdir, logger)

    trainer_kwargs = {
        "num_sanity_val_steps": 0,
        "logger": instantiate_from_config(get_trainer_logger(lightning_config, workdir, args.debug)),
        "callbacks":  [instantiate_from_config(callback_cfg[k]) for k in callback_cfg],
        "strategy": get_trainer_strategy(lightning_config),
        'precision': lightning_config.get('precision', 32),
        "sync_batchnorm": False,
    }
    return Trainer(**trainer_config, **trainer_kwargs)

def load_and_configure_model(config, args):
    model_config = config.pop("model", OmegaConf.create())
    model_config['do_alignment'] = args.do_alignment
    model, processor = load_wm(repo_id=args.model_path, training_args=model_config)
    return model, processor

def load_and_configure_data(config, processor, batch_size):
    dataset = Panda(processor=processor, **config.data)
    data_module = DataModuleFromConfig(batch_size=batch_size, train=dataset, num_workers=config.data.num_workers)
    return data_module

def main():
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    args_dict = vars(args)
    config = OmegaConf.merge(*configs, args_dict)
    lightning_config = config.pop("lightning", OmegaConf.create())
    workdir, ckptdir, cfgdir, loginfo, logger = setup_environment(args)
    trainer = configure_trainer(args, lightning_config, workdir, ckptdir, logger)
    model, processor = load_and_configure_model(config, args)
    dataloader = load_and_configure_data(config, processor, config.data.batch_size)

    if args.train:
        try:
            if lightning_config.strategy.startswith('deepspeed'):
                logger.info("<Training in DeepSpeed Mode>")
                with torch.amp.autocast("cuda"):
                    trainer.fit(model, dataloader)
            else:
                logger.info("<Training in DDPSharded Mode>")
                trainer.fit(model, dataloader)
        except Exception as e:
            raise e
    
    model.eval()
    if args.val:
        trainer.validate(model, dataloader)
    if args.test:
        trainer.test(model, dataloader)

if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     import sys, pdb, bdb
    #     type, value, tb = sys.exc_info()
    #     if type == bdb.BdbQuit or type == SystemExit:
    #         exit()
    #     print(type, value)
    #     pdb.post_mortem(tb)