import yaml
import argparse
import numpy as np
import glob
import json

from models import *
from experiment import VAEXperiment
from nn_reg_experiment import RegExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from yaml import load, dump

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

parser.add_argument('--reg_only', default=False, type=lambda x: (str(x).lower() in ['true', '1', 'y']))

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    version=config['logging_params']['version'],
    debug=False,
    create_git_tag=False,
    prefix='vae_',
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])


runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                 logger=tt_logger,
                 val_check_interval=1.,
                 num_sanity_val_steps=5,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
if not args.reg_only:
    runner.fit(experiment)

# NN_Reg part starts here...

dir_path = f"{config['logging_params']['save_dir']}/{config['logging_params']['name']}/version_{config['logging_params']['version']}"
ckpt_path = glob.glob(dir_path + '/checkpoints/*')[0]
experiment = VAEXperiment.load_from_checkpoint(ckpt_path, vae_model=model, params=config['exp_params'],
                                               map_location='cuda:0')

nn_reg_tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
    version=f"version_{config['logging_params']['version']}/nn_reg",
    prefix='nn_reg_',
)

nn_reg_model = vae_models[config['reg_params']['name']](**config['reg_params'])
nn_reg_experiment = RegExperiment(experiment.model, nn_reg_model,
                                  config['reg_exp_params'])

nn_reg_runner = Trainer(default_root_dir=f"{nn_reg_tt_logger.save_dir}",
                        logger=nn_reg_tt_logger,
                        val_check_interval=1.,
                        num_sanity_val_steps=5,
                        **config['trainer_params'])

print(f"======= Training {config['reg_params']['name']} =======")
nn_reg_runner.fit(nn_reg_experiment)
test_loss = nn_reg_runner.test()

with open(f'{dir_path}/nn_log.json', 'w') as fp:
    json.dump(test_loss[0], fp)

