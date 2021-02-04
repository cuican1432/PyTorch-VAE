import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
from nn_reg_experiment import RegExperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
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
runner.fit(experiment)

# nn_reg part
# nn_reg_model = vae_models[config['reg_params']['name']](**config['reg_params'])
# nn_reg_experiment = RegExperiment(experiment.model.encode, nn_reg_model,
#                                   config['reg_exp_params'])

# nn_reg_runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
#                         # logger=tt_logger,
#                         val_check_interval=1.,
#                         num_sanity_val_steps=5,
#                         **config['trainer_params'])

# print(f"======= Training {config['reg_params']['name']} =======")
# nn_reg_runner.fit(nn_reg_experiment)
