import yaml
import argparse
import numpy as np
import glob.glob

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

# NN_Reg part starts here...

# Need to load the correct check point...
# Haven't checked these lines since colab does not support glob.glob

# dir_path = config['logging_params']['save_dir'] + config['logging_params']['name']
# versions = glob.glob(dir_path + '/*')
# max_ver_num = max([int(i.split('_')[-1]) for i in versions])
# ckpt_path = glob.glob(dir_path + '/version_{}/checkpoints/*'.format(max_ver_num))[0]

ckpt_path = '../vae_learning_20210204ckpt/logs/VanillaVAE/version_1/checkpoints/epoch=28-step=138648.ckpt'
experiment = VAEXperiment.load_from_checkpoint(ckpt_path, vae_model = model, params = config['exp_params'], map_location = 'cuda:0')

nn_reg_tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['nn_reg_name'],
    debug=False,
    create_git_tag=False,
    # prefix='nn_reg_',
)

# experiment is not on cuda? why?? I've specified map_location in load_from_checkpoint???
# print('experiment model is load on cuda: ', next(experiment.model.encoder.parameters()).is_cuda)

# Therefore I have to assign them to cuda manually...
experiment.model.encoder.cuda()
experiment.model.fc_mu.cuda()
experiment.model.fc_var.cuda()

nn_reg_model = vae_models[config['reg_params']['name']](**config['reg_params'])
nn_reg_experiment = RegExperiment(experiment.model.encode, nn_reg_model,
                                  config['reg_exp_params'])

nn_reg_runner = Trainer(default_root_dir=f"{nn_reg_tt_logger.save_dir}",
                        logger=nn_reg_tt_logger,
                        val_check_interval=1.,
                        num_sanity_val_steps=5,
                        **config['trainer_params'])

print(f"======= Training {config['reg_params']['name']} =======")
nn_reg_runner.fit(nn_reg_experiment)
nn_reg_result = nn_reg_runner.test()
