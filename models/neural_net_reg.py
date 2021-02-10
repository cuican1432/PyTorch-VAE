import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


# from sklearn.metrics import r2_score

class NeuralNetReg(BaseVAE):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 drop_out: float,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(NeuralNetReg, self).__init__()

        modules = []
        if hidden_dims is None:
            hidden_dims = [100, 100, 100]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=drop_out))
            )
            input_dim = h_dim

        self.model = nn.Sequential(*modules)
        self.final_layer = nn.Linear(input_dim, output_dim)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        output = self.model(input)
        return self.final_layer(output)

    def r2_score(self, y_pred, y):
        n = y.shape[0]
        sum_of_err = torch.sum(torch.pow(y_pred - y, 2)).item()
        y_sum = torch.sum(y).item()
        y_sq_sum = torch.sum(torch.pow(y, 2)).item()
        return 1 - sum_of_err / (y_sq_sum - (y_sum ** 2) / n + 1e-8)

    def train_loss_function(self,
                            *args,
                            **kwargs) -> dict:
        """
        Computes the RMSE and R2_score on all 5 cosmo params together
        :param args:
        :param kwargs:
        :return:
        """
        preds = args[0]
        input = args[1]

        rmse_loss = torch.sqrt(F.mse_loss(preds, input))
        r2_score = self.r2_score(preds, input)
        return {'loss': rmse_loss, 'R_squared_score': r2_score}

    def valid_loss_function(self,
                            *args,
                            **kwargs) -> dict:
        """
        Computes the RMSE and R2_score on all 5 cosmo params seperately
        :param args:
        :param kwargs:
        :return:
        """
        preds = {}
        input = {}
        preds['Om'] = args[0][:, 0]
        preds['Ob2'] = args[0][:, 1]
        preds['h'] = args[0][:, 2]
        preds['ns'] = args[0][:, 3]
        preds['s8'] = args[0][:, 4]
        input['Om'] = args[1][:, 0]
        input['Ob2'] = args[1][:, 1]
        input['h'] = args[1][:, 2]
        input['ns'] = args[1][:, 3]
        input['s8'] = args[1][:, 4]

        rmse_loss = torch.sqrt(F.mse_loss(args[0], args[1]))
        rmse_Om = torch.sqrt(F.mse_loss(preds['Om'], input['Om']))
        rmse_Ob2 = torch.sqrt(F.mse_loss(preds['Ob2'], input['Ob2']))
        rmse_h = torch.sqrt(F.mse_loss(preds['h'], input['h']))
        rmse_ns = torch.sqrt(F.mse_loss(preds['ns'], input['ns']))
        rmse_s8 = torch.sqrt(F.mse_loss(preds['s8'], input['s8']))
        r2_Om = self.r2_score(preds['Om'], input['Om'])
        r2_Ob2 = self.r2_score(preds['Ob2'], input['Ob2'])
        r2_h = self.r2_score(preds['h'], input['h'])
        r2_ns = self.r2_score(preds['ns'], input['ns'])
        r2_s8 = self.r2_score(preds['s8'], input['s8'])
        return {'loss': rmse_loss,
                'RMSE_Om': rmse_Om.item(), 'RMSE_Ob2': rmse_Ob2.item(), 'RMSE_h': rmse_h.item(),
                'RMSE_ns': rmse_ns.item(), 'RMSE_s8': rmse_s8.item(),
                'R_Squared_Om': r2_Om, 'R_Squared_Ob2': r2_Ob2, 'R_Squared_h': r2_h, 'R_Squared_ns': r2_ns,
                'R_Squared_s8': r2_s8}
