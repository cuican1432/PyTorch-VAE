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
        
        return 1 - sum_of_err / (y_sq_sum - (y_sum ** 2) / n)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        preds = args[0]
        input = args[1]

        rmse_loss = torch.sqrt(F.mse_loss(preds, input))
        r2_score = self.r2_score(preds, input)
        return {'RMSE_loss': rmse_loss, 'R_squared_score': r2_score}
