import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

from sklearn.metrics import r2_score

class model_3hl(nn.Module):
    
    def __init__(self, inp, h1, h2, h3, out, dr):
        super(model_3hl, self).__init__()
        
        self.fc1 = nn.Linear(inp, h1) 
        self.fc2 = nn.Linear(h1,  h2)
        self.fc3 = nn.Linear(h2,  h3)
        self.fc4 = nn.Linear(h3,  out)
        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        # initialize the weights of the different layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or \
                 isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    # forward pass
    def forward(self, x):
        out = self.dropout(self.LeakyReLU(self.fc1(x)))
        out = self.dropout(self.LeakyReLU(self.fc2(out)))
        out = self.dropout(self.LeakyReLU(self.fc3(out)))
        out = self.fc4(out)         
        return out

class VanillaVAE(BaseVAE):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 drop_out: float,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [100, 100, 100]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Squential(
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
        r2_score = r2_score(preds, input)
        return {'RMSE_loss': rmse_loss, 'R_squared_score': r2_score}
