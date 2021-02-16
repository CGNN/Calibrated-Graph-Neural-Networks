import torch
from torch import nn

class SelfExpress(nn.Module):
  def __init__(self, n_hidden):
    super(SelfExpress, self).__init__()
    self.coef = nn.Parameter(torch.zeros((n_hidden, n_hidden)))
  def forward(self, z):
    # embed()
    # embed()
    # z need to be transformed
    z_c = torch.mm(self.coef, z)
    
    # reg_loss = self.coef.pow(2).sum()
    recon_loss = 0.5 * (z_c - z).pow(2).sum()
    return recon_loss # + reg_loss