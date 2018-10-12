import torch
from torch.nn import Module


# log(DC(zc)) + log(1 − DC (enc(x)))
class LC(Module):
    def __init__(self):
        super(LC, self).__init__()

    def forward(self, d_c_enc_x, d_c_z_c):
        return torch.sum(torch.log(d_c_z_c) + torch.log(1.0 - d_c_enc_x))


# xnoise = dec(zc)
# log(DI (xj )) + log(1 − DI (xnoise)) + log(1 − DI (xrec))
class LI(Module):
    def __init__(self):
        super(LI, self).__init__()

    def forward(self, d_i_x, d_i_dec_z_c, d_i_x_rec):
        return torch.sum(torch.log(d_i_x) + torch.log(1.0 - d_i_dec_z_c) + torch.log(1.0 - d_i_x_rec))


class LRec(Module):
    def __init__(self):
        super(LRec, self).__init__()

    def forward(self, x, x_rec):
        return torch.norm(x - x_rec)
