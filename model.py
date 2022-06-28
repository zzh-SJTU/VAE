import torch
import torch.nn as nn
#encoder部分包括均值和方差的拟合网络
class encoder(nn.Module):
    def __init__(self, dim_image, dim_hid, latent_dim):
        super(encoder, self).__init__()
        self.mean_net  = nn.Linear(dim_hid, latent_dim)
        self.varianece_net = nn.Linear (dim_hid, latent_dim)
        self.fc_layer = nn.Sequential(nn.Linear(dim_image, dim_hid),nn.ReLU(),nn.Linear(dim_hid, dim_hid),nn.ReLU())
    def forward(self, x):
        hid = self.fc_layer(x)
        var = self.varianece_net(hid)   
        mean = self.mean_net(hid)             
        return mean, var
#整个VAE模型包括encoder，decoder两个部分，decoder为简单的线性全连接网络用于生成图像
class model(nn.Module):
    def __init__(self, encoder, decoder):
        super(model, self).__init__()
        self.Encoder = encoder
        self.Decoder = decoder
    def forward(self, x):
        mean, var = self.Encoder(x)
        z = mean + torch.exp(0.5 * var)*torch.randn_like(var).to('cuda')
        generate= self.Decoder(z)
        return generate, mean, var