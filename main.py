import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from torch.optim import Adam,SGD
from torchvision.utils import save_image
import numpy as np
import argparse
from tqdm import tqdm
from torchvision import datasets, transforms
import random
from model import model, encoder
parser = argparse.ArgumentParser("")
parser.add_argument("--seed", type=int, default=40)
parser.add_argument("--dim_image", type=int, default=28)
parser.add_argument("--dim_hid", type=int, default=500)
#可以通过改变--dim_z复现不同z维度下的结果
parser.add_argument("--dim_z", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--train_epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1.1e-3)
parser.add_argument("--only_reconstruct", type=bool, default=True)
args = parser.parse_args()
#设置随机数种子便于复现结果
def set_seed(seed):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(args.seed)
#加载训练测试数据集
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
#模型的加载
encoder = encoder(args.dim_image*args.dim_image,args.dim_hid,args.dim_z)
decoder = nn.Sequential(nn.Linear(args.dim_z, args.dim_hid),nn.ReLU(),nn.Linear(args.dim_hid, args.dim_hid),nn.ReLU(),nn.Linear(args.dim_hid, args.dim_image*args.dim_image),nn.Sigmoid())
model = model(encoder=encoder, decoder=decoder).to("cuda")
#损失函数定义（2个部分：重建损失BCE loss，分布的KL散度）
def F_loss(x, generate, mean, var):
    recon_loss = nn.functional.binary_cross_entropy(generate, x, reduction='sum')
    if args.only_reconstruct == True:
        KL_divergence = 0
    else:
        KL_divergence = - torch.sum(1+ var - mean.pow(2) - var.exp())/2
    return recon_loss + KL_divergence
#定义优化器
optimizer = Adam(model.parameters(), lr=args.lr)
#训练过程
for epoch in range(args.train_epochs):
    overall_loss = 0
    with tqdm(total=len(train_loader),colour='MAGENTA') as pbar:
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(args.batch_size, args.dim_image*args.dim_image)
            data = data.to("cuda")
            optimizer.zero_grad()
            generate, mean, var = model(data)
            loss = F_loss(data, generate, mean, var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.update(1)
    print("第", epoch + 1, "个epoch训练后", "loss平均为: ", overall_loss / (batch_idx*args.batch_size))

# iterator over test set
#图像生成过程
z_list = []
if args.dim_z == 1:
    #只优化重建误差的图像生成
    if args.only_reconstruct == True:
        for i in np.arange(-16,5,0.1):
            z = torch.tensor([i],dtype=torch.float).cuda()
            z = z.view(1,-1)
            z_list.append(z)
        z_cat = torch.cat(z_list,dim=0)
        generate = model.Decoder(z_cat)
        nelem = generate.size(0)
        nrow = 50
        save_image(generate.view(nelem,1,28,28), 'generated\kdn'+str(args.dim_z)+'z_start'+str(i) + '.png', nrow=nrow)
    else:
    #z的维度为1时的图像生成
        for i in np.arange(-1.5,1.5,0.01):
            z = torch.tensor([i],dtype=torch.float).cuda()
            z = z.view(1,-1)
            z_list.append(z)
        z_cat = torch.cat(z_list,dim=0)
        generate = model.Decoder(z_cat)
        nelem = generate.size(0)
        nrow = 50
        save_image(generate.view(nelem,1,28,28), 'generated\generated_image_dim_z_'+str(args.dim_z)+'z_start'+str(i) + '.png', nrow=nrow)
#z的维度为2时的图像生成
if args.dim_z == 2:
    for i in np.arange(-5,5,0.2):
        for j in np.arange(-5,5,0.2):
            z = torch.tensor([i,j],dtype=torch.float).cuda()
            z = z.view(1,-1)
            z_list.append(z)
    z_cat = torch.cat(z_list,dim=0)
    x_hat = model.Decoder(z_cat)
    nelem = x_hat.size(0)
    nrow  = 50
    save_image(x_hat.view(nelem,1,28,28), 'generated\generated_image_dim_z_'+str(args.dim_z)+'z_start'+str(i) + '.png', nrow=nrow)
# to be finished by you 
