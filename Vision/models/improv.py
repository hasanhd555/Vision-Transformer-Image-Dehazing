import torch
import torch.nn as nn
import torch.nn.functional as F
class DNet(nn.Module):
    """Dark channel prior learning network"""
    def __init__(self):
        super(DNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=9, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv4 = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.upsample1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.upsample2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.upsample3(out)

        out = self.conv4(out)
        out = self.relu4(out)

        return out

class TNet(nn.Module):
    """Transmission prior learning network"""
    def __init__(self):
        super(TNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=9, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv3 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv4 = nn.Conv2d(in_channels=9, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.upsample1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.upsample2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        out = self.upsample3(out)

        out = self.conv4(out)
        out = self.sigmoid(out)

        return out

class GIFBlock(nn.Module):
    """Guided Image Filtering block"""
    def __init__(self, radius=1, eps=1e-8):
        super(GIFBlock, self).__init__()
        self.radius = radius
        self.eps = eps

    def forward(self, x, guidance):
        mean_x = F.avg_pool2d(x, self.radius, stride=1, padding=self.radius//2)
        mean_guidance = F.avg_pool2d(guidance, self.radius, stride=1, padding=self.radius//2)
        corr = F.avg_pool2d(x * guidance, self.radius, stride=1, padding=self.radius//2)
        var_guidance = F.avg_pool2d(guidance ** 2, self.radius, stride=1, padding=self.radius//2) - mean_guidance ** 2
        a = (corr - mean_x * mean_guidance) / (var_guidance + self.eps)
        b = mean_x - a * mean_guidance
        return a * guidance + b

class ProximalDehazeNet(nn.Module):
    """Proximal Dehaze-Net architecture"""
    def __init__(self, num_stages=3):
        super(ProximalDehazeNet, self).__init__()
        self.num_stages = num_stages
        self.dnet = DNet()
        self.tnet = TNet()
        self.gif_block = GIFBlock()

    def forward(self, hazy_image):
       
        batch_size, _, height, width = hazy_image.shape

        U = torch.zeros(batch_size, 1, height, width, device=hazy_image.device)
        T = torch.ones(batch_size, 4, height, width, device=hazy_image.device)
        Q = hazy_image.clone()

        for _ in range(self.num_stages):
           # print("U shape: ",U.shape)
            #print("T shape: ",T.shape)
            U_hat = torch.cat([U, hazy_image], dim=1)
            U = self.dnet(U_hat)

            T_hat = torch.cat([T, hazy_image], dim=1)
            T = self.tnet(T_hat)
            T = self.gif_block(T, hazy_image)
			
            Q = (hazy_image - (1 - T)) / T

        return Q


 