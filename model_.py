import torch
from torch import nn
import torch.nn.functional as F
from src.config import conf
# [Test Needed] Spectral Normalization for Discriminator
# Spectral Normalization on Conv Start
#123


class ConvSNRelu(nn.Module):
    # https://blog.csdn.net/StreamRock/article/details/83590347
    # https://blog.csdn.net/mahui6144/article/details/124577710?spm=1001.2101.3001.6650.8&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-8-124577710-blog-83590347.235%5Ev27%5Epc_relevant_default_base1&utm_relevant_index=11
    def __init__(self, in_c, out_c, ks=5, pad=2, s=2, spec=False):
        super(ConvSNRelu, self).__init__()
        conv_2d = nn.Conv2d(in_c, out_c, kernel_size=ks, padding=pad, stride=s)
        lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.module = nn.Sequential()
        if spec:
            self.module.add_module("conv2d_1", nn.utils.spectral_norm(conv_2d))
            self.module.add_module("lrelu_1", lrelu)
        else:
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("lrelu_1", lrelu)
            
    def forward(self, x):
        return self.module(x)



class ConvINRelu(nn.Module):
    def __init__(self, in_c, out_c, ks=5, pad=2, s=2, isn=False):
        super(ConvINRelu, self).__init__()
        conv_2d = nn.Conv2d(in_c, out_c, kernel_size=ks, padding=pad, stride=s)
        lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.module = nn.Sequential()
        if isn:
            ins_norm = nn.InstanceNorm2d(out_c, affine=True)
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("ins_norm_1", ins_norm)
            self.module.add_module("lrelu_1", lrelu)
        else:
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("lrelu_1", lrelu)

    def forward(self, x):
        return self.module(x)





# deconv layer
class DeConvINRelu(nn.Module):
    def __init__(self, in_c, out_c, ks=6, pad=2, s=2, isn=False):
        super(DeConvINRelu, self).__init__()
        conv_2d = nn.ConvTranspose2d(in_c, out_c, kernel_size=ks, padding=pad, stride=s)
        lrelu = nn.ReLU(inplace=False)  
        self.module = nn.Sequential()
        if isn:
            ins_norm = nn.InstanceNorm2d(out_c, affine=True)
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("batchnorm_1", ins_norm)
            self.module.add_module("lrelu_1", lrelu)
        else:
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("lrelu_1", lrelu)

    def forward(self, x):
        return self.module(x)


class DeConvINTanh(nn.Module):
    def __init__(self, in_c, out_c, ks=6, pad=2, s=2, isn=False):
        super(DeConvINTanh, self).__init__()
        conv_2d = nn.ConvTranspose2d(in_c, out_c, kernel_size=ks, padding=pad, stride=s)
        tanh = nn.Tanh()  
        self.module = nn.Sequential()
        if isn:
            ins_norm = nn.InstanceNorm2d(out_c, affine=True)
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("batchnorm_1", ins_norm)
            self.module.add_module("tanh_1", tanh)
        else:
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("tanh_1", tanh)

    def forward(self, x):
        return self.module(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        conv1 = ConvINRelu(1, 64,ks=3,pad=1, isn=True)
        conv2 = ConvINRelu(64, 128,ks=3,pad=1, isn=True)
        conv3 = ConvINRelu(128, 256,ks=3,pad=1, isn=True)
        conv4 = ConvINRelu(256, 512,ks=3,pad=1, isn=True)
        conv5 = ConvINRelu(512, 512,ks=3,pad=1, isn=True)
        conv6 = ConvINRelu(512, 512,ks=3,pad=1, isn=False)
        
        self.module = nn.ModuleList([conv1, conv2, conv3, conv4, conv5, conv6])

    def forward(self, x):
        out = []
        for mod in self.module:
            x = mod(x)
            out.append(x)
        return out

    
class DenseLayer(nn.Module):
    def __init__(self, in_c, out_c,bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        self.module = nn.Sequential()
        self.module.add_module("conv2d_1", nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, stride=1, bias=False))
        self.module.add_module("conv2d_1_", nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=1, bias=False))
        self.module.add_module("batch_normal_1",nn.BatchNorm2d(out_c))

        
        self.module.add_module("conv2d_2", nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1, bias=False))
        self.module.add_module("batch_normal_2",nn.BatchNorm2d(out_c))

       
        self.drop_rate = drop_rate
        self.drop_out = nn.Dropout(p=self.drop_rate)
        '''
        self.module.add_module("conv2d_1", nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1))
        self.module.add_module("batch_normal_1",nn.BatchNorm2d(out_c))
        self.module.add_module("relu_1", nn.ReLU(inplace=False))
        self.module.add_module("conv2d_2", nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1))
        self.module.add_module("batch_normal_2",nn.BatchNorm2d(out_c))
        '''

    def forward(self, x):
        out = self.module(x)
        if self.drop_rate > 0:
            out = self.drop_out(out)
        return torch.cat([x, out], 1)  

class DenseBlock(nn.Module):
    def __init__(self, in_c, out_c,num_layer, bn_size, drop_rate):
        super(DenseBlock, self).__init__()
        self.module = nn.Sequential()
        for i in range(num_layer):
            layer = DenseLayer(in_c+i*out_c, out_c, bn_size,drop_rate)
            self.module.add_module("denselayer%d" % (i+1,), layer)
        self.module.add_module("norm", nn.BatchNorm2d(in_c+num_layer*out_c))
        self.module.add_module("conv", nn.Conv2d(in_c+num_layer*out_c, out_c,kernel_size=1, stride=1, bias=False))
        
        self.tanhs = nn.Tanh()
        
        self.module_2 = nn.Sequential()
        self.module.add_module("norm", nn.BatchNorm2d(in_c+out_c))
        self.module_2.add_module("conv", nn.Conv2d(in_c+out_c, out_c,kernel_size=1, stride=1, bias=False))
    def forward(self, x):
        out = self.module(x)
        out = torch.cat([x, out], 1)
        out = self.module_2(out)
        out = self.tanhs(out)
        return out

        
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResBlock, self).__init__()
        #self.conv1 = ConvINRelu(in_c, out_c, ks=3, pad=1, s=1, isn=True)
        #self.conv2 = ConvINRelu(out_c, out_c, ks=3, pad=1, s=1, isn=True)
        self.module = nn.Sequential()
        self.module.add_module("conv2d_1", nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1, bias=False))
        self.module.add_module("batch_normal_1",nn.BatchNorm2d(out_c))
        self.module.add_module("relu_1", nn.ReLU(inplace=False))
        self.module.add_module("conv2d_2", nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1, bias=False))
        self.module.add_module("batch_normal_2",nn.BatchNorm2d(out_c))
        self.relu_out = nn.ReLU(inplace=False)
        
        
    def forward(self, x):
        out = self.module(x)
        out = out + x
        out = self.relu_out(out)
        return out
    
class ResBlock_256(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResBlock_256, self).__init__()
        #self.conv1 = ConvINRelu(in_c, out_c, ks=3, pad=1, s=1, isn=True)
        #self.conv2 = ConvINRelu(out_c, out_c, ks=3, pad=1, s=1, isn=True)
        self.module = nn.Sequential()
        self.module.add_module("batch_normal_1",nn.BatchNorm2d(in_c))
        self.module.add_module("conv2d_1", nn.Conv2d(in_c, out_c*4, kernel_size=1, padding=0, stride=1))
        
        self.module.add_module("batch_normal_2",nn.BatchNorm2d(out_c*4))
        self.module.add_module("conv2d_2", nn.Conv2d(out_c*4, out_c*4, kernel_size=3, padding=1, stride=1))
        
        
        self.module.add_module("batch_normal_3",nn.BatchNorm2d(out_c*4))
        self.module.add_module("conv2d_3", nn.Conv2d(out_c*4, out_c, kernel_size=1, padding=0, stride=1))
        
        self.relu_out = nn.ReLU(inplace=False)
        
    def forward(self, x):
        out = self.module(x)
        #x = self.module_2(x)
        out = out + x
        out = self.relu_out(out)
        return out
    

class ResBlock_type2(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResBlock_type2, self).__init__()
        #self.conv1 = ConvINRelu(in_c, out_c, ks=3, pad=1, s=1, isn=True)
        #self.conv2 = ConvINRelu(out_c, out_c, ks=3, pad=1, s=1, isn=True)
        self.module = nn.Sequential()
        self.module.add_module("batch_normal_1",nn.BatchNorm2d(in_c))
        self.module.add_module("relu_1", nn.ReLU(inplace=False))
        self.module.add_module("conv2d_1", nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1))
        self.module.add_module("drop",nn.Dropout(p=0.1))
        self.module.add_module("batch_normal_2",nn.BatchNorm2d(out_c))
        self.module.add_module("relu_2", nn.ReLU(inplace=False))
        self.module.add_module("conv2d_2", nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1))
        
      
        
    def forward(self, x):
        out = self.module(x)
        out = out + x
        return out  
    
class Fused_MBConv(nn.Module):
    def __init__(self, in_c, out_c, sc):
        super(Fused_MBConv, self).__init__()
        #self.conv1 = ConvINRelu(in_c, out_c, ks=3, pad=1, s=1, isn=True)
        #self.conv2 = ConvINRelu(out_c, out_c, ks=3, pad=1, s=1, isn=True)
        self.module = nn.Sequential()
        self.module.add_module("conv2d_1", nn.Conv2d(in_c, sc*out_c, kernel_size=3, padding=1, stride=1))
        self.module.add_module("batch_normal_1",nn.BatchNorm2d(sc*out_c))
        
        self.module.add_module("conv2d_2", nn.Conv2d(sc*out_c, out_c, kernel_size=1, padding=0, stride=1))
        self.module.add_module("batch_normal_2",nn.BatchNorm2d(out_c))
        
        self.tanhs = nn.Tanh()
        
        
      
        
    def forward(self, x):
        out = self.module(x)
        out = out + x
        out = self.tanhs(out)
        return out
       
    

class WNet(nn.Module):
    def __init__(self, M=5):
        super(WNet, self).__init__()
        self.left = Encoder()
        self.right = Encoder()
        
        #self.left_ResBlock1 = nn.Sequential(*[Fused_MBConv(64, 64, 4) for i in range(M - 4)])
        #self.left_ResBlock2 = nn.Sequential(*[Fused_MBConv(128, 128, 4) for i in range(M - 2)])
        #self.left_ResBlock3 = nn.Sequential(*[Fused_MBConv(256, 256, 2) for i in range(M)])
        #self.right_ResBlock3 = nn.Sequential(*[Fused_MBConv(256, 256, 2) for i in range(M)])
        
        
        self.left_DenseBlock1 = DenseBlock(64, 64, M - 4, 1, 0.1)
        self.left_DenseBlock2 = DenseBlock(128, 128, M - 2, 1, 0.1)
        self.left_DenseBlock3 = DenseBlock(256, 256, M, 1, 0.1)
        self.righ_DenseBlock3 = DenseBlock(256, 256, M, 1, 0.1)
        
        #decoder
        self.deconv1 = DeConvINRelu(1024, 512, isn=True)
        self.deconv2 = DeConvINRelu(512 * 3, 512, isn=True)
        self.deconv3 = DeConvINRelu(512 * 3, 256, isn=True)
        self.deconv4 = DeConvINRelu(256 * 3, 128, isn=True)
        self.deconv5 = DeConvINRelu(128 * 2, 64, isn=True)
        self.deconv6 = DeConvINRelu(64 * 2, 1, isn=False)
        self.tanhs = nn.Tanh()
        #self.drop_out = nn.Dropout(p=0.5) # comment this
        self.drop_out = nn.Dropout(p=0.5) # uncomment this

    def forward(self, lx, rx):
        left_out = self.left(lx)
        righ_out = self.right(rx)

        # 字形
        #left_out_0 = self.left_ResBlock1(left_out[0])
        #left_out_1 = self.left_ResBlock2(left_out[1])
        #left_out_2 = self.left_ResBlock3(left_out[2])
        left_out_0 = self.left_DenseBlock1(left_out[0])
        left_out_1 = self.left_DenseBlock2(left_out[1])
        left_out_2 = self.left_DenseBlock3(left_out[2])
        left_out_3 = left_out[3]
        left_out_4 = left_out[4]
        left_out_5 = left_out[5]


        # 風格
        #righ_out_2 = self.right_ResBlock3(righ_out[2])
        righ_out_2 = self.righ_DenseBlock3(righ_out[2])
        righ_out_3 = righ_out[3]
        righ_out_4 = righ_out[4]
        righ_out_5 = righ_out[5]


        # 合併

        de_0 = self.deconv1(torch.cat([left_out_5, righ_out_5], dim=1))
        de_0 = self.drop_out(de_0) # comment this
        
        de_1 = self.deconv2(torch.cat([left_out_4, de_0, righ_out_4], dim=1))
        de_1 = self.drop_out(de_1) # comment this
        
        de_2 = self.deconv3(torch.cat([left_out_3, de_1, righ_out_3], dim=1))
        de_2 = self.drop_out(de_2) # comment this

        de_3 = self.deconv4(torch.cat([left_out_2, de_2, righ_out_2], dim=1))
        de_4 = self.deconv5(torch.cat([left_out_1, de_3], dim=1))
        de_5 = self.deconv6(torch.cat([left_out_0, de_4], dim=1))
        de_5 = self.tanhs(de_5)
        return de_5, left_out_5, righ_out_5

class Discriminator(nn.Module):
    def __init__(self, num_fonts=21, num_characters=conf.num_chars+1):
        super(Discriminator, self).__init__()
        
        
        self.conv1 = ConvSNRelu(3, 64,  ks=5, pad=2, s=2, spec=False)
        self.conv2 = ConvSNRelu(64, 64 * 2,  ks=5, pad=2, s=2, spec=True)
        self.conv3 = ConvSNRelu(64 * 2, 64 * 4,  ks=5, pad=2, s=2, spec=True)
        self.conv4 = ConvSNRelu(64 * 4, 64 * 8,  ks=5, pad=2, s=1, spec=True)
        self.conv5 = ConvSNRelu(64 * 8, 64 * 16,  ks=5, pad=2, s=1, spec=True)
        #self.conv6 = ConvSNLRelu(64 * 16, 64 * 32, 8, 8, ks=5, pad=2, s=1, spec=True)
        '''
        self.conv1 = ConvSNRelu(3, 32,  ks=5, pad=2, s=2, spec=False)
        self.conv2 = ConvSNRelu(32, 32 * 2,  ks=5, pad=2, s=2, spec=True)
        self.conv3 = ConvSNRelu(32 * 2, 32 * 4,  ks=5, pad=2, s=2, spec=True)
        self.conv4 = ConvSNRelu(32 * 4, 32 * 8,  ks=5, pad=2, s=1, spec=True)
        self.conv5 = ConvSNRelu(32 * 8, 32 * 16,  ks=5, pad=2, s=1, spec=True)
        self.conv5 = ConvSNRelu(32 * 16, 32 * 32,  ks=5, pad=2, s=1, spec=True)
        '''
        
        self.fc1 = nn.Linear(1024 * 8 * 8, 1)
        self.fc2 = nn.Linear(1024 * 8 * 8, num_fonts)
        self.fc3 = nn.Linear(1024 * 8 * 8, num_characters)
        
    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        features = []
        x = self.conv1(x)
        features.append(x)
        x = self.conv2(x)
        features.append(x)
        x = self.conv3(x)
        features.append(x)
        x = self.conv4(x)
        features.append(x)
        x = self.conv5(x)
        features.append(x)
        #x = self.conv6(x)
        #features.append(x)

        x = x.view(-1, 1024 * 8 * 8)
        x1 = torch.sigmoid(self.fc1(x))  # real or fake
        x2 = F.softmax(self.fc2(x), dim=1)  # font category
        x3 = F.softmax(self.fc3(x), dim=1)  # char category
        x1 = x1.squeeze(-1)
        x2 = x2.squeeze(-1)
        x3 = x3.squeeze(-1)
        return x1, x2, x3, features

# 全連接分類
class ClSEncoderP(nn.Module):
    def __init__(self, num_characters=conf.num_chars+1):
        super(ClSEncoderP, self).__init__()
        self.fc = nn.Linear(512, num_characters)

    def forward(self, x):
        return self.fc(x)

# 全連接分類
class CLSEncoderS(nn.Module):
    def __init__(self, num_fonts=conf.num_fonts+1):
        super(CLSEncoderS, self).__init__()
        self.fc = nn.Linear(512, num_fonts)

    def forward(self, x):
        return self.fc(x)

#unuse --------------------------------------------------------------------------------------------------------------------------------

class ConvBNRelu(nn.Module):
    def __init__(self, in_c, out_c, ks=3, pad=1, s=1, bn=False):
        super(ConvBNRelu, self).__init__()
        conv_2d = nn.Conv2d(in_c, out_c, kernel_size=ks, padding=pad, stride=s, bias=False)
        relu = nn.ReLU(inplace=False)
        self.module = nn.Sequential()
        if bn:
            batch_norm = nn.BatchNorm2d(in_c)
            self.module.add_module("batchnorm_1", batch_norm)
            self.module.add_module("relu_1", relu)
            self.module.add_module("conv2d_1", conv_2d)
        else:
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("relu_1", relu)

    def forward(self, x):
        return self.module(x)

class ConvLNRelu(nn.Module):
    def __init__(self, in_c, out_c, out_h, out_w, ks=5, pad=2, s=2, ln=False):
        super(ConvLNRelu, self).__init__()
        conv_2d = nn.Conv2d(in_c, out_c, kernel_size=ks, padding=pad, stride=s)
        lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.module = nn.Sequential()
        if ln:
            layer_normal=nn.LayerNorm([out_c,out_h,out_w])
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("layer_normal_1", layer_normal)
            self.module.add_module("lrelu_1", lrelu)
        else:
            self.module.add_module("conv2d_1", conv_2d)
            self.module.add_module("lrelu_1", lrelu)

    def forward(self, x):
        return self.module(x)

    

if __name__ == "__main__":
    inp = torch.ones(1, 1, 64, 64)
    # wnet = WNet()
    # out = wnet(inp, inp)
    # print("output shape ", out.shape)
    d = Discriminator(80)
    out = d(inp, inp, inp)
    for o in out[3]:
        print(o.shape)


