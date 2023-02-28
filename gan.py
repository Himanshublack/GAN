# Imports
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting the hyperparameters

batchSize = 64
imageSize = 64

# Creating the transformations

transform = transforms.Compose([transforms.Resize(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = dset.CIFAR10(root ='./data', download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the Generator(making the brain of the generator using reverse vectors as we are forming an image from vactors)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias = False),
            nn.Tanh()
            )
    def foreward(self, input):
        output = self.main(input)
        return output

# Creating the generator
netG = Generator()
netG.apply(weights_init)     

# Defining the Discriminator(Making the brain of discriminator to distingutish between the real and fake images)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace= True),
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
            )
    def foreward(self, input):
        output = self.main(input)
        return output.view(-1)
# Creating the Discriminator

netD = Discriminator()
netD.apply(weights_init)

# training the DCGAN
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

if __name__ == '__main__':
    for epoch in range(10):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real,_ = data
            target = Variable(torch.ones(input.size()[0]))
            output = netD(input)
            errD_real = criterion(output, target)
            
            noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
            fake = netG(noise)
            target = Variable(torch.zeros(input.size()[0]))
            output = netD(fake.detach())   
            errD_fake = criterion(output, target)
            
            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step() # step is used to update the weights of discriminator accordingly
            
            #updating the weights of the generator
            netG.zero_grad()
            target = Variable(torch.ones(input.size()[0]))
            output = netG(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()
            
            print('[%d/%d][%d/%d] Loss of D: 0.4%f Loss of G: 0.4%f' %(epoch, 10, i, len(dataloader), errD.data[0], errG.data[0]))
            if i % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png'%"./resultsgan", normalize = True)
                fake.netG(noise)
                vutils.save_image(fake.data, '%s/fake_samples%03d.png'%("./resultsgan", epoch), normalize = True)
       
           
        

            