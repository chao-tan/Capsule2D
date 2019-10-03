import argparse
from model import CapsBasicNet as network
from utils import MMSE as MMSE
import datasets
import torch
import os
import torch.backends.cudnn as cudnn
import time
import sys
import datetime
import cv2
import numpy as np


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=100, help='number of total epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=80, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=64, help='size of image height')
parser.add_argument('--img_width', type=int, default=64, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=1, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=0, help='number of residual blocks in generator')
parser.add_argument('--gpu-ids',type=str,default="0")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


os.makedirs('images/', exist_ok=True)
os.makedirs('saved_models/', exist_ok=True)
os.makedirs('logs/', exist_ok=True)

model = network()
criterion = MMSE()
optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr, betas=(opt.b1, opt.b2))
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

if cuda:
    model = model.cuda()
    cudnn.benchmark = True

if opt.epoch != 0:
    model.load_state_dict(torch.load('saved_models/%d.pth' % opt.epoch))



mnist = datasets.MovingMnist_Generation(digtnum=2,width=64,height=64,seq_length=1)

def sample_images(batches_done):
    model.eval()
    img,_ = mnist.next_batch(4)
    imgs = torch.from_numpy(img).float().type(Tensor)
    imgs = torch.cat((imgs,torch.ones(1,1,1,64,64).type(Tensor)*255),0)
    imgs = imgs.squeeze(1)
    recons_img,_ = model.forward(imgs)
    recons_img = recons_img.cpu().data.numpy()


    img_show_a = np.concatenate((img[0,0,0,:,:],img[1,0,0,:,:],img[2,0,0,:,:],img[3,0,0,:,:],np.ones(shape=(64,64))*255),axis=1)
    img_show_b = np.concatenate((recons_img[0,0,:,:],recons_img[1,0,:,:],recons_img[2,0,:,:],recons_img[3,0,:,:],recons_img[4,0,:,:]),axis=1)
    img_show = np.concatenate((img_show_a,img_show_b),axis=0)
    cv2.imwrite("images/"+str(batches_done)+".jpg",img_show)




def evaluate(cur_model,batch_no):
    ELOSS = 0.0
    cur_model.eval()
    for i in range(int(1000/opt.batch_size)):
        img, _ = mnist.next_batch(opt.batch_size)
        img = torch.from_numpy(img).type(Tensor).squeeze(1)

        gen_img,_ = model.forward(img)
        loss = criterion(img, gen_img)
        ELOSS = ELOSS + loss.item()

        sys.stdout.write("\r>>>>>>>>>[EVALUATE %d] [Batch %d/%d] [EVALoss: %f]" %
                         (batch_no,i+1, int(1000/opt.batch_size),
                          loss.item()))

    ELOSS = ELOSS /int(1000/opt.batch_size)
    file = open("logs/eval.txt","a+")
    file.write(str(ELOSS)+"\n")





#----------
#  Training
#----------

prev_time = time.time()

LOSS = 0.0
for epoch in range(opt.epoch, opt.n_epochs):
    for i in range(int(5000/opt.batch_size)):
        model.train()
        img,_ = mnist.next_batch(opt.batch_size)
        img = torch.from_numpy(img).type(Tensor).squeeze(1)

        optimizer.zero_grad()
        img_gen,_ = model.forward(img)
        loss = criterion(img,img_gen)
        loss.backward()
        optimizer.step()

        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = epoch * int(5000/opt.batch_size) + i
        batches_left = opt.n_epochs * int(5000/opt.batch_size) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, int(5000/opt.batch_size),
                                                        loss.item(),time_left))

        file = open("logs/train.txt", "a+")
        file.write(str(loss.item())+"\n")

        LOSS = LOSS + loss.item()

        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    print(("\r[Epoch %d/%d] [AVELoss: %f]" %(epoch, opt.n_epochs,
                                                 LOSS/int(5000/opt.batch_size))))

    evaluate(model,epoch+1)
    LOSS = 0.0
    lr_scheduler.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        torch.save(model.state_dict(), 'saved_models/%d.pth' % epoch)