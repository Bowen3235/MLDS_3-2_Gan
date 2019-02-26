import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms 
import torch.utils.data as tordata
import numpy as np
import random as rd
from torch.autograd import Variable
import cv2
import os
import itertools

class Gan(object):
    def __init__(self):
        self.gener = Generator()
        self.discr = Discriminator()
        self.ge_optimizer = optim.Adam( self.gener.parameters() , lr = 1e-2 ,betas=(0.5, 0.99)  )
        self.di_optimizer = optim.Adam( self.discr.parameters() , lr = 2*1e-4 ,betas=(0.5, 0.99)  )
        self.info_optimizer = optim.Adam( itertools.chain( self.gener.parameters() , self.discr.parameters()),\
             lr = 1e-4 , betas=( 0.5 , 0.99 ) )
        self.batch_size = 64
        self.data = []
        self.data_im = []
        self.data_tag =[]
        self.wrong_tag =[]
        self.encoder = 0
        self.g_loss = nn.BCELoss(  )
        self.d_loss = nn.BCELoss(  )
        self.info_loss = nn.CrossEntropyLoss()
        self.tag_str_num = []
        self.fake_img = []
        self.fake_tag = []
    def make_noise(self , num):
        noise = torch.FloatTensor(np.random.normal(0.5,0.16,( num , 128))) 
        return noise
    def train( self):
        #Gepo = input("Generator epo ? ")
        #Depo = input("Discriminator epo ?" )
        Gepo = 2
        Depo = 1
        total_epo = 10
        for i in range( total_epo ):
            noise = self.make_noise( self.data_tag.size()[0] )
            #print( noise[0] )
            #print( noise.size() , self.data_im.size() , self.data_tag.size() , self.wrong_tag.size() , self.tag_str_num.size() )
            The_Dataloader = tordata.DataLoader( dataset = tordata.TensorDataset( noise ,  self.data_im ,self.data_tag ,self.wrong_tag  ,self.tag_str_num ),batch_size = self.batch_size,shuffle = True)
            for idx ,( noise , img , tags , w_tags , tags_w_str ) in enumerate( The_Dataloader ):
                    #print( tags[0] )
                    #print( noise[0] )
                    self.train_gen( i , idx ,noise ,tags )
                    
                    self.fake_img = self.gener( tags ,self.make_noise( noise.size(0) ) ).detach()
                    self.train_dis( i , idx , img/255*2 - 1 , tags , w_tags )
                    
                    if idx % 100 ==0:
                        name =  "epo_" + str( i ) + "iterater" + str(idx)
                        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                        self.fake_img = self.fake_img.view( self.fake_img.size()[0] , 64 ,64 ,3 )
                        cv2.imshow( name,self.fake_img.data.numpy()[-1]/2+0.5 )
                        #print( self.fake_img.data.numpy()[-1] )
                        cv2.waitKey(1)
                        cv2.destroyAllWindows()
                        name = "new_info_image_text/" + "epo_" + str( i ) + "iterater" + str(idx) +"/"
                        os.makedirs( os.path.dirname( name ), exist_ok=True )
                        cv2.imwrite( name+"img.jpg" , (self.fake_img.data.numpy()[-1]/2+0.5)*255 )


                        torch.save( self.gener ,name+"generator.pt" )
                        torch.save( self.discr ,name+"discriminator.pt" )
            



    def train_gen( self , epo , idx ,  noise , tags  ):
        self.gener.train()
        self.discr.train()
        print( "="*10+"train generator"+"="*10 )
        self.ge_optimizer.zero_grad()
        img = self.gener( tags , noise )
        self.fake_img = img.detach()
        #print( self.fake_img[0] )
        fake , __ = self.discr( tags , img )
        valid = Variable(torch.FloatTensor( fake.size()[0] , 1).fill_(1.0), requires_grad=False)
        lost = self.g_loss( fake , valid )
        lost.backward()
        self.ge_optimizer.step()
        #print( fake.norm() )
        print( 'Train Epoch {} , batch_num :{}, Loss: {:.6f}'.format( epo+1  , idx+1 , lost.data.item() )  )
    def train_dis( self , epo , idx , img , tags , w_tags ):
        #self.gener.train()
        #self.discr.train()
        print( "="*10+"train discriminator"+"="*10 )
        fake_label =  torch.FloatTensor( np.random.uniform( 0.0 , 0.0 , len( self.fake_img ) ) ).view( 64 ,1 )
        i_label    =  torch.FloatTensor( np.random.uniform( 0.0 , 0.0 , img.size()[0] )).view( 64 ,1 ) 
        r_label    =  torch.FloatTensor( np.random.uniform( 1.0 , 1.0 , img.size()[0] )).view( 64 ,1 )
        self.di_optimizer.zero_grad()
        f_pre = self.discr( tags , self.fake_img )
        f_lost = self.d_loss( f_pre , fake_label )
        r_eval = self.discr( tags , img )
        r_lost = self.d_loss( r_eval , r_label )
        lost = f_lost + r_lost
        lost = lost/2
        lost.backward()
        self.di_optimizer.step()
        i_eval , __ = self.discr( w_tags , img )
        i_lost = self.d_loss(  i_eval , i_label )
        i_lost.backward()
        self.di_optimizer.step()
        #print( f_pre.norm() )
        self.info_optimizer.zero_grad()
        __ , info_eval = self.discr( tags , img )
        info_loss = self.info_loss( info_eval , tags )
        info_loss.backward()
        info_optimizer.step()

        print( 'Train Epoch {} , batch_num :{}, Loss: {:.6f}'.format( epo+1  , idx+1 , lost.data.item() )  )
    def set_coder( self , filename ):
        self.encoder = torch.load("coder_model/"+filename+"_encoder.pt")
    def read_data( self , in_data_im , in_data_tag ):
        print( "Making Gan Data..." )
        self.data_im = torch.FloatTensor( in_data_im ).view( len( in_data_im ) , 3 ,64 ,64 )
        print( self.data_im[0]/255*2-1 )
        #cv2.imshow( 'main',self.data_im[0].view( 64 ,64 ,3 ).numpy()/255 )
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        self.wrong_tag = torch.FloatTensor([])
        self.tag_str_num = torch.FloatTensor( np.array( in_data_tag ))
        self.data_tag = in_data_tag

        for i in range(int(len(self.data_tag))):
            self.wrong_tag = torch.cat(\
                ( self.wrong_tag, self.data_tag[np.random.randint( 0 , len(self.data_tag) )]  )\
                ,0 )
        self.wrong_tag = self.wrong_tag.view( self.data_im.size()[0] , 24 )
        #print( torch.tensor( self.data[0][0] ).size() )

        
        

class Generator(nn.Module):
    def __init__(self):
        super( Generator , self).__init__()
        self.taglay    = nn.Linear( 24 , 119 )
        self.to128     = nn.Linear( 119 + 128 ,128 )
        self.main = nn.Sequential(
                # [128, 1, 1] -> [-1, 1025, 4, 4]
                nn.ConvTranspose2d( 128 , 1024, 4, 1, 0),

                nn.ConvTranspose2d(1024, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),

                # [-1, 256, 8, 8]
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),

                # [-1, 128, 16, 16]
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),

                # [-1, 3, 32, 32]
                nn.ConvTranspose2d(128, 3, 4, 2, 1),
                nn.Tanh()
            )

    def forward(self ,text_input ,noise_input):
        text_input = self.taglay( text_input )
        x = torch.cat( (text_input , noise_input), 1 )

        x = self.to128( x )

        x = x.view( \
            x.size()[0]  , 128 , 1 , 1 )

        x = self.main( x )

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super( Discriminator , self).__init__()
        self.main = nn.Sequential(
                # [-1, 3, 32, 32] -> [-1, 128, 16, 16]
                nn.Conv2d(3, 128, 4, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),

                # [-1, 256, 8, 8]
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),

                # [-1, 512, 4, 4]
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv2d(512, 1024, 4, 2, 1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),
        )
        self.taglay = nn.Linear( 24 , 32 )
              #1024 , 4 , 4
        self.last = nn.Conv2d(1024, 1 , 5, 1, 0)
        self.lable = nn.Conv2d( 1024 , 24 , 5 , 1, 0 )
           

    def forward(self , input_text , pic):
        pic = self.main( pic )
        pic = pic.view( pic.size(0) , -1 )
        input_text = self.taglay( input_text )
        input_text = input_text.view( input_text.size(0) , -1)
        input_text = input_text.repeat( ( 1 , 72*4 ) )
        x_1  = torch.cat( (pic , input_text) ,1 ).view( input_text.size(0) , 1024 , 5 , 5 )
        x  = torch.sigmoid( self.last( x_1 ).view( x_1.size(0) , 1 ) )
        lable = torch.softmax( self.lable( x_1 ).view( x_1.size(0) , 1 , 24 ) , dim = 0 )
        return x , lable




