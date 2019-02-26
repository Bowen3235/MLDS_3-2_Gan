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

class Gan(object):
    def __init__(self):
        self.gener = Generator()
        self.discr = Discriminator()
        self.ge_optimizer = optim.Adam( self.gener.parameters() , lr = 1e-4 ,betas=(0.5, 0.99)  )
        self.di_optimizer = optim.Adam( self.discr.parameters() , lr = 1e-4 ,betas=(0.5, 0.99)  )
        self.batch_size = 64
        self.data = []
        self.data_im = []
        self.data_tag =[]
        self.wrong_tag =[]
        self.encoder = 0
        self.g_loss = nn.BCELoss( reduction='elementwise_mean' )
        self.d_loss = nn.BCELoss( reduction='elementwise_mean' )
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
                    self.train_dis( i , idx , img/255*2 - 1 , tags , w_tags )

                    if idx % 100 ==0:
                        name =  "epo_" + str( i ) + "iterater" + str(idx)
                        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                        self.fake_img = self.fake_img.view( self.fake_img.size()[0] , 64 ,64 ,3 )
                        cv2.imshow( name,self.fake_img.data.numpy()[-1]/2+0.5 )
                        #print( self.fake_img.data.numpy()[-1] )
                        cv2.waitKey(1)
                        cv2.destroyAllWindows()
                        name = "new_info_image_text_showing score/" + "epo_" + str( i ) + "iterater" + str(idx) +"/"
                        os.makedirs( os.path.dirname( name ), exist_ok=True )
                        cv2.imwrite( name+"img.jpg" , (self.fake_img.data.numpy()[-1]/2+0.5)*255 )
                        
                        img = img.view( img.size()[0] ,  64 ,64 ,3 )

                        #print( img[-1].size() )
                        #print( (img[-1].numpy()/255*2 - 1 )/2+0.5  )
                        #cv2.imshow( name,( img[-1].numpy()/255*2 - 1 )/2+0.5 )
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        #cv2.imwrite( name+str(tags.numpy()[-1])+"img.jpg" , (( img[-1].numpy()/255*2 - 1 )/2+0.5)*255 )
                        
                        torch.save( self.gener ,name+"generator.pt" )
                        torch.save( self.discr ,name+"discriminator.pt" )
                    
                    '''
                    name = "new_image/" + "epo_" + str( i ) + "iterater" + str(idx) +"/"
                    os.makedirs( os.path.dirname( name ), exist_ok=True )
                    cv2.imwrite( name+"img.jpg" , self.fake_img.data.numpy()[-1]*255 )
                    torch.save( self.gener ,name+"generator.pt" )
                    torch.save( self.discr ,name+"discriminator.pt" )
                        '''
                    '''
                    name = "new_image/" + "epo_" + str( i ) + "iterater" + str(idx) +"/"
                    os.makedirs( os.path.dirname( name ), exist_ok=True )
                    tag_file = open( name+"tag.txt" , 'w' )
                    for j,(k) in enumerate( self.fake_img.data.numpy() ): 
                        im_name = name + str(j)+".jpg"
                        tag_stt_true = ''.join( chr(int(c)) for c in tags_w_str[j][:5] )
                        tag_stt_true +=" hair "
                        tag_stt_true += ''.join( chr(int(c)) for c in tags_w_str[j][6:] )
                        tag_stt_true +=" eyes"
                        #print( name )
                        cv2.imwrite( im_name , k )
                        tag_file.write(str(j)+"  "+tag_stt_true+"\n")
                    tag_file.close()





        
            noise = self.make_noise( self.batch_size )
            tags = []
            tags_str = []
            for j in np.random.randint( 0, len(self.data_tag) ,self.batch_size ):
                tags.append(self.data_tag[ j ])
                tags_str.append( self.tag_str_num[j] )
            tags = torch.FloatTensor( tags )
            im = self.gener( tags , noise ).view( ( self.batch_size ,64,64,3 ) )
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow( 'image',im.data.numpy()[0]/225 )
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            #print( tags_str )
            name = "new_image/" + "epo_" + str( i ) + "/"
            os.makedirs( os.path.dirname( name ), exist_ok=True )
            tag_file = open( name+"tag.txt" , 'w' )
            for j,(k) in enumerate( im.data.numpy() ): 
                im_name = name + str(j)+".jpg"
                tag_stt_true = ''.join( chr(int(c)) for c in tags_str[j][:5] )
                tag_stt_true +=" hair "
                tag_stt_true += ''.join( chr(int(c)) for c in tags_str[j][6:] )
                tag_stt_true +=" eyes"
                #print( name )
                cv2.imwrite( im_name , k )
                tag_file.write(str(j)+"  "+tag_stt_true+"\n")
            tag_file.close()
        torch.save( self.gener ,"generator.pt" )
        torch.save( self.discr ,"discriminator.pt" )
        print( "succeed savine at generator.pt" )
        '''
            



    def train_gen( self , epo , idx ,  noise , tags  ):
        self.gener.train()
        self.discr.train()
        print( "="*10+"train generator"+"="*10 )
        self.ge_optimizer.zero_grad()
        img = self.gener( tags , noise )
        self.fake_img = img.detach()
        #print( self.fake_img[0] )
        fake = self.discr( tags , img )
        valid = Variable(torch.FloatTensor( fake.size()[0] , 1).fill_(1.0), requires_grad=False)
        lost = self.g_loss( fake , valid )
        lost.backward()
        self.ge_optimizer.step()
        #print( fake.norm() )
        print( 'Train Epoch {} , batch_num :{}, Loss: {:.6f}'.format( epo+1  , idx+1 , lost.data.item() )  )
        '''
        self.fake_tag = torch.FloatTensor([])
        noise = self.make_noise( len(self.data_tag) )
        tags  = torch.FloatTensor( np.array(self.data_tag) )
        noise_Dataloader = tordata.DataLoader( dataset = tordata.TensorDataset( noise ,tags ),batch_size = self.batch_size,shuffle = True)
        for idx,(noise,tags) in enumerate(noise_Dataloader):
            self.ge_optimizer.zero_grad()
            #print( tags.size() , "," , noise.size() )
            noise = noise.view( ( noise.size()[0] , 1 , 100 ) )
            #print( "     "+"="*10+"generating fake image"+"="*10 )
            img = self.gener(  tags , noise )
            #print( "     "+"="*10+"end generating fake image"+"="*10 )
            if self.fake_img.size()[0] < len(self.data_tag)/3 :
                self.fake_img = torch.cat( (self.fake_img,img.detach()) , 0 )
                self.fake_tag = torch.cat( (self.fake_tag,tags) , 0 )
            #print( "     "+"="*10+"end adding image"+"="*10 )
            fake  = self.discr( tags , img )
            valid = Variable(torch.FloatTensor( fake.size()[0] , 1).fill_(1.0), requires_grad=False)
            lost  = self.g_loss( fake , valid )
            lost.backward(  )
            #print( "     "+"="*10+"do gradient descent"+"="*10 )
            self.ge_optimizer.step() 
            #print( "     "+"="*10+"end gradient descent"+"="*10 )
            print( 'Train Epoch {}-{} , batch_num :{}, Loss: {:.6f}'.format( epo+1 , mini_epo+1 , idx+1 , lost.data.item() )  )
        return lost.data.item()
        '''
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
        self.di_optimizer.zero_grad()
        i_eval = self.discr( w_tags , img )
        i_lost = self.d_loss(  i_eval , i_label )
        i_lost.backward()
        self.di_optimizer.step()
        #print( f_pre.norm() )
        np.set_printoptions(precision=4, suppress=True)
        print( 'label score {:.3f}  , fake img score {:.3f} '.format( np.mean(f_pre.detach().numpy()) , np.mean(i_eval.detach().numpy() )) )
        print( 'Train Epoch {} , batch_num :{}, Loss: {:.6f}'.format( epo+1  , idx+1 , lost.data.item() )  )

        '''
        
        for i in wrong_tags:
            print(i)
        
        #print( data_im.size() , fake_img.size() )
        start = np.random.randint( 0 , int( len(self.data_im)/3 * 2 - 3 ) , 2 )
        mini_size = int( len(self.data_im)/3 )
        im = torch.cat( 
                            ( self.data_im[ start[0] : start[0]+mini_size ]\
                            , self.data_im[ start[1] : start[1]+mini_size ]\
                            , self.fake_img ) , 0 \
                        )
        tags = torch.cat( 
                            ( torch.FloatTensor( self.data_tag[ start[0] : start[0]+mini_size ] ) \
                            , torch.FloatTensor( self.wrong_tag[:mini_size] ) \
                            , self.fake_tag ) , 0 )

        valid = Variable( torch.FloatTensor( mini_size , 1).fill_(1.0), requires_grad=False)
        fake = Variable( torch.FloatTensor( int(len( self.fake_img ) + mini_size ) , 1).fill_(0.0) , requires_grad=False)


        #print( im.size() , tags.size() )
        #print( fake.size() , valid.size() )

        DataLoader = tordata.DataLoader( dataset = tordata.TensorDataset( im , tags , torch.cat( ( valid ,fake ) , 0 ) ),batch_size = self.batch_size,shuffle = True)
        #print( "**==========end of processing DataLoader..." )
        for idx ,( img , tag , target ) in enumerate( DataLoader ):

            pre = self.discr( tag , img )
            self.di_optimizer.zero_grad()
            lost = self.d_loss( pre , target )
            lost.backward(  )
            self.di_optimizer.step()
            print( 'Train Epoch {}-{} , batch_num :{}, Loss: {:.6f}'.format( epo+1 , mini_epo+1 , idx+1 , lost.data.item() )  )
        return lost.data.item()
        '''
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
        self.to128     = nn.Linear( 24 + 128 ,128 )
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

                nn.Conv2d( 1024 , 1024 , 4 , 1, 0 ),
                #1024 , 1 ,1
        )
             
        self.last = nn.Linear( 1024+24 ,1 )

    def forward(self , input_text , pic):
        pic = self.main( pic )
        #print(pic.size())
        #print( input_text.size() )
        pic = pic.view( pic.size(0) , -1 )
        x  = torch.cat( (pic , input_text) ,1 ).view( input_text.size(0) , 1048 )
        x  = torch.sigmoid( self.last( x ).view( x.size(0) , 1 ) )
        return x




