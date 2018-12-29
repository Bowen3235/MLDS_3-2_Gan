import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms 
import torch.utils.data as tordata
import numpy as np
import random as rd
from torch.autograd import Variable


class coder( object ):
	def __init__( self ):
		self.encoder = encoder()
		self.decoder = decoder()
		self.embedding = embedding( self.encoder , self.decoder )
		self.data = [[],[]]
		self.optimizer = optim.Adam( self.embedding.parameters()  , lr = 1e-5 )
		self.criterion = nn.MSELoss(  )
	def read_data(self, data , size):
		print( "making coder DataLoader ... " )
		batch_size = size
		testing = []
		or_data = []
		tes_num = []
		for i in range(1000):
			tes_num.append( rd.randint( 1 , len(data) ) )
		for i in range(len(data)) :
			if i in tes_num:
				testing.append( data[i][1] )
			else:
				or_data.append( data[i][1] )

		testing = torch.from_numpy( np.array(testing) )
		or_data = torch.from_numpy( np.array(or_data) )

		loader = tordata.DataLoader( dataset = tordata.TensorDataset( or_data , or_data ),batch_size = batch_size,shuffle = True)
		test_loader = tordata.DataLoader(dataset = tordata.TensorDataset( testing , testing ),batch_size = batch_size,shuffle = True)

		self.data[0] = loader
		self.data[1] = test_loader 

	def train(self , epoches):
		train_loader = self.data[0]
		for epoch in range( epoches ):
			for batch_idx, ( data, target ) in enumerate( train_loader ):
				self.embedding.train()
				data, target = Variable( data ,requires_grad=True) , Variable( target ,requires_grad=True).float()
				self.optimizer.zero_grad()
				output = self.embedding( data )
				lost = self.criterion( output , target )
				lost.backward()
				self.optimizer.step()
				print( 'Train Epoch {} , batch_num :{}, Loss: {:.6f}'.format( epoch+1 , batch_idx , lost.data.item()/len(data) )  )
			self.embedding.eval()
			correct = 0
			test_loss = 0
			test_loader = self.data[1]
			for (data,target) in test_loader:
				output = self.embedding( data )
				target = target.float()
				test_loss += abs(self.criterion( target ,output ).data.item())
				pred = output.data
				for i in range(len(pred)):
					if np.abs( pred[i] - target.data[i] ).sum() < 1*len(pred[i]):
						correct += 1
			test_loss /= len(test_loader.dataset)
			print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
	def save( self , filename ):
		torch.save( self.embedding.get_en() , "coder_model/"+filename+"_encoder.pt" )
		torch.save( self.embedding.get_de() , "coder_model/"+filename+"_decoder.pt" )
		print( "succeed savine at coder_model/{}_encoder.pt".format(filename) )
	def read( self , filename ):
		self.encoder = torch.load("coder_model/"+filename+"_encoder.pt")
		self.decoder = torch.load("coder_model/"+filename+"_decoder.pt")
		self.embedding = embedding( self.encoder , self.decoder)
		print( "succeed read in"+"coder_model/"+filename+"_encoder.pt" )
	def setting_lr( self ,lrr ):
		self.optimizer = optim.Adam( self.embedding.parameters()  , lr = 10**(-1*lrr) )
	def test( self , test ):
		test = torch.from_numpy( np.array(test) )
		x = self.encoder( test )
		y = self.embedding( test )
		return x ,y

class embedding( nn.Module ):
	def __init__(self, modelA, modelB):
		super(embedding, self).__init__()
		self.modelA = modelA
		self.modelB = modelB
	def forward(self,x):
		x = self.modelA(x)
		x = self.modelB(x)
		return x
	def get_en(self):
		return self.modelA
	def get_de(self):
		return self.modelB

class encoder( nn.Module ):
	def __init__(self):
		super( encoder , self).__init__()
		self.input = nn.Linear( 12 , 24 )
		self.ri = nn.ReLU()

		self.fc1 = nn.Linear( 24 , 48 )
		self.r1 = nn.ReLU()

		self.fc2 = nn.Linear( 48 , 119 )
	def forward(self,x):
		x = self.ri(self.input( (x.float()) ) )
		x = self.r1(self.fc1((x)))
		result = (self.fc2((x)))
		return result

class decoder( nn.Module ):
	def __init__(self):
		super( decoder , self).__init__()
		self.input = nn.Linear( 119 , 48 )
		self.ri = nn.ReLU()


		self.fc2 = nn.Linear( 48 , 24 )
		self.r2 = nn.ReLU()

		self.fc3 = nn.Linear( 24 , 12 )
		
	def forward(self,x):
		x = self.ri(self.input( (x.float()) ) )
		x = self.r2(self.fc2((x)))
		result = (self.fc3((x)))
		return result
