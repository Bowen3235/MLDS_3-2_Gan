import numpy as np
import csv
from coder import*
from Gan_new import*
import cv2
from PIL import Image
import torch

epo = 100


class Maneger(object):
	def __init__(self):
		self.data_im = []
		self.data_tag = []
		self.data = []
		self.guess = []
		self.coder = coder()
		self.Gan = Gan( )
		self.testgan = torch.load( 'generator.pt' )
	def train_coder(self , epo):
		self.coder.train( epo )
	def save_coder( self ,filename ):
		self.coder.save( filename )
	def read_coder( self ,filename ):
		self.coder.read( filename )
		self.Gan.set_coder( filename )
	def set_coder_lr(self , lrr ):
		lrr = ord(lrr) - ord('0')
		self.coder.setting_lr( lrr )
	def test_coder( self , test ):
		temp = [test[ : test.find('hair') - 1 ],test[ test.find('hair')+5 : test.find('eye')-1 ]]
		print (temp)
		tempp = []
		for j in temp[0]:
			tempp.append( ord(j) )
		if len(tempp)!=6:
			tempp = list(np.append(tempp , np.zeros( 6 - len(tempp) ) ))
		for j in temp[1]:
			tempp.append( ord(j) )
		if ( len( tempp )!=12 ):
			tempp = np.append( tempp , np.zeros( 12 - len(tempp) ) )
		print( "After parsing  : {}".format(tempp) )
		o1 , o2 = self.coder.test( tempp )
		print( "focusing layer : {}".format(o1) )
		print( "after decoder  : {}".format( ''.join( chr(i) for i in o2 ) ) )
	def train_gan(self):
		self.Gan.train()
	def test(self):
		condition = input( "condition?\n" )
		temp = [condition[ : condition.find('hair') - 1 ],condition[ condition.find('hair')+5 : condition.find('eye')-1 ]]
		#print( temp )
		tar = np.array([ np.zeros( 12 , dtype = float) , np.zeros( 12 ,dtype = float ) ])
		for idx ,(j) in enumerate( temp ):
			if j == 'orange':
				tar[idx][0] = 1.0
			elif j == 'white':
				tar[idx][1] = 1.0
			elif j =='aqua':
				tar[idx][2] = 1.0
			elif j == 'gray' or j=='yellow' :
				tar[idx][3] = 1.0
			elif j == 'green':
				tar[idx][4] = 1.0
			elif j == 'red':
				tar[idx][5] = 1.0
			elif j == 'purple':
				tar[idx][6] = 1.0
			elif j == 'pink':
				tar[idx][7] = 1.0
			elif j == 'blue':
				tar[idx][8] = 1.0
			elif j == 'black':
				tar[idx][9] = 1.0
			elif j == 'brown' :
				tar[idx][10] = 1.0
			elif j == 'blonde' :
				tar[idx][11] = 1.0
		tar = torch.FloatTensor(tar).view( 1,24 )
		noise = torch.FloatTensor(np.random.normal(0.5,0.16,( 1 , 128))) 
		pic = self.testgan( tar , noise ).view(64,64,3).detach()
		print( pic.size() )
		cv2.imshow( "image" , pic.numpy()/2+0.5 )
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		print( "Start making data base of all condition" )

		for i in range( 12 ):
			condition = ""
			if i == 0 :
				condition += 'orange'
			elif i == 1 :
				condition += 'white'
			elif i == 2 :
				condition += 'aqua'
			elif i == 3 :
				condition += 'gray'
			elif i == 4 :
				condition += 'green'
			elif i == 5 :
				condition += 'red'
			elif i == 6 :
				condition += 'purple'
			elif i == 7 :
				condition += 'pink'
			elif i == 8 :
				condition += 'blue'
			elif i == 9 :
				condition += 'black'
			elif i == 10 :
				condition += 'brown'
			elif i == 11 :
				condition += 'blonde'
			condition += "_hair_"
			condition_1 = condition

			for j in range( 12 ):
				condition = ""
				if j == 0 :
					condition += 'orange'
				elif j == 1 :
					condition += 'white'
				elif j == 2 :
					condition += 'aqua'
				elif j == 3 :
					condition += 'yellow'
				elif j == 4 :
					condition += 'green'
				elif j == 5 :
					condition += 'red'
				elif j == 6 :
					condition += 'purple'
				elif j == 7 :
					condition += 'pink'
				elif j == 8 :
					condition += 'blue'
				elif j == 9 :
					condition += 'black'
				elif j == 10 :
					condition += 'brown'
				elif j == 11 :
					condition += 'blonde'
				condition += "_eye"
				tar = np.array([ np.zeros( 12 , dtype = float) , np.zeros( 12 ,dtype = float ) ])
				tar[0][i] = tar[1][j] = 1
				tar = torch.FloatTensor(tar).view( 1,24 )
				noise = torch.FloatTensor(np.random.normal(0.5,0.16,( 1 , 128))) 
				pic = (self.testgan( tar , noise ).view(64,64,3).detach()/2+0.5)*255
				os.makedirs( os.path.dirname( "test_pic_1103/" ), exist_ok=True )
				cv2.imwrite( "test_pic_1103/"+condition_1+condition+".jpg" , pic.numpy() )
				





	def read_data(self):
		#data_tag = open( "AnimeDataset/tags_clean.csv" ) 
		#orin_data_tag = list(csv.reader( data_tag ))
		'''
		for i in orin_data_tag:
			temp = i[1]
			temp = temp.split('\t')
			tempp = [[],[]]
			for j in temp:
				hair_index = j.find( ' hair' )
				eye_index  = j.find( ' eye' )
				if( hair_index!=-1 ):
					tempp.append( j[:hair_index] )
				if( eye_index!=-1 ):
					tempp.append( j[:eye_index] )
			i[1:] = tempp
			print( i )
		'''
		k = open( "small_data/tags.csv" ) 
		extra_tag = list(csv.reader( k ))
		k.close()
		'''
		for i in extra_tag:
			temp = [i[1][ : i[1].find('hair') - 1 ],i[1][ i[1].find('hair')+5 : i[1].find('eye')-1 ]]
			tempp = []
			for j in temp[0]:
				tempp.append( ord(j) )
			if len(tempp)!=6:
				tempp = list(np.append(tempp , np.zeros( 6 - len(tempp) ) ))
			for j in temp[1]:
				tempp.append( ord(j) )
			if ( len( tempp )!=12 ):
				tempp = np.append( tempp , np.zeros( 12 - len(tempp) ) )
			i[1] = tempp
		'''
		for i in extra_tag:
			temp = [i[1][ : i[1].find('hair') - 1 ],i[1][ i[1].find('hair')+5 : i[1].find('eye')-1 ]]
			#print( temp )
			tar = np.array([ np.zeros( 12 , dtype = float) , np.zeros( 12 ,dtype = float ) ])
			for idx ,(j) in enumerate( temp ):
				if j == 'orange':
					tar[idx][0] = 1.0
				elif j == 'white':
					tar[idx][1] = 1.0
				elif j =='aqua':
					tar[idx][2] = 1.0
				elif j == 'gray' or j=='yellow' :
					tar[idx][3] = 1.0
				elif j == 'green':
					tar[idx][4] = 1.0
				elif j == 'red':
					tar[idx][5] = 1.0
				elif j == 'purple':
					tar[idx][6] = 1.0
				elif j == 'pink':
					tar[idx][7] = 1.0
				elif j == 'blue':
					tar[idx][8] = 1.0
				elif j == 'black':
					tar[idx][9] = 1.0
				elif j == 'brown' :
					tar[idx][10] = 1.0
				elif j == 'blonde' :
					tar[idx][11] = 1.0

			self.data_tag.append( np.reshape( tar , 24 ) )
		self.data_tag = torch.FloatTensor( self.data_tag )
		print(self.data_tag[0])

		for i in range( len(extra_tag) ):
			filename = 'small_data/images/' + str( i ) + '.jpg'
			im = cv2.imread( filename )
			#self.data.append( [ im , extra_tag[i][1:] ]  )
			self.data_im.append(im)
			##saving parsed data
		#self.coder.read_data( self.data , 8000 )
		self.Gan.read_data( self.data_im , self.data_tag )

def main():
	np.set_printoptions(precision=4)
	Mgr = Maneger()
	buf = []
	while( buf!="q -f" ):
		buf = input("cmd> ")
		if( buf.find("train coder")!=-1 ):
			if( len(Mgr.data)==0 ):
				print( "no data read in yet." )
			else:
				Mgr.train_coder( int(buf[buf.find('train coder')+len('train coder')+1:]) )
		elif( buf.find('save coder')!= -1 ):
			if( len(Mgr.data)==0 ):
				print( "no data read in yet." )
			else:
				Mgr.save_coder( buf[buf.find('save coder')+11:] )
		elif( buf.find('read coder')!= -1 ):
			Mgr.read_coder( buf[buf.find('read coder')+11:] )			
		elif( buf.find('set coder lr')!= -1  ):
			Mgr.set_coder_lr( buf[buf.find('set coder lr')+len('set coder lr')+1:] )
		elif( buf.find('test coder')!=-1 ):
			Mgr.test_coder( buf[buf.find('test coder')+len('test coder')+1:] )
		elif( buf=="read data" ):
			Mgr.read_data()
		elif( buf=="train gan" ):
			Mgr.train_gan()
		elif( buf == "test" ):
			Mgr.test()
		elif( buf!="q -f" ): 
			print("illegal cmd \"{}\"".format(buf))




if __name__ == '__main__':
	main()
