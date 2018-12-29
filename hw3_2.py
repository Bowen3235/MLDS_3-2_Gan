import numpy as np
import csv
from coder import*
from Gan_new import*
import cv2
from PIL import Image

epo = 100


class Maneger(object):
	def __init__(self):
		self.data_im = []
		self.data_tag = []
		self.data = []
		self.guess = []
		self.coder = coder()
		self.Gan = Gan( )
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
		k = open( "extra_data/tags.csv" ) 
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
			filename = 'extra_data/images/' + str( i ) + '.jpg'
			im = cv2.imread( filename )
			#self.data.append( [ im , extra_tag[i][1:] ]  )
			self.data_im.append(im)
			##saving parsed data
		#self.coder.read_data( self.data , 8000 )
		self.Gan.read_data( self.data_im , self.data_tag )

def main():
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
		elif( buf!="q -f" ): 
			print("illegal cmd \"{}\"".format(buf))




if __name__ == '__main__':
	main()