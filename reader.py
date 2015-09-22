import numpy as np 
import cv2


class ReadImage:
	def __init__(self,imageName):
		self.imageName = imageName
		self.img = cv2.imread(self.imageName,0)
		self.unroll()

	def display(self):
		print self.img.shape
		cv2.imshow('1',self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def unroll(self):
		self.x,self.y = self.img.shape
		self.unrolled = self.img.reshape(self.x*self.y,1)
		print self.unrolled.shape
		#for i in xrange(0,50625):
			#print self.unrolled[i]

	def getData(self):
		return self.unrolled,self.x*self.y



'''
def main():
	IMG = ReadImage('MRI.jpg')
	#IMG.unroll()
	#IMG.display()

if __name__ == '__main__':
	main()
'''