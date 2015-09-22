import numpy as np
from reader import ReadImage
import cv2
import math
from scipy import ndimage



class FCM():
	def __init__(self,imageName,n_clusters,epsilon=0.05,max_iter=-1):
		self.m = 2
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.epsilon = epsilon

		read = ReadImage(imageName)
		self.X, self.numPixels = read.getData()
		self.X = self.X.astype(np.float)
		print "initial X:",self.X,self.X.shape

		self.U = []
		for i in range(self.numPixels):
			index = i % n_clusters
			l = [ 0 for j in range(n_clusters) ]
			l[index] = 1
			self.U.append(l)
		self.U = np.array(self.U).astype(np.float)
		self.U = self.U.reshape(self.numPixels,self.n_clusters)

		#self.U_new = np.zeros((self.numPixels,self.n_clusters))
		self.U_new = np.copy(self.U)
		#self.h = np.zeros((self.n_clusters,self.numPixels))
		
		self.C = []
		self.C = [1,85,255]
		#self.C = [0,255]
		#self.C = [150,200]
		self.C = np.array(self.C).astype(np.float)
		self.C = self.C.reshape(self.n_clusters,1)
		print "initial C:\n",self.C,self.C.shape

		Lambda = 2
		self.hesitation = np.zeros((self.numPixels,self.n_clusters))
		for i in range(self.numPixels):
			for j in range(self.n_clusters):
				self.hesitation[i][j] = 1.0 - self.U[i][j] - ( (1 - self.U[i][j]) / (1 + (Lambda * self.U[i][j]) ) )

		print self.hesitation



	def update_U(self):
		for i in range(self.numPixels):
			for j in range(self.n_clusters):
				sumation = 0
				for k in range(self.n_clusters):
					sumation += ( self.eucledian_dist(self.X[i],self.C[j]) / self.eucledian_dist(self.X[i],self.C[k]) ) ** (2 / (self.m-1) )
				self.U[i][j] = 1 / sumation

		print "U : ",self.U

	def update_C(self):
		for j in range(self.n_clusters):
			num_sum = 0
			den_sum = 0
			for i in range(self.numPixels):
				num_sum += np.dot((self.U[i][j] ** self.m),self.X[i])
				den_sum += self.U[i][j] ** self.m
			self.C[j] = np.divide(num_sum,den_sum)

		print "C : ",self.C

	def calculate_h(self):
		#self.h = np.zeros((self.n_clusters,self.numPixels))
		h = np.zeros((self.n_clusters,self.numPixels))
		u_rolled = np.zeros((self.numPixels ** 0.5,self.numPixels ** 0.5))
		kernel = np.ones((5,5))
		#kernel[2][2] = 4
		#kernel[2][1] = kernel[1][2] = kernel[3][2] = kernel[2][3] = 2
		#print self.U.transpose().shape,self.U.transpose()[0].shape
		for i in range(self.n_clusters):
			u_rolled = self.U.transpose()[i].reshape(self.numPixels ** 0.5,self.numPixels ** 0.5)
			print u_rolled.shape
			h_rolled = ndimage.convolve(u_rolled,kernel,mode='constant',cval=0.0)
			#self.h[i] = h_temp.reshape(1,self.numPixels)
			h[i] = h_rolled.reshape(1,self.numPixels)

		h = h.transpose()
		#self.h = self.h.transpose()
		print "\n",h,h.shape
		return h

	def compute_intuitionistic_U(self):
		Lambda = 0.5
		for i in range(self.numPixels):
			for j in range(self.n_clusters):
				self.hesitation[i][j] = 1.0 - self.U[i][j] - ( (1 - self.U[i][j]) / (1 + (Lambda * self.U[i][j]) ) )
		int_U = np.add(self.U,self.hesitation)
		self.U = np.copy(int_U)


	def computeNew_U(self):
		p = 1
		q = 2
		self.h = self.calculate_h()
		for j in range(self.numPixels):
			numer = 0.0
			denom = 0.0
			for i in range(self.n_clusters):
				numer = (self.U[j][i] ** p) * (self.h[j][i] ** q)
				for k in range(self.n_clusters):
					denom += (self.U[j][k] ** p) * (self.h[j][k] ** q)
				self.U_new[j][i] = numer/denom

		self.U = np.copy(self.U_new)

	def calculate_DB_score(self):
		sigma = np.zeros((3,1)).astype(np.float)
		count = np.zeros((3,1))
		result = np.zeros(shape=(self.numPixels,1))
		result = np.argmax(self.U, axis = 1)
		#self.Y = np.copy(self.X.astype(np.uint8))
		#for i in xrange(self.numPixels):
		#	self.Y[i] = self.C[self.result[i]].astype(np.int)

		for i in range(self.n_clusters):
			sigma[i] = 0
			for j in range(self.numPixels):
				if result[j] == i:
					count[i] += 1
					sigma[i] += self.eucledian_dist(self.C[i],self.X[j])
			sigma[i] = sigma[i]/count[i]

		#print result,sigma,count

		R_01 = (sigma[0] + sigma[1])/self.eucledian_dist(self.C[0],self.C[1])
		R_02 = (sigma[0] + sigma[2])/self.eucledian_dist(self.C[0],self.C[2])
		R_12 = (sigma[1] + sigma[2])/self.eucledian_dist(self.C[1],self.C[2])

		D0 = max(R_01,R_02)
		D1 = max(R_01,R_12)
		D2 = max(R_02,R_12)

		DB_score = (D0 + D1 + D2)/self.n_clusters
		print "DB_score: ",DB_score

	def calculate_D_score(self):
		sigma = np.zeros((3,1)).astype(np.float)
		count = np.zeros((3,1))
		result = np.zeros(shape=(self.numPixels,1))
		result = np.argmax(self.U, axis = 1)

		for i in range(self.n_clusters):
			sigma[i] = 0
			for j in range(self.numPixels):
				if result[j] == i:
					count[i] += 1
					sigma[i] += self.eucledian_dist(self.C[i],self.X[j])
			sigma[i] = sigma[i]/count[i]

		denom = max(sigma[0],sigma[1],sigma[2])
		#print denom

		d_01 = self.eucledian_dist(self.C[0],self.C[1])
		d_02 = self.eucledian_dist(self.C[0],self.C[2])
		d_12 = self.eucledian_dist(self.C[1],self.C[2])

		D_01 = d_01/denom
		D_02 = d_02/denom
		D_12 = d_12/denom

		D_score = min(D_01,D_02,D_12)
		print "D_score: ",D_score



	def calculate_scores(self):
		self.Vpc = 0.0
		sum_j = 0.0
		for j in range(self.numPixels):
			sum_i = 0.0
			for i in range(self.n_clusters):
				sum_i += self.U[j][i] ** 2
				#print "sum_i: ",sum_i
			sum_j += sum_i
			#print "sum_j: ",sum_j
		self.Vpc = sum_j/self.numPixels
		print "VPC: ",self.Vpc

		self.Vpe = 0.0
		sum_j = 0.0
		for j in range(self.numPixels):
			sum_i = 0
			for j in range(self.n_clusters):
				sum_i += self.U[j][i] * math.log(self.U[j][i])
			sum_j += sum_i
		self.Vpe = -1 * (sum_j/self.numPixels)
		print "VPE: ",self.Vpe

		self.Vxb = 0.0
		sum_j = 0.0
		for j in range(self.numPixels):
			sum_i = 0
			for i in range(self.n_clusters):
				sum_i += self.U[j][i] * (self.eucledian_dist(self.X[j],self.C[i]) ** 2)
			sum_j += sum_i
		numer = 1 * sum_j
		#dist = [ self.eucledian_dist(self.C[0],self.C[1]) ** 2, self.eucledian_dist(self.C[1],self.C[2]) ** 2 ,self.eucledian_dist(self.C[0],self.C[2]) ** 2]
		#denom = self.numPixels * min(dist)
		denom = self.numPixels * ( self.eucledian_dist(self.C[0],self.C[1]) ** 2 )
		self.Vxb = numer/denom
		print "VXB: ",self.Vxb

		self.calculate_DB_score()
		self.calculate_D_score()



	def eucledian_dist(self,a,b):
		return np.linalg.norm(a-b)

	def form_clusters(self):
		d = 100
		if self.max_iter != -1:
			for i in range(self.max_iter):
				print "loop : " , int(i)
				self.update_C()
				#temp = np.copy(self.U)
				temp = np.copy(self.U_new)
				self.update_U()
				self.compute_intuitionistic_U()
				self.computeNew_U()
				d = sum(abs(sum(self.U_new - temp)))
				print d
				self.segmentImage(i)
				if d < self.epsilon:
					break
		else:
			i = 0
			while d > self.epsilon:
				self.update_C()
				temp = np.copy(self.U)
				self.update_U()
				d = sum(abs(sum(self.U - temp)))
				print "loop : " , int(i)
				print d
				self.segmentImage(i)
				i += 1

	
	def segmentImage(self,image_count):
		self.result = np.zeros(shape=(self.numPixels,1))
		self.result = np.argmax(self.U, axis = 1)
		self.Y = np.copy(self.X.astype(np.uint8))
		print self.Y.shape
		#a = raw_input("press any key!")
		for i in xrange(self.numPixels):
			self.Y[i] = self.C[self.result[i]].astype(np.int)
			#print self.Y[i]
		self.Y = self.Y.reshape(self.numPixels ** 0.5,self.numPixels ** 0.5)
		#self.Y = self.Y.reshape(75,75)
		print self.Y,self.Y.shape,self.Y.dtype
		cv2.imwrite('output_sifcm/' + str(image_count) + '.jpg' , self.Y)
		image_count += 1
		#cv2.imshow('image',self.Y)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()



def main():
	#cluster = FCM('afcm1.jpg',2,0.01,300)
	#cluster = FCM('MRI.jpg',3,0.00005,100)
	cluster = FCM('t1_mri_scaled.jpg',3,0.05,100)
	cluster.form_clusters()
	cluster.calculate_scores()
	#cluster.calculate_h()
	#cluster.show_result()

if __name__ == '__main__':
	main()