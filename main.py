import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal
import scipy as sp
from scipy.ndimage import gaussian_filter,laplace, gaussian_laplace
import numpy as np
import cv2
from skimage import exposure,color
import math


def gaussian_kernel(sigma_pixel=1):
	radius = 3 * sigma_pixel #gaussian function spans 3 sigmas
	d = 2* radius 
	diameter = int(math.ceil(d))
	if(diameter%2) == 0:
		diameter = diameter + 1.
	ax = np.arange(-diameter // 2 + 1., diameter // 2 + 1.)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-(xx**2 + yy**2) / (diameter))
	return kernel/kernel.sum()



kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
file = raw_input("What is the file name?: ")
#img = cv2.imread(file,0)
im = plt.imread(file)
imgo = color.rgb2gray(im)
#cv2.imshow('image',img)
#cv2.waitKey(1000)
#cv2.waitKey(1)
#cv2.destroyAllWindows()
#cv2.waitKey(1)

for i in range(0,5):
	img = imgo
	sigma = 3*pow(2,i)
	array = gaussian_kernel(sigma)
	plt.imshow(array)
	plt.show()
	#print(np.matrix(array))
	LoG = sp.signal.convolve2d(array,kernel,mode='same',boundary ='symm')
	LoG = LoG/LoG.sum()
	#img_g = sp.signal.convolve2d(img, array, mode='same', boundary = 'symm')
	#ret,th1 = cv2.threshold(img_g,np.mean(img_g),255,cv2.THRESH_BINARY)
	img_LoG = sp.signal.convolve2d(img, LoG, mode='same', boundary = 'symm')
	#print("max = " + str(img_LoG.max()))
	#print("min = " + str(img_LoG.min()))
	#plt.imshow(img_LoG,cmap='gray')
	#plt.show()
	#cv2.imwrite("blur" + str(i)+ ".jpg", img_g)
	zero_crossings = np.where(np.diff(np.sign(img_LoG)))[0]

	minLoG = cv2.morphologyEx(img_LoG, cv2.MORPH_ERODE, np.ones((3,3)))
	maxLoG = cv2.morphologyEx(img_LoG, cv2.MORPH_DILATE, np.ones((3,3)))
	zeroCross = np.logical_or(np.logical_and(minLoG < 0,  img_LoG > 0), np.logical_and(maxLoG > 0, img_LoG < 0))
	#zeroCross= cv2.normalize(zeroCross,  zeroCross, 0, 255, cv2.NORM_MINMAX)
	zeroCross = np.multiply(zeroCross,255)
	#plt.imshow(zeroCross,cmap='gray')
	#plt.show()
	#img_LoG = cv2.normalize(img_LoG,  img_LoG, 0, 255, cv2.NORM_MINMAX)
	#cv2.imwrite("LoG" + str(i)+ ".jpg", img_LoG)
	cv2.imwrite("edge" + str(i)+ ".jpg", zeroCross)


	#img_edge = zero_crossing(img_LoG)
	#img_edge= cv2.normalize(img_edge,  img_edge, 0, 255, cv2.NORM_MINMAX)
	#cv2.imwrite("Edge" + str(i)+ ".jpg", img_edge)
	#print("\n")
	#img_gaussian = cv2.GaussianBlur(img,(diameter,diameter),sigma)
	#ret,th1 = cv2.threshold(img_gaussian,np.mean(img_gaussian),255,cv2.THRESH_BINARY)
	#th1 =apply_brightness_contrast(img_gaussian,0,64)
	#th1 = img_gaussian
	#LoG = laplace(th1)
	#cv2.imwrite("laplace" + str(i) +".jpg", LoG)
	#LoGl = cv2.Laplacian(img_gaussian,cv2.CV_16S, diameter)
	#LoG = cv2.convertScaleAbs(LoGl)
	#cv2.imshow('image',img)
	#cv2.imshow('image',th1)



	#cv2.waitKey(500)
	#cv2.waitKey(1)
	#cv2.destroyAllWindows()
	#cv2.waitKey(1)