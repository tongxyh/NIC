import os,shutil
import numpy as np
from glob import glob
from PIL import Image
import psnr

TARGET_DIR = "./encoded_files/"
REC_DIR = "./rec/"
QP_DIR = "/media/tong/Softwarefile/trashing/test"
filesize = 0

nums = 8311318521

qp_min = 35
qp_max = 38
qp_range = qp_max - qp_min + 1

matrix = np.load("rd.npy")
i = 0
grad = np.zeros([330,qp_range])

#grad[:,3] = 0
grad[:,qp_range-1] = 10000000000

for i in range(330):
	#print im_dir
	for qp in range(qp_min,qp_max):
		# grad = -delta(mse)/delta(bits) 
		grad[i,qp-qp_min] =  - 1.0*(matrix[i,(qp+1)-qp_min,1] - matrix[i,qp-qp_min,1])/(matrix[i,(qp+1)-qp_min,0] - matrix[i,qp-qp_min,0])
		#print grad[i,qp-35]

table = np.zeros([330])
grad_step  = grad[:,0]

table += qp_min

filesize = np.sum(matrix[:,0,0])
while filesize > 15749090 - 2917:

	#find image index
	index = np.argmin(grad_step)
	#print index,table[index]
	#print grad_step
	#update the filesize
	
	if filesize - matrix[index,int(table[index])-qp_min,0] + matrix[index,int(table[index]+1)-qp_min,0] < 15749090 - 2917:
		distance = 15749090 - 2917 - filesize
		distortion = 100000000.0
		for i in range(102):
			if table[i] < qp_max:
				step = matrix[index,int(table[i])-qp_min,0] - matrix[index,int(table[i]+1)-qp_min,0]
				if step >= distance and distortion > matrix[index,int(table[i]+1)-qp_min,0] - matrix[index,int(table[i])-qp_min,0]:
					distortion = matrix[index,int(table[i]+1)-qp_min,0] - matrix[index,int(table[i])-qp_min,0]
					index = i
	
	filesize = filesize - matrix[index,int(table[index])-qp_min,0] + matrix[index,int(table[index]+1)-qp_min,0]
	#update image state
	
		
	table[index] += 1

	#update grad state
	grad_step[index]  = grad[index,int(table[index])-qp_min]
	#print table

print ("QP Decision:\n",table)
print ("Filesize:",filesize + 901)

#table = np.array(table)
#np.save("rd.npy",table)

dirs = sorted(glob("/media/tong/Softwarefile/trashing/valid/test/*.png"))

dat = {}

for i in range(330):
	im_dir = dirs[i].split('/')[-1]
	#print im_dir
	dat[im_dir] = table[i]
 
	shutil.copy(QP_DIR+"/coded_test_"+str(int(table[i]))+"/"+im_dir[:-4]+".bin" , TARGET_DIR)
	shutil.copy(QP_DIR+"/test_"+str(int(table[i]))+"/"+im_dir[:-4]+".png" , REC_DIR)

np.save("qp",dat)

#dct = np.load("qp.npy").item()
#print dct["0067.png"]
print "Here"
image_file = glob("/media/tong/Softwarefile/trashing/valid/test/*.png")
image_rec = glob("./rec/*.png")
image_file = sorted(image_file)
image_rec = sorted(image_rec)
#print(image_file.split("/"))[-1]

result = psnr.evaluate(zip(image_file,image_rec))
print ("PSNR:",result)
