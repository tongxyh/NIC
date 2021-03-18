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

matrix = np.zeros([330,4,2])
i = 0
for im_absolute in sorted(glob("/media/tong/Softwarefile/trashing/valid/test/*.png")):
	im_dir = im_absolute.split('/')[-1]
	
	#print im_dir
	for qp in range(35,39):
		#Read Image Shape		
		im = Image.open(im_absolute)
		w,h = im.size
		
		nums += w*h*3

		bpg_file = os.path.getsize(QP_DIR+"/coded_test_"+str(qp)+"/"+im_dir[:-4]+".bin")		
		mse_in_all = psnr.eval_mse(zip([im_absolute],[QP_DIR+"/test_"+str(qp)+"/"+im_dir[:-4]+".png"]))

		matrix[i,qp-35,0] = bpg_file
		matrix[i,qp-35,1] = mse_in_all

		print i,qp
	i += 1
print nums
print matrix
np.save("rd.npy",matrix)

