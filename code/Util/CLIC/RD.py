import os,shutil
import numpy
from glob import glob
from PIL import Image
import psnr

TARGET_DIR = "./encoded_files/"
REC_DIR = "./rec/"
QP_DIR = "./valid"
filesize = 0
for im_absolute in glob("/home/tong/Work/valid/*.png"):
	im_dir = im_absolute.split('/')[-1]
	#print im_dir
	for qp in range(35,38):
		#Read Image Shape		
		im = Image.open(im_absolute)
		w,h = im.size

		#Get Bitstream Size
		#print QP_DIR+"/coded_"+str(qp)+"/"+im_dir[:-4]+".bpg"
		bpg_file = os.path.getsize(QP_DIR+"/coded_"+str(qp)+"/"+im_dir[:-4]+".bin")		
		
		#Select
		bpp = bpg_file*8.0/w/h
		#print bpg_file
		if bpp<=0.15:
			filesize += bpg_file
			shutil.copy(QP_DIR+"/coded_"+str(qp)+"/"+im_dir[:-4]+".bin" , TARGET_DIR)
			shutil.copy(QP_DIR+"/valid_"+str(qp)+"/"+im_dir[:-4]+".png" , REC_DIR)
			#im = Image.open(QP_DIR+str(qp)+"/"+im)
			print qp
			break
		if qp == 37:
			print qp
			filesize += bpg_file
			shutil.copy(QP_DIR+"/coded_"+str(qp)+"/"+im_dir[:-4]+".bin" , TARGET_DIR)
			shutil.copy(QP_DIR+"/valid_"+str(qp)+"/"+im_dir[:-4]+".png" , REC_DIR)
			#im = Image.open(QP_DIR+str(qp)+"/"+im)
			#print im_dir,bpp

image_file = glob("/home/tong/Work/valid/*.png")
image_rec = glob("./rec/*.png")
image_file = sorted(image_file)
image_rec = sorted(image_rec)
#print(image_file.split("/"))[-1]
print "filesize:",filesize
result = psnr.evaluate(zip(image_file,image_rec))
print "PSNR:",result
