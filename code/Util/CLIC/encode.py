# encode with name
import os
from glob import glob

# CLIC Valid
images = glob("./valid/*.png")
# Kodak
# images = glob("/data/ljp105/NIC_Dataset/test/ClassD_Kodak/*.png")
bins = glob("./bin/*.bin")
for image in bins:
    filename = image.split('/')[-1]
    # cmd = "python inference.py --encode -i %s -o /output/%s -m_dir /model/wwxn1997/temp_models/ --block_width 2048 --block_height 1024 -m 0"%(image, filename[:-3]+"bin")
    # os.system(cmd)
    cmd = "python inference.py --decode.py -i ./bin/%s -o /output/%s -m_dir /model/wwxn1997/temp_models/ --block_width 2048 --block_height 1024 -m 0"%(filename[:-3]+"bin", filename[:-4]+"_rec.png")
    os.system(cmd)

    