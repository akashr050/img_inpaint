import os
from glob import glob
import numpy as np
folder_location = "/media/abhishek/New Volume1/celeba/CelebA/Img/img_celeba.7z/"
path = "img_celeba"
files = glob(folder_location+path+'/*.jpg')
files = np.random.permutation(files)
val = open(folder_location+"evaluation.txt", "a")
with open(folder_location+"train.txt", "a") as wrt:
  for file in files[:200000]:
    wrt.write(file.replace(folder_location+path+"/","")+"\n")
  for file in files[200001:]:
    val.write(file.replace(folder_location+path+"/","")+"\n")
val.close()
