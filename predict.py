from nets.segnet import resnet50_segnet
from PIL import Image,ImageFont,ImageDraw
import numpy as np
import random
import os
import copy
random.seed(0)

class_colors = [[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],
                [152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]]

NCLASSES = 19
HEIGHT = 416
WIDTH = 416
model = resnet50_segnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model.load_weights("logs/last1.h5")
# please change to your train path
imgs = os.listdir(r"C:\Users\WL\Desktop\Segnet Transfer-Learning\img/")

for jpg in imgs:

    img = Image.open(r"C:\Users\WL\Desktop\Segnet Transfer-Learning\img/"+jpg)

    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    pr = model.predict(img)[0]
    pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2), NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT/2), int(WIDTH/2),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))

    image = Image.blend(old_img, seg_img, 1)
    image.save("./img_out/" + jpg)
