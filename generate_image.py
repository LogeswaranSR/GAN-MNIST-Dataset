# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:12:52 2023

@author: Loges

Python file to generate image using GAN model
"""

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

## Load model from the file
generator=load_model(r"models/mnist_g.h5")

latent_dim=100
rows, cols=1,1 ## Define the no. of rows and cols in your image here

noise=np.random.randn(rows*cols,latent_dim)
img=generator.predict(noise)

## Presenting the image using matplotlib

img=(0.5*img)+0.5
idx=0
fig, axs=plt.subplots(rows,cols)
if rows>1:
    for i in range(rows):
        if cols>1:
            for j in range(cols):
                axs[i,j].imshow(img[idx].reshape(28,28), cmap='gray')
                axs[i,j].axis('off')
                idx+=1
        else:
            axs[i].imshow(img[idx].reshape(28,28), cmap='gray')
            axs[i].axis('off')
            idx+=1
else:
    if cols>1:
        for j in range(cols):
            axs[j].imshow(img[idx].reshape(28,28), cmap='gray')
            axs[j].axis('off')
            idx+=1
    else:
        axs.imshow(img.reshape(28,28), cmap='gray')
fig.show()
plt.close()