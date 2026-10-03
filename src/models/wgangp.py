import tensorflow as tf 
import math 
import numpy as np 
import glob
import imageio
import matplotlib.pyplot as plt
import os
import PIL
from tensorflow.keras import layers
import time

#making the logs
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator:
    def __init__(self):
       self.model = tf.keras.Sequential()
       self.model.add(layers.Dense(15*15*256,use_bias=False, input_shape=(4,)))
       self.model.add(layers.BatchNormalization())
       self.model.add(layers.LeakyReLU())

       self.model.add(layers.Reshape((15,15,256)))
       assert self.model.output_shape == (None, 15,15,256)

       self.model.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False))
       assert self.model.output_shape == (None,30,30, 128)
       self.model.add(layers.BatchNormalization())
       self.model.add(layers.LeakyReLU())

       self.model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
       assert self.model.output_shape == (None,60,60, 64)
       self.model.add(layers.BatchNormalization())
       self.model.add(layers.LeakyReLU())

       self.model.add(layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False))
       assert self.model.output_shape == (None,120,120, 32)
       self.model.add(layers.BatchNormalization())
       self.model.add(layers.LeakyReLU())

       self.model.add(layers.Conv2DTranspose(16, (5,5), strides=(2,2), padding='same', use_bias=False))
       assert self.model.output_shape == (None,240,240, 16)
       self.model.add(layers.BatchNormalization())
       self.model.add(layers.LeakyReLU())

       self.model.add(layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
       assert self.model.output_shape == (None,480,480, 3)
       print(f'gen shape is {self.model.summary()}')

    def getLoss(self,output):
        return crossEntropy(tf.ones_like(output), output)
    




class Discrimintator: 
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(64, (5,5), strides = (2,2), padding='same', input_shape= [480,480,3]))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Conv2D(128, (5,5), strides = (2,2), padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))
        print(f'disc shape is {self.model.summary()}')

    def getLoss(self,real,fake):
        realLoss = crossEntropy(tf.ones_like(real), real)
        fakeLoss = crossEntropy(tf.zeros_like(fake),fake)
        totalLoss = realLoss + fakeLoss
        return totalLoss


genOptimizer = tf.keras.optimizers.Adam(1e-4)
discOptimzer = tf.keras.optimizers.Adam(1e-4)

def trainStep(gen,disc,images):

    noise = tf.random.normal([tf.shape(images)[0], 4])


    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
        genImages = gen.model(noise,training= True)

        real = disc.model(images, training = True)
        fake = disc.model(genImages, training =True)

        lossforGen = gen.getLoss(fake)

        lossforDisc =disc.getLoss(real,fake)

       

    gradforGen = genTape.gradient(lossforGen, gen.model.trainable_variables)
    gradforDisc = discTape.gradient(lossforDisc, disc.model.trainable_variables)
    
    genOptimizer.apply_gradients(zip(gradforGen, gen.model.trainable_variables))
    discOptimzer.apply_gradients(zip(gradforDisc, disc.model.trainable_variables))

    return lossforGen, lossforDisc

def train_wagangp(gen,disc,dataset,epochs):
    fixedNoise = tf.random.normal([1,4])
    os.makedirs("gen_samples",exist_ok=True)

    for epoch in range(epochs):
        start = time.time()

        images = next(iter(dataset))
        print(images.shape)
        genLoss, discLoss =trainStep(gen,disc,images)

        print(f'EPICH {epoch + 1} GenLoss : {genLoss.numpy():.4f} discLoss{discLoss.numpy():.4f}')

        # make for each
        genImage = gen.model(fixedNoise,training= False)[0].numpy()
        pixels = ((genImage+ 1) * 127.5).clip(0,255).astype(np.uint8)
        imageio.imwrite(f"gen_samples/epoch_{epoch +1}.png",pixels)
        



def loadDataSet(dataset,batch):
                def pairs():
                   for i in dataset.contents:
                       yield np.array(i, dtype=np.float32) / 127.5 -1 

                return tf.data.Dataset.from_generator(
                           pairs,
                           output_signature= tf.TensorSpec(shape=(480,480,3), dtype=tf.float32),
                       ).shuffle(30).batch(batch)