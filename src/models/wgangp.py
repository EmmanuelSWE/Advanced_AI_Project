import tensorflow as tf 
import math 
import numpy as np 
import glob
import imageio
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras import layers
import time
from utils.const_list_utils import remove

#making the logs
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator:
    def __init__(self):

       self.optimizer = tf.keras.optimizers.Adam(1e-4)
       action = layers.Input((4,), name = "action")
       task = layers.Input((3,), name ="task")
       x =  layers.Concatenate()([action,task])
       x = layers.Dense(15*15*256, activation=tf.nn.leaky_relu, use_bias=False)(x)

       x = layers.Reshape((15,15,256))(x)

       x= layers.Conv2DTranspose(128,5,strides=2,padding='same',use_bias=False, activation=tf.nn.leaky_relu)(x)
       x = layers.BatchNormalization()(x)

       x= layers.Conv2DTranspose(64,5,strides=2,padding='same',use_bias=False, activation=tf.nn.leaky_relu)(x)
       x = layers.BatchNormalization()(x)

       x= layers.Conv2DTranspose(32,5,strides=2,padding='same',use_bias=False, activation=tf.nn.leaky_relu)(x)
       x = layers.BatchNormalization()(x)

       x= layers.Conv2DTranspose(16,5,strides=2,padding='same',use_bias=False, activation=tf.nn.leaky_relu)(x)
       x = layers.BatchNormalization()(x)

       image= layers.Conv2DTranspose(3,5,strides=2,padding='same',use_bias=False, activation=tf.nn.tanh)(x)
       
       self.model = tf.keras.Model(inputs = {"action":action, "task":task}, outputs = image)


       print(f'gen shape is {self.model.summary()}')

    def getLoss(self,output):
        return -tf.reduce_mean(output)
    




class Discrimintator: 
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        image = layers.Input((480,480,3), name = "image")
        task = layers.Input((3,), name = 'task')
        x = layers.Conv2D(64,5,strides=2,activation= tf.nn.leaky_relu,padding='same')(image)
       

        x = layers.Conv2D(128,5,strides=2,activation=tf.nn.leaky_relu, padding='same')(x)
        


        flatten = layers.Flatten()(x)
        t = layers.Concatenate()([flatten,task])
        choice= layers.Dense(1)(t)

        self.model = tf.keras.Model(inputs= {"image" : image, "task": task},outputs =choice)
        print(f'disc shape is {self.model.summary()}')

    def getLoss(self,real,fake,penalty):
        return (
             tf.reduce_mean(fake) - tf.reduce_mean(real) + 10.0 * penalty
        )



def gradiantPenalty(disc,real,fake,tasks):
     batchSize = tf.shape(real)[0]
     alpha = tf.random.uniform((batchSize, 1,1 ,1),0.0, 1.0)

     interpolated = (
          real + alpha * (fake - real)
     )

     with tf.GradientTape() as gpTape:
          gpTape.watch(interpolated)
          scores = disc.model(
               {"image": interpolated, "task": tasks},
               training = True
          )
     gradiants = gpTape.gradient(scores, interpolated)
     gradiants = tf.reshape(gradiants,(batchSize, -1))
     norms = tf.sqrt(tf.reduce_sum(tf.square(gradiants),axis=1) + 1e-12)

     return tf.reduce_mean(tf.square(norms-1.0))

def trainStep(gen,disc,images,tasks):

    batchSize = tf.shape(images)[0]
    noise = tf.random.normal([tf.shape(images)[0], 4])
    


    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
        genImages = gen.model({"action" : noise, "task" : tasks},training= True)

        real = disc.model({"image" : images, "task" : tasks}, training = True)
        fake = disc.model({"image" : genImages, "task" : tasks}, training =True)

        penalty = gradiantPenalty(
             disc,images,tf.stop_gradient(genImages),tasks
        )

        lossforGen = gen.getLoss(fake)

        lossforDisc =disc.getLoss(real,fake,penalty)

        print(
             'real score:', tf.reduce_mean(real),
             'fakescore', tf.reduce_mean(fake),
             "penalty", penalty
        )

       

    gradforGen = genTape.gradient(lossforGen, gen.model.trainable_variables)
    gradforDisc = discTape.gradient(lossforDisc, disc.model.trainable_variables)
    
    gen.optimizer.apply_gradients(zip(gradforGen, gen.model.trainable_variables))
    disc.optimizer.apply_gradients(zip(gradforDisc, disc.model.trainable_variables))

    return lossforGen, lossforDisc


def train_wagangp(gen,disc,dataset,epochs):
    fixedNoise = tf.random.normal([1,4])
    fixedTask = tf.constant([[1.,0.,0.]],dtype=tf.float32)

    os.makedirs("gen_samples",exist_ok=True)
    history = {"gen_loss": [], "disc_loss": []}

    for epoch in range(epochs):
        start = time.time()
        genLosses = []
        discLosses = []

        #images = next(iter(dataset))
        #print(images.shape)
        #genLoss, discLoss =trainStep(gen,disc,images)

        for images,tasks in dataset:
             genLoss, discLoss = trainStep(gen,disc,images=images,tasks=tasks)
             genLosses.append(float(genLoss.numpy()))
             discLosses.append(float(discLoss.numpy()))

        history['gen_loss'].append(np.mean(genLosses))
        history['disc_loss'].append(np.mean(discLosses))

        print(f'EPICH {epoch + 1} GenLoss : {genLoss.numpy():.4f} discLoss{discLoss.numpy():.4f}')

        # make for each
        genImage = gen.model({"action": fixedNoise, "task": fixedTask},training= False)[0].numpy()
        pixels = ((genImage+ 1) * 127.5).clip(0,255).astype(np.uint8)
        imageio.imwrite(f"gen_samples/epoch_{epoch +1}.png",pixels)
    return history
        

def plotHistory(history,epochs):
     fig, axes = plt.subplots(1,2,figsize = (11,4))
     epochs = range(1,len(history['gen_loss']) + 1)

     axes[0].plot(epochs,history['gen_loss'],marker="o")
     axes[0].set_title("GeneratorLoss")
     axes[1].plot(epochs, history["disc_loss"],marker="^")
     axes[1].set_title("Discriminator Loss")

     for ax in axes:
          ax.set_xlabel("Epoch")
          ax.set_ylabel("Loss")
          ax.grid(True)

     plt.tight_layout()
     plt.savefig("GAN_Trained_Losses.png",dpi = 150)
     

def loadDataSet(dataset,batch):
                def pairs():
                   for i , task in dataset.contents:
                       if isinstance(i,str):
                            with Image.open(i) as image:
                                 imageArray = np.asarray(image.convert("RGB"), np.float32)
                       else:
                            imageArray = np.array(i, dtype=np.float32)
                       yield (imageArray/ 127.5 -1,
                              np.asarray(task,np.float32))


                return tf.data.Dataset.from_generator(
                           pairs,
                           output_signature=( tf.TensorSpec(shape=(480,480,3), dtype=tf.float32),tf.TensorSpec(shape=(3,), dtype=tf.float32))
                       ).shuffle(30).batch(batch)