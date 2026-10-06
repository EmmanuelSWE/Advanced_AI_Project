"""WGAN-GP image generator.

The Generator makes a 480x480 first frame from random noise plus a one-hot task. The Discrimintator
(the critic) scores images; the gradient penalty keeps that score smooth (roughly 1-Lipschitz).
Contains the losses, the training loop, a loss plot and the tf.data loader.
"""
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
import utils.const_list_utils as listUtils

# show only ERROR messages from TensorFlow's logger (hides warnings and info)
#making the logs
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



# defined but not used: the WGAN-GP losses below do not use cross-entropy
crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator:
    """Generator network: (4-number noise, task one-hot) -> 480x480 RGB image in [-1, 1]."""
    def __init__(self):

       """Build the generator model and its Adam optimizer."""
       # Adam with learning rate 1e-4; each network has its own optimizer
       self.optimizer = tf.keras.optimizers.Adam(1e-4)
       # two inputs: a 4-number noise vector (named 'action') and the 3-number task one-hot
       action = layers.Input((4,), name = "action")
       task = layers.Input((3,), name ="task")
       x =  layers.Concatenate()([action,task])
       # Dense expands the 7 inputs to a 15x15x256 map; BatchNormalization steadies training
       x = layers.Dense(15*15*256, activation=tf.nn.leaky_relu, use_bias=False)(x)

       x = layers.Reshape((15,15,256))(x)

       # five stride-2 transposed convolutions double the size each time: 15 -> 30 -> 60 -> 120 -> 240 -> 480
       x= layers.Conv2DTranspose(128,5,strides=2,padding='same',use_bias=False, activation=tf.nn.leaky_relu)(x)
       x = layers.BatchNormalization()(x)

       x= layers.Conv2DTranspose(64,5,strides=2,padding='same',use_bias=False, activation=tf.nn.leaky_relu)(x)
       x = layers.BatchNormalization()(x)

       x= layers.Conv2DTranspose(32,5,strides=2,padding='same',use_bias=False, activation=tf.nn.leaky_relu)(x)
       x = layers.BatchNormalization()(x)

       x= layers.Conv2DTranspose(16,5,strides=2,padding='same',use_bias=False, activation=tf.nn.leaky_relu)(x)
       x = layers.BatchNormalization()(x)

       # last layer: 3 channels (RGB) with tanh, so pixel values are in [-1, 1]
       image= layers.Conv2DTranspose(3,5,strides=2,padding='same',use_bias=False, activation=tf.nn.tanh)(x)
       
       self.model = tf.keras.Model(inputs = {"action":action, "task":task}, outputs = image)


       # summary() prints the table itself and returns None, so this also prints None
       print(f'gen shape is {self.model.summary()}')

    def getLoss(self,output):
        """Generator loss: minus the mean critic score of the fake images (the generator wants a high score)."""
        return -tf.reduce_mean(output)
    




class Discrimintator: 
    """Critic (the WGAN discriminator): gives an image + task one real-valued score instead of a probability."""
    def __init__(self):
        """Build the critic model and its Adam optimizer."""
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        # two inputs: a 480x480 RGB image and the 3-number task one-hot
        image = layers.Input((480,480,3), name = "image")
        task = layers.Input((3,), name = 'task')
        # two stride-2 convolutions (480 -> 240 -> 120), then flatten
        x = layers.Conv2D(64,5,strides=2,activation= tf.nn.leaky_relu,padding='same')(image)
       

        x = layers.Conv2D(128,5,strides=2,activation=tf.nn.leaky_relu, padding='same')(x)
        


        flatten = layers.Flatten()(x)
        # append the task one-hot; one Dense unit with no activation gives the unbounded score
        t = layers.Concatenate()([flatten,task])
        choice= layers.Dense(1)(t)

        self.model = tf.keras.Model(inputs= {"image" : image, "task": task},outputs =choice)
        print(f'disc shape is {self.model.summary()}')

    def getLoss(self,real,fake,penalty):
        """Critic loss: mean(fake) - mean(real) + gp_lambda * gradient penalty (default 10, the usual WGAN-GP weight)."""
        return (
             tf.reduce_mean(fake) - tf.reduce_mean(real) + listUtils.HYPERPARAMS['gp_lambda'] * penalty
        )



def gradiantPenalty(disc,real,fake,tasks):
     """Gradient penalty: how far the critic's gradient norm is from 1 on points between real and fake images.

     Args: disc (critic), real and fake image batches, tasks (task one-hots).
     Returns: scalar penalty.
     """
     batchSize = tf.shape(real)[0]
     # one random mixing weight per image in [0, 1], shaped (batch, 1, 1, 1) so it broadcasts over the pixels
     alpha = tf.random.uniform((batchSize, 1,1 ,1),0.0, 1.0)

     # points on the straight line between each real image and its fake image
     interpolated = (
          real + alpha * (fake - real)
     )

     # GradientTape records operations so the critic score can be differentiated with respect to the input image
     with tf.GradientTape() as gpTape:
          gpTape.watch(interpolated)
          scores = disc.model(
               {"image": interpolated, "task": tasks},
               training = True
          )
     # gradient of the score with respect to the interpolated images
     gradiants = gpTape.gradient(scores, interpolated)
     # flatten each image's gradient and take its L2 norm (1e-12 avoids the square root of zero)
     gradiants = tf.reshape(gradiants,(batchSize, -1))
     norms = tf.sqrt(tf.reduce_sum(tf.square(gradiants),axis=1) + 1e-12)

     # penalty = mean squared distance of the gradient norm from 1
     return tf.reduce_mean(tf.square(norms-1.0))

def trainStep(gen,disc,images,tasks):

    """One training step on a batch: update the critic and the generator once each.

    Args: gen, disc (wrapper objects), images (real batch), tasks (task one-hots).
    Returns: (generator loss, critic loss).
    """
    batchSize = tf.shape(images)[0]
    # fresh random noise for every image in the batch
    noise = tf.random.normal([tf.shape(images)[0], 4])
    


    # two tapes record the forward pass: one for the generator's gradients, one for the critic's
    with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
        genImages = gen.model({"action" : noise, "task" : tasks},training= True)

        real = disc.model({"image" : images, "task" : tasks}, training = True)
        fake = disc.model({"image" : genImages, "task" : tasks}, training =True)

        # stop_gradient: the penalty must not send gradients into the generator
        penalty = gradiantPenalty(
             disc,images,tf.stop_gradient(genImages),tasks
        )

        lossforGen = gen.getLoss(fake)

        lossforDisc =disc.getLoss(real,fake,penalty)

        # debug print of the mean critic scores and the penalty on every step
        print(
             'real score:', tf.reduce_mean(real),
             'fakescore', tf.reduce_mean(fake),
             "penalty", penalty
        )

       

    # gradient of each loss with respect to its own network's weights
    gradforGen = genTape.gradient(lossforGen, gen.model.trainable_variables)
    gradforDisc = discTape.gradient(lossforDisc, disc.model.trainable_variables)
    
    # one optimizer step for each network, i.e. one critic update and one generator update per batch
    gen.optimizer.apply_gradients(zip(gradforGen, gen.model.trainable_variables))
    disc.optimizer.apply_gradients(zip(gradforDisc, disc.model.trainable_variables))

    return lossforGen, lossforDisc


def train_wagangp(gen,disc,dataset,epochs):
    """Train the GAN for `epochs` epochs over `dataset`, saving one sample image per epoch in gen_samples/.

    Returns: dict with the per-epoch mean 'gen_loss' and 'disc_loss'.
    """
    # fixed noise and a fixed task (task 0, reach) so the saved sample images are comparable across epochs
    fixedNoise = tf.random.normal([1,4])
    fixedTask = tf.constant([[1.,0.,0.]],dtype=tf.float32)

    # folder for the per-epoch sample images
    os.makedirs("gen_samples",exist_ok=True)
    history = {"gen_loss": [], "disc_loss": []}

    for epoch in range(epochs):
        start = time.time()
        genLosses = []
        discLosses = []

        #images = next(iter(dataset))
        #print(images.shape)
        #genLoss, discLoss =trainStep(gen,disc,images)

        # one pass over all batches; each batch is (images, task vectors)
        for images,tasks in dataset:
             genLoss, discLoss = trainStep(gen,disc,images=images,tasks=tasks)
             genLosses.append(float(genLoss.numpy()))
             discLosses.append(float(discLoss.numpy()))

        history['gen_loss'].append(np.mean(genLosses))
        history['disc_loss'].append(np.mean(discLosses))

        # prints the losses of the last batch of the epoch (the history stores the epoch mean)
        print(f'EPICH {epoch + 1} GenLoss : {genLoss.numpy():.4f} discLoss{discLoss.numpy():.4f}')

        # make for each
        # generate one sample image; (x + 1) * 127.5 maps [-1, 1] to [0, 255]
        genImage = gen.model({"action": fixedNoise, "task": fixedTask},training= False)[0].numpy()
        pixels = ((genImage+ 1) * 127.5).clip(0,255).astype(np.uint8)
        imageio.imwrite(f"gen_samples/epoch_{epoch +1}.png",pixels)
    return history
        

def plotHistory(history,epochs):
     """Plot the generator and critic loss curves and save them to GAN_Trained_Losses.png."""
     fig, axes = plt.subplots(1,2,figsize = (11,4))
     # the `epochs` argument is replaced here by a range matching the history length
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
                """Turn a Dataset of (image, task) items into a streaming tf.data pipeline.

                Images are scaled to [-1, 1] to match the generator's tanh output.
                Args: dataset (a Dataset), batch (batch size). Returns: a batched tf.data.Dataset.
                """
                def pairs():
                   # generator that yields one (image, task) pair at a time, so images are read from disk lazily
                   for i , task in dataset.contents:
                       # an image is either a file path (str) or already an image/array
                       if isinstance(i,str):
                            with Image.open(i) as image:
                                 imageArray = np.asarray(image.convert("RGB"), np.float32)
                       else:
                            imageArray = np.array(i, dtype=np.float32)
                       # scale pixels from [0, 255] to [-1, 1]
                       yield (imageArray/ 127.5 -1,
                              np.asarray(task,np.float32))


                # from_generator streams the samples; shuffle(30) uses a buffer of 30, then batch
                return tf.data.Dataset.from_generator(
                           pairs,
                           output_signature=( tf.TensorSpec(shape=(480,480,3), dtype=tf.float32),tf.TensorSpec(shape=(3,), dtype=tf.float32))
                       ).shuffle(30).batch(batch)