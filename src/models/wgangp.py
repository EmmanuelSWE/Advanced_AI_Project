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


class Generator:
    def __init__(self):
       self.model = tf.keras.Sequential()
       self.model.add(layers.Dense(7*7*256,use_bias=False, input_shape=(4,)))
       self.model.add(layers.BatchNormalization())
       self.model.add(layers.LeakyReLU())

       self.model.add

def train_wagangp():
    None