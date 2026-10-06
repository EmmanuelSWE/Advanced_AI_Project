# Model: a base class sketch with empty history and results slots and the method names a model has.
import tensorflow as tf 
import math 
import numpy as np 
import  matplotlib.pyplot
from datasets.dataset import Dataset
import matplotlib.pyplot as plt
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

class Model: 
    # Base class for the models; the real models live in the other files of models/.
    def __init__(self,localRun):
        # Create empty history and results slots.
        self.history = None
        self.Results = None

    # method names a model has (the bodies are just None)
    def loadDataSet(): 
        None 

    def plotTraining():
        None 

    def plotTestting():
        None 

    def predict():
        None 
