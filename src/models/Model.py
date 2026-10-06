"""Model: an unused stub base class (nothing in the project imports it).

The imports are leftovers and the methods only contain `None`.
"""
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
    """Placeholder base class for the models; the real models live in the other files of models/."""
    def __init__(self,localRun):
        """Create empty history/results slots; `localRun` is accepted but not stored."""
        self.history = None
        self.Results = None

    # placeholder methods: each body is just None, so nothing is implemented here
    def loadDataSet(): 
        None 

    def plotTraining():
        None 

    def plotTestting():
        None 

    def predict():
        None 
