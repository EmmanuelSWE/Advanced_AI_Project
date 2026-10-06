"""Dataset: a tiny container class holding a named list of samples.

The imports below (numpy, time) are not used in this file.
"""
import numpy as np
import time 


class Dataset:
    """A named list of samples; the sample layout depends on who fills it (GAN, policy or predictor)."""
    def __init__(self,name):
        """Create an empty dataset called `name`."""
        self.name = name 
        self.rootPath = None
        self.contents = []

    def addToDataset(self,img): 
        """Append one sample (a tuple) to the list."""
        self.contents.append(img)

    def clearDataset(self):
        """Empty the list and print a message."""
        self.contents = []
        print(f"dataset of name {self.name} has been cleared")

    def getItem(self,index):
        """Return the sample at `index`."""
        return self.contents[index]

    def loadcontents(self,imgs,path):
        """Store the root path and append every item of `imgs` to the list.

        enumerate() makes each stored item an (index, image) tuple, not just the image.
        """
        self.rootPath = path
        # enumerate yields (index, image) pairs, so those pairs are what get stored
        for image in enumerate(imgs):
            self.contents.append(image)

        print(f'contents Loaded from root path : {self.rootPath} to dataset : {self.name}')

