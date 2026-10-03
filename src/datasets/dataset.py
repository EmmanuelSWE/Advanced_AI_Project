import numpy as np
import time 


class Dataset:
    def __init__(self,name):
        self.name = name 
        self.rootPath = None
        self.contents = []

    def addToDataset(self,img): 
        self.contents.append(img)

    def clearDataset(self):
        self.contents = []
        print(f"dataset of name {self.name} has been cleared")

    def getItem(self,index):
        return self.contents[index]

    def loadcontents(self,imgs,path):
        self.rootPath = path
        for image in enumerate(imgs):
            self.contents.append(image)

        print(f'contents Loaded from root path : {self.rootPath} to dataset : {self.name}')

