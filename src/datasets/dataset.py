import numpy as np
import time 


class Dataset:
    def __int__(self,name):
        self.name = name 
        self.rootPath = None
        self.contents = None

    def addToDataset(self,img): 
        self.contents = np.concatenate((self.contents,img),axis= 0);

    def clearDataset(self):
        self.contents = None
        print(f"dataset of name {self.name} has been cleared")

    def getImage(self,index):
        return self.contents[index]

    def loadcontents(self,imgs,path):
        self.rootPath = path
        for image in enumerate(imgs):
            self.contents = np.concatenate((self.contents,image),axis=0)

        print(f'contents Loaded from root path : {self.rootPath} to dataset : {self.name}')

    