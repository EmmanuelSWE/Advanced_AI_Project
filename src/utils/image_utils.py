from PIL import Image
import imageio.v3 as iio 
import time
from pathlib import Path 
from utils.const_list_utils import GEN_PATH_CONST as genPath
from utils.const_list_utils import remove
import numpy as np

def createImagePaths(arrNames):
    remove(genPath) # first remove
    for _, name in enumerate(arrNames):
      
        Path(f'{genPath}/{name}').mkdir(parents = True,exist_ok=True);
        print(f'made paths for task: {name}')
    print('All Image paths made succesfully!')

def saveImageToPath(path,image,step,index):
    pathToSave = genPath+'/' + path 
    if(Path(pathToSave).exists()):
        iio.imwrite(f"{pathToSave}/{path}_step{step}_id{index}.png",image)
        #print(f"saved image {path} of  {step}")
    else: 
        print(f'{pathToSave} path does not exist for step {step}')


def loadImageFromPath(path):
    return Image.open(path, 'r', None)


def getSize(image): 
    return image.size



def imageArray(value):
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return np.array(image.convert("RGB"), dtype= np.uint8, copy = True)

    return np.array(value, dtype=np.uint8, copy= True)
