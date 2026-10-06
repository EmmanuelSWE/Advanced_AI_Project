"""Image helpers: create the folders for generated images, save and load PNGs, convert to NumPy arrays."""
from PIL import Image
import imageio.v3 as iio 
import time
from pathlib import Path 
from utils.const_list_utils import GEN_PATH_CONST as genPath
from utils.const_list_utils import remove
import numpy as np

def createImagePaths(arrNames):
    """Delete the generated-images folder and create one sub-folder per task name in arrNames."""
    remove(genPath) # first remove
    for _, name in enumerate(arrNames):
      
        Path(f'{genPath}/{name}').mkdir(parents = True,exist_ok=True);
        print(f'made paths for task: {name}')
    print('All Image paths made succesfully!')

def saveImageToPath(path,image,step,index):
    """Save `image` as PNG named {path}_step{step}_id{index}.png in the task folder `path`;
    prints a message if the folder does not exist.
    """
    pathToSave = genPath+'/' + path 
    if(Path(pathToSave).exists()):
        iio.imwrite(f"{pathToSave}/{path}_step{step}_id{index}.png",image)
        #print(f"saved image {path} of  {step}")
    else: 
        print(f'{pathToSave} path does not exist for step {step}')


def loadImageFromPath(path):
    """Open an image file with PIL and return the PIL image."""
    return Image.open(path, 'r', None)


def getSize(image): 
    """Return the (width, height) of a PIL image."""
    return image.size



def imageArray(value):
    """Return a uint8 RGB array from a file path (str or Path) or from an array / PIL image."""
    if isinstance(value, (str, Path)):
        # `with` closes the file after reading; convert('RGB') makes sure there are 3 channels; copy=True makes an independent array
        with Image.open(value) as image:
            return np.array(image.convert("RGB"), dtype= np.uint8, copy = True)

    # not a path: convert the array-like input to a uint8 array
    return np.array(value, dtype=np.uint8, copy= True)
