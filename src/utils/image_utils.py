
import imageio.v3 as iio 
import time
from pathlib import Path 
from utils.const_list_utils import GEN_PATH_CONST as genPath


def createImagePaths(arrNames):
    for _, name in enumerate(arrNames):
        Path(f'{genPath}/{name}').mkdir(parents = True,exist_ok=True);
        print(f'made paths for task: {name}')
    print('All Image paths made succesfully!')

def saveImageToPath(path,image,step):
    pathToSave = genPath+'/' + path 
    if(Path(pathToSave).exists()):
        iio.imwrite(f"{pathToSave}/{path}_step{step}.png",image)
    print(f'{pathToSave} path does not exist')


def loadImageFromPath():
    pass


    
