
from pathlib import Path
import shutil
TASKS_CONST = ['reach-v3','button-press-v3', 'drawer-open-v3']
GEN_PATH_CONST = '../DATASET_generations'
DATASET_PATH = '../MODELS_DATASET_CSV'
DATASET_NAME_CONST = 'RepCRIL_222127212_HE_Ashimwe_DATASET.csv'
HEADING_CONST= ['imgID','demonstarionID','task','step','path','action']


# file manipulation util didnt know where to put it
def remove(pathstr):
    delPath =Path(pathstr)
    if delPath.exists() and  delPath.is_dir():
        shutil.rmtree(delPath)
        print("Full directory removes")
        return

