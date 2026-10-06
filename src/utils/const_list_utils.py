"""Shared constants and small helpers: task names, folder and CSV names, the one-hot task vector, folder removal."""
import tensorflow as tf
from pathlib import Path
import numpy as np
import shutil
# the three Meta-World tasks, in the order they are learned
TASKS_CONST = ['reach-v3','button-press-v3', 'drawer-open-v3']
# relative paths (from src/): generated screenshots folder and the CSV index folder/file
GEN_PATH_CONST = '../DATASET_generations'
DATASET_PATH = '../MODELS_DATASET_CSV'
DATASET_NAME_CONST = 'RepCRIL_222127212_HE_Ashimwe_DATASET.csv'
# CSV column names
HEADING_CONST= ['imgID','demonstarionID','task','step','path','action']

# tunable hyperparameters; the defaults are the values the code used before this dict existed.
# The menu item hyperparamConfig (UI/MenuItems.py) edits this dict, and the training code reads it
# when training starts, so a change applies to the next run. The type of each default (int or float)
# decides how typed input is converted. The GAN learning rate (1e-4) is not here: its optimizers are
# built once when the Menu is created, so changing it later would need a new mechanism.
HYPERPARAMS = {
    'policy_lr': 1e-3,        # policy Adam learning rate (Keras 'adam' default)
    'policy_epochs': 5,
    'policy_batch': 1,
    'pred_lr': 1e-4,          # predictor Adam learning rate
    'pred_epochs': 5,         # menu item 5
    'pred_batch': 30,         # menu items 5 and 6
    'gan_epochs': 5,          # menu items 3 and 4
    'gan_batch': 1,           # menu item 3
    'gan_test_batch': 4,      # menu item 4
    'gp_lambda': 10.0,        # gradient penalty weight in the critic loss
    'cril_pred_epochs': 5,    # CRIL (menu item 7)
    'cril_pred_batch': 1,
    'cril_gan_epochs': 30,
    'cril_gan_batch': 1,
    'cril_sample_steps': 30,  # length of the saved sample rollouts
}


# file manipulation util didnt know where to put it
def remove(pathstr):
    """Delete the folder at `pathstr` and everything in it, if it exists; otherwise do nothing."""
    delPath =Path(pathstr)
    if delPath.exists() and  delPath.is_dir():
        shutil.rmtree(delPath)
        print("Full directory removes")
        return


# TASKS is an immutable tuple; TASK_TO_ID is a dict comprehension mapping task name -> index (0, 1, 2)
TASKS = (TASKS_CONST[0],TASKS_CONST[1],TASKS_CONST[2])
TASK_TO_ID = {name: i for i, name in enumerate(TASKS)}


def task_vector(row):
    """One-hot vector for the task of a CSV row.

    Args: row (list of strings; row[2] is the task name). Returns: float32 array of length 3, e.g. [1, 0, 0].
    """
    # tf.one_hot(index, 3) gives the one-hot; .numpy() converts it to NumPy; float32 matches the model inputs
    return tf.one_hot(TASK_TO_ID[row[2]],len(TASKS)).numpy().astype(np.float32)