# Builds Dataset objects from the CSV rows (used by the menu path, not by the CRIL loop).
from datasets.dataset import Dataset
import utils.image_utils as imgUtils
import utils.csv_utils as csvUtils
import utils.const_list_utils as listUtils
import itertools
import numpy as np



def makeDataset(name,demonStart,demonEnd,dsTyputie):
  
  """Build a Dataset from the demos numbered demonStart..demonEnd.

  Args:
      name: dataset name.
      demonStart, demonEnd: first and last demonstration number (inclusive).
      dsTyputie: 1 = (image, task) for the GAN; 2 = ((image, action), task) for the policy;
          anything else = (image, action, next image, task) for the predictor.
  Returns: a Dataset.
  """
  contents = []
  contents = csvUtils.readFile(demonEnd,demonStart) # getting all the contents needed for the dataset based on the amount needed 
  tempDS = Dataset(name)
  for i, content in enumerate(contents):
    task = listUtils.task_vector(content)
    # the dataset type decides what each sample contains
    if dsTyputie == 1:
      tempDS.addToDataset((getImage(content), task)) # generator
    elif dsTyputie ==2:
      tempDS.addToDataset((getImageAndAction(content), task)) # behavioral cloning
    else: 
      # predictor pairs: a row is paired with the next row of the same demonstration
      if(i +1 < len(contents) -1 ):
        # only pair rows from the same demonstration and the same task
        if(getDemo(content) == getDemo(contents[i+1]) and content[2] == contents[i+1][2]):
          #if this condition is reached then we can make the 4 actions togeth
          # the pair keeps this row's action and the next row's image
          img,action = getImageAndAction(content)
          nxt_img, nxt_action = getImageAndAction(contents[i+1])
          tempDS.addToDataset((img,action,nxt_img,task))

  return tempDS

def getImage(arr):
  # Open the image whose path is in CSV column 4; returns a PIL image (opened lazily).
  return imgUtils.loadImageFromPath(arr[4])

def getImageAndAction(arr):
  # Return (image, action) for a CSV row; the action is parsed from text such as [0.1 0.2 0.3 0.4] into a float32 array.
  # get the action 
  action = arr[5]
  # drop the leading '[' and cut the text at ']'
  action = action[1::].split(']')[0]
  #print(f"action is {action}")
  # np.fromstring with sep=' ' parses the space-separated numbers
  action = np.fromstring(action, sep= ' ', dtype=np.float32)
  return (getImage(arr),action)


def getDemo(arr):
   # Return the demonstration number from the image file name (.../task_step3_id12.png -> 12).
   tokens = arr[4].split("_")
   demoimg = tokens[-1].split(".")
   demoNum = int(demoimg[0].strip()[2:])
   return demoNum
   