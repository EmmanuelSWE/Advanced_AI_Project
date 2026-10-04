from datasets.dataset import Dataset
import utils.image_utils as imgUtils
import utils.csv_utils as csvUtils
import utils.const_list_utils as listUtils
import itertools
import numpy as np



def makeDataset(name,demonStart,demonEnd,dsTyputie):
  
  contents = []
  contents = csvUtils.readFile(demonEnd,demonStart) # getting all the contents needed for the dataset based on the amount needed 
  tempDS = Dataset(name)
  for i, content in enumerate(contents):
    task = listUtils.task_vector(content)
    if dsTyputie == 1:
      tempDS.addToDataset((getImage(content), task)) # generator
    elif dsTyputie ==2:
      tempDS.addToDataset((getImageAndAction(content), task)) # behavioral cloning
    else: 
      if(i +1 < len(contents) -1 ):
        if(getDemo(content) == getDemo(contents[i+1]) and content[2] == contents[i+1][2]):
          #if this condition is reached then we can make the 4 actions togeth
          img,action = getImageAndAction(content)
          nxt_img, nxt_action = getImageAndAction(contents[i+1])
          tempDS.addToDataset((img,action,nxt_img,task))

  return tempDS

def getImage(arr):
  return imgUtils.loadImageFromPath(arr[4])

def getImageAndAction(arr):
  # get the action 
  action = arr[5]
  action = action[1::].split(']')[0]
  #print(f"action is {action}")
  action = np.fromstring(action, sep= ' ', dtype=np.float32)
  return (getImage(arr),action)


def getDemo(arr):
   tokens = arr[4].split("_")
   demoimg = tokens[-1].split(".")
   demoNum = int(demoimg[0].strip()[2:])
   return demoNum
   