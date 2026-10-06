"""Collect the demonstration dataset from Meta-World.

Runs Meta-World's scripted expert policies, saves a frame per step as a PNG and writes one CSV row per step.
runAndStoreDemonstrations() is not called from main.py or the menu; it has to be run by hand.
"""
#import all the expoert demonstration actions i need
import gymnasium as gym 
import metaworld 
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
from metaworld.policies.sawyer_button_press_v3_policy import SawyerButtonPressV3Policy
from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
from gymnasium.wrappers import HumanRendering
from utils.const_list_utils import GEN_PATH_CONST as pathgen 
from utils.const_list_utils import TASKS_CONST as tasks

import utils.csv_utils as csvUtils 
import utils.image_utils as imageUtil
import random
import numpy as np
import time

# successful demonstrations stored so far per task (module-level state shared by all episodes)
counterDictionary = {'reach-v3': 0, 'button-press-v3' : 0, 'drawer-open-v3': 0}


# function to create the environment
def createEnv(name, seed):
    """Create the Meta-World MT1 environment for one task.

    rgb_array rendering with the 'topview' camera lets frames be returned as images. Args: name (task), seed.
    """
    return gym.make("Meta-World/MT1", env_name=name, render_mode= 'rgb_array', camera_name = 'topview', num_tasks = 45, seed = seed)


#HEADING_CONST= ['imgID','task','step','path','action','seed']
#function to store images infolder
def runAndStoreDemonstrations():
    """Collect 100 successful expert demonstrations for every task and store frames + CSV rows."""
    #create the storage
    imageUtil.createImagePaths(tasks)
    csvUtils.createDataset() 

    # imgStartId is a running image id across all tasks
    imgStartId = 0
    # candidate seeds; one is picked at random for each task
    seed = [11,121,111,45,999]

    for _,task in enumerate(tasks):

        print(f"PATH NAME IS {task}")
        policy = getPolicy(task)
        env = createEnv(task, seed[random.randint(0, len(seed) - 1)] )
        rgb_env = env 
        #env = HumanRendering(env) # to show the display on the window

        # try/finally makes sure env.close() runs even if an episode raises an error
        try: 
           # keep running episodes until 100 successful demonstrations are stored for this task
           while counterDictionary[task] < 100:
                imgStartId = execEpisode(env=env,policy=policy,rgb_env=rgb_env,task=task,startIndex=imgStartId)
                
           
        finally:
            env.close()


def getPolicy(name): # small function to get the policy
    """Return the scripted expert policy for the task name (None if the name is unknown)."""
    print(f"NAME OF POLICY ILL GET FOR {name}")
    if name == tasks[0]:
        return SawyerReachV3Policy()
    elif name == tasks[1]:
        return SawyerButtonPressV3Policy()
    elif  name == tasks[2]:
        return SawyerDrawerOpenV3Policy()


#function to execute task episode
#HEADING_CONST= ['imgID','demonstarionID','task','step','path','action']
def execEpisode(env, policy , rgb_env, task, startIndex):

            """Run one episode of at most 500 steps. If the expert succeeds, save its rows and frames.

            Args: env, policy, rgb_env (same env, used for rendering), task, startIndex (first image id).
            Returns: the next free image id.
            """
            print(f'NUMBER OF SUCCESFUL DEMONSTRATIONS FOR TASK {task} IS NOW {counterDictionary[task] }')
            obs,info = env.reset()
            demo_id = counterDictionary[task] + 1
            currentIndex = startIndex 
            currentRecord = []
            imgRecord = []
            #env.render()
            for step in range(500):
                # render the frame first, then let the expert choose the action from the observation
                image = rgb_env.render()
                action = policy.get_action(obs)

                # row layout follows HEADING_CONST; the path is where the PNG is saved if the episode succeeds
                currentRecord.append([currentIndex,demo_id,task,step,f"{pathgen}/{task}/{task}_step{step}_id{demo_id}.png",action])
                imgRecord.append(image)
                print(f"action is {action}")
                obs,reward,terminated,truncated,info = env.step(action)

                

                currentIndex += 1

                # success: count the demo and write rows and frames; failed episodes are not saved
                if info["success"] == 1: 
                    counterDictionary[task]   += 1 # counting a succesful demonstration
                    image = rgb_env.render()
                    imgRecord.append(image)
                    print(f"render of task {task} done SUCCESSFULLY STROING NOW")
                    csvUtils.writeToDataset(contents=currentRecord)
                    for i, c in enumerate(currentRecord):
                         imageUtil.saveImageToPath(task,imgRecord[i],c[3], demo_id) # index 3 is where the step is stored
                    imageUtil.saveImageToPath(task,image,step + 1, demo_id) # save last image
                         
                    break
                # episode ended without success: nothing is saved
                if terminated or truncated:
                    break 

            print(f'task {task} COMPLETE GOING TO NEXT EPISODE')
            return currentIndex

