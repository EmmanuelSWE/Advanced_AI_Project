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

counterDictionary = {'reach-v3': 0, 'button-press-v3' : 0, 'drawer-open-v3': 0}


# function to create the environment
def createEnv(name, seed):
    return gym.make("Meta-World/MT1", env_name=name, render_mode= 'rgb_array', camera_name = 'topview', num_tasks = 45, seed = seed)


#HEADING_CONST= ['imgID','task','step','path','action','seed']
#function to store images infolder
def runAndStoreDemonstrations():
    #create the storage
    imageUtil.createImagePaths(tasks)
    csvUtils.createDataset() 

    imgStartId = 0
    seed = [11,121,111,45,999]

    for _,task in enumerate(tasks):

        print(f"PATH NAME IS {task}")
        policy = getPolicy(task)
        env = createEnv(task, seed[random.randint(0, len(seed) - 1)] )
        rgb_env = env 
        #env = HumanRendering(env) # to show the display on the window

        try: 
           while counterDictionary[task] < 100:
                execEpisode(env=env,policy=policy,rgb_env=rgb_env,task=task,startIndex=imgStartId)
           imgStartId += 100
        finally:
            env.close()


def getPolicy(name): # small function to get the policy
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

            print(f'NUMBER OF SUCCESFUL DEMONSTRATIONS FOR TASK {task} IS NOW {counterDictionary[task] }')
            obs,info = env.reset()
            currentIndex = startIndex + 1
            currentRecord = []
            imgRecord = []
            #env.render()
            for step in range(500):
                image = rgb_env.render()
                action = policy.get_action(obs)

                currentRecord.append([currentIndex,startIndex,task,step,f"{pathgen}/{task}_step{step}.png",action])
                imgRecord.append(image)
                print(f"action is {action}")
                obs,reward,terminated,truncated,info = env.step(action)

                

                currentIndex += 1

                if info["success"] == 1: 
                    counterDictionary[task]   += 1 # counting a succesful demonstration
                    image = rgb_env.render()
                    imgRecord.append(image)
                    print(f"render of task {task} done SUCCESSFULLY STROING NOW")
                    csvUtils.writeToDataset(contents=currentRecord)
                    for i, c in enumerate(currentRecord):
                         imageUtil.saveImageToPath(task,imgRecord[i],c[3], c[3] + random.randint(1,100)) # index 3 is where the step is stored
                    imageUtil.saveImageToPath(task,image,step, step + time.time()+ random.randint(1,100)) # save last image
                         
                    break
                if terminated or truncated:
                    break 

            print(f'task {task} COMPLETE GOING TO NEXT EPISODE')
            return currentIndex

