#import all the expoert demonstration actions i need
import gymnasium as gym 
import metaworld 
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy
from metaworld.policies.sawyer_button_press_v3_policy import SawyerButtonPressV3Policy
from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy
from gymnasium.wrappers import HumanRendering
from utils.const_list_utils import GEN_PATH_CONST as pathgen 
from utils.const_list_utils import TASKS_CONST as tasks
import utils.image_utils as imageUtil

# function to create the environment
def createEnv(name):
    return gym.make("Meta-World/MT1", env_name=name, render_mode= 'rgb_array', camera_name = 'topview')


#function to store images infolder
def runAndStoreDemonstrations():
    imageUtil.createImagePaths(tasks)

    for _,task in enumerate(tasks):
        print(f"PATH NAME IS {task}")
        policy = getPolicy(task)
        env = createEnv(task)
        rgb_env = env 
        env = HumanRendering(env)

        try: 
            obs,info = env.reset()
            #env.render()
            for step in range(500):
                action = policy.get_action(obs)
                obs,reward,terminated,truncated,info = env.step(action)
                image = rgb_env.render()
                imageUtil.saveImageToPath(task,image,step)

                if info["success"] == 1: 
                    print(f"render of task {task} done")
                if terminated or truncated:
                    break 
        finally:
            env.close()







def getPolicy(name): # small function to get the policy
    if name == pathgen[0]:
        return SawyerReachV3Policy()
    elif name == pathgen[1]:
        return SawyerButtonPressV3Policy()
    else:
        return SawyerDoorOpenV3Policy()