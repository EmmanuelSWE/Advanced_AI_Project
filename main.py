#imports 
import torch 
import torch.nn as nn 
import numpy as np 
import random 
from behaviorcloning import behavior_cloning
from wgangp import train_wagangp
from predictor import train_predictor, predict_images 
from utils import generate_images

#enviroment for the different type of reaching
envs = [  'reach-v2',
    'button-press-v2',
    'drawer-open-v2']

# paths for the videos  
raw_data_paths = [ 'reach-v2_topview',
    'button-press-v2_topview',
    'drawer-open-v2_topview']

# paths for the generated data 
generated_paths = ['generated_images/reach-v2_',
    'generated_images/reach-v2_button-press-v2_',
    'generated_images/reach-v2_button-press-v2_drawer-open-v2_']


def perform(task: int, what:str,seed=0):
    current_raw_path = raw_data_paths[0:task+1]

    if task == 0: #since its 0 tasks itll just take the nothing
        current_pseudo_path = current_raw_path
    else:
        current_pseudo_path = [] 
        for i in range(task):
            current_pseudo_path.append(generated_paths[task-1]+'/'+envs[i]+'_') # why start at the previous ohh casue the index will start from 1 
        current_pseudo_path.append(raw_data_paths[task])

    current_envs = envs[0:task+1] #current environments

    if what == 'train_policy':
        if task == 0:
            load_old_pol = False 
        else: 
            load_old_pol = True 
        # settingi the path for the policy
        old_pol_path = 'trained_pol/'+ str(seed) + '/'
        task_name = ''
        for j in range(task):
            task_name += (envs[j] + '_')
        old_pol_path += task_name + '/policy_' + task_name+'.pth'


        #run behavoir cloning algorithm 
        behavior_cloning(
            current_pseudo_path,
            current_envs,
            test_data_path = current_raw_path,
            batch_size=100,
            max_epoch=50,
            what='normal',
            load_old_policy = load_old_pol,
            old_pilicy = old_pol_path,
            seed = seed
        )
    elif what == 'train_generator':
        if task == 0:
            load_old_pol = False 
        else: 
            load_old_pol = True 

        train_wagangp(current_pseudo_path, current_envs, oad_old_models = load_old_pol,old_g_path = 'trained_generators/reach-v2_/G73000.pth',
            old_d_path =  'trained_generators/reach-v2_/D73000.pth')
        
    elif what == 'train_predictor':
        if task == 0:
                    load_old_pol = False 
        else: 
                    load_old_pol = True 
        
        predictor_path = 'trained_predictiors/'
        for j in range(task):
            predictor_path += (envs[j] + '_')

        train_predictor(current_pseudo_path,
                        current_envs,
                        load_old_pol,
                        old_pol_path = predictor_path)

    elif what == 'generate_first_frames':
        generate_images(
             current_envs,
             generated_path='trained_generators/reach-v2/G11000.pth',
             trail_num = 100,
             method='wgangp'
        )
    elif what == 'predict_frames':
        policy_path = 'trained_policies/' + str(seed) +'/'
        task_name = ' '
        for j in range(task+1):
            task_name += (envs[j] + '_')
        
        policy_path += task_name
        policy_path += '/policy_'
        policy_path += task_name
        policy_path += '.pth'

        img_path = 'generated_images/'
        predictor_path = 'trained_predictors/'
        for j in range(task+1):
             img_path += (envs[j]+'_')
             predictor_path += (envs[j] + '_')
        predictor_path += '/predictor.pth'

        predict_images(img_path, policy_path, predictor_path, envs)
    elif what == 'baseline':
        if task == 0:
                    load_old_pol = False 
        else: 
                    load_old_pol = True 
                # settingi the path for the policy
        old_pol_path = 'baseline/trained_pol/'+ str(seed) + '/'
        task_name = ''
        for j in range(task):
            task_name += (envs[j] + '_')
        old_pol_path += task_name + '/policy_' + task_name+'.pth'
        
        
            #run behavoir cloning algorithm 
        behavior_cloning(
                    current_pseudo_path,
                    current_envs,
                    test_data_path = current_raw_path,
                    batch_size=100,
                    max_epoch=50,
                    what='baseline',
                    load_old_policy = load_old_pol,
                    old_pilicy = old_pol_path,
                    seed = seed
                )
    elif what == 'finetune':
        if task == 0:
                    load_old_pol = False 
        else: 
                    load_old_pol = True 
                # settingi the path for the policy
        old_pol_path = 'finetune/'+ str(seed) + '/'
        task_name = ''
        for j in range(task):
            task_name += (envs[j] + '_')
        old_pol_path += task_name + '/policy_' + task_name+'.pth'
        
        
                #run behavoir cloning algorithm 
        behavior_cloning(
                    current_pseudo_path,
                    current_envs,
                    test_data_path = current_raw_path,
                    batch_size=100,
                    max_epoch=50,
                    what='finetune',
                    load_old_policy = load_old_pol,
                    old_pilicy = old_pol_path,
                    seed = seed
                )


        
def setup_seed(seed):
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.determinstic = True 

if __name__ == '__main__':
      for seed in [11]:
        setup_seed(seed)

        for task in range(len(envs)):
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('seed',seed,'task: ', task)
            perform( task= task, what='train_policy', seed=seed)

print('YUHHHHHH')