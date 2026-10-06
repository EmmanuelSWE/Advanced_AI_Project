"""Menu items: one small class per action the console menu can run.

Every item inherits MenuItem and overrides exec(models, dataset); the Menu calls exec() on whichever
item was picked (polymorphism). `models` and `dataset` are the dicts built in UI/Menu.py.
"""

import utils.meta_wrld_utils as metaUtils
import utils.csv_utils as csvUtils
import utils.image_utils as imgUtils
import utils.dataset_utils as dsUtils
from models.behaviorcloning import CNN
import PIL.Image as Image
import utils.plotter_utils as plotter
import models.wgangp as WGAN
import models.predictor as pred
import models.cril_flow as cril
import utils.const_list_utils as listUtils

class MenuItem: 
    """Base class for all menu entries; holds a name, a description and the save flag."""
    def __init__(self,name,description,save):
        """Store the title and description shown in the menu and whether to save models afterwards."""
        self.name = name 
        self.description = description 
        self.save = save

    def exec(self,models,dataset):
        """Default action; every subclass overrides this."""
        print (f"executing menu item{self.name} ") 


# POLOCY MENU ITTEMS
class policyTrain(MenuItem):
    """Menu item: trains the policy (behavior cloning) and plots its curves."""
    def __init__(self,save):
        # super().__init__ runs MenuItem.__init__ to set name, description and save
        super().__init__('Train Policy', 'Menu item is used to test the policy',save)

    def exec(self,models,dataset):
        """Fit the policy on the train and validation splits, plot the curves, optionally save policy_model.keras."""
        #get the policyDataset 
        print('Attempting to make the behavior cloning')
        policy = models["policy"]
        behaveTrainSet = dataset["policy"][0]
        behaveValSet = dataset["policy"][1]
        

        # debug prints: how many fields one sample has and the type of its first field
        print(len(behaveTrainSet.getItem(0)))
        print(type(behaveTrainSet.getItem(0)[0]))
        policy.behavior_cloning(behaveTrainSet,behaveValSet)
        print("training complete")
        
        policy.plotTrainingData()
        if(self.save):
            policy.model.save('policy_model.keras')
        

class policyTest(MenuItem):
    """Menu item: evaluates the policy on the test split and plots the result."""
    def __init__(self,save):
        super().__init__('Test Policy', 'Menu item is used to train the policy',save)

    def exec(self,models,dataset):
        """Evaluate the policy on the test split and save the result plot."""
        #get the policyDataset 
        print('Attempting to test')
        policy = models["policy"]
        behaveTrainSet = dataset["policy"][0]
        behaveValSet = dataset["policy"][1]
        behaveTestSet = dataset["policy"][2]

        policy.evaluateTraining(behaveTestSet)
        policy.plotTestData()
        print("ttesting done")
        



# GAN MENU ITEMS
class ganTrain(MenuItem):
    """Menu item: trains the WGAN-GP image generator."""
    def __init__(self,save):
        super().__init__('train GAN', 'Menu item is used to train the GAN',save)

    def exec(self,models,dataset):
        """Train the GAN for 5 epochs on the GAN train split, plot the losses, optionally save generator and critic."""
        print("Training the gan")
        #get the policyDataset 
       ## okay now the gan 
        gen = models["GAN"][0]
        disc = models["GAN"][1]
        genTrainSet = dataset["GAN"][0]
        genValSet = dataset["GAN"][1]
        genTestSet = dataset["GAN"][2]
        print("content allocated")
        #gen = WGAN.Generator()
        #disc = WGAN.Discrimintator()

        # load the dataset and train 



        # wrap the dataset in a streaming tf.data pipeline with batch size 1 (keeps memory low)
        genTrainSet = WGAN.loadDataSet(genTrainSet,1)
        print("datasetLoaded")
        # train_wagangp returns the loss history (5 epochs); plotHistory draws and saves it
        WGAN.plotHistory(WGAN.train_wagangp(gen,disc,genTrainSet,5),5)
        # a bare string expression: it has no effect and prints nothing
        ("Training done ")

        if(self.save):
            gen.model.save('generator_model.keras')
            disc.model.save('critic_model.keras')
       

class ganTest(MenuItem):
    """Menu item: runs the GAN with a different batch size and plots the losses."""
    def __init__(self,save):
       
        super().__init__('test GAN', 'test GAN',save)

    def exec(self,models,dataset):
        """Despite the name, this calls the same training routine as ganTrain (batch size 4, 5 epochs) and plots the losses.
        It does not compute a separate test metric and does not save anything.
        """
        print("Testing the gan")
        #get the policyDataset 
        ## okay now the gan 
        gen = models["GAN"][0]
        disc = models["GAN"][1]
        genTrainSet = dataset["GAN"][0]
        genValSet = dataset["GAN"][1]
        genTestSet = dataset["GAN"][2]
        #gen = WGAN.Generator()
        #disc = WGAN.Discrimintator()
        
        # load the dataset and train 
        
        genTrainSet = WGAN.loadDataSet(genTrainSet,4)
        
        WGAN.plotHistory(WGAN.train_wagangp(gen,disc,genTrainSet,5),5)



# PREDICTOR MENU ITEMS 

class predTrain(MenuItem):
    """Menu item: trains the next-frame predictor."""
    def __init__(self,save):
        super().__init__('train Predictor', 'Menu item is used to train the predictor',save)

    def exec(self,models,dataset):
       
        """Train the predictor, plot the curves, save an example prediction picture, optionally save predictor_model.keras."""
        # predictor 
        pred = models["pred"]
        predTrainSet = dataset["pred"][0]
        predValSet = dataset["pred"][1]
        predTestSet = dataset["pred"][2]

        # arguments: train set, validation set, epochs = 5, batch = 30
        pred.train(predTrainSet,predValSet,5,30)

        pred.plotTraining()

        # saves a side-by-side picture (current frame, predicted next frame, actual next frame) of sample 0
        pred.showPrediction(predTrainSet)
        if(self.save):
            pred.model.save('predictor_model.keras')




class predTest(MenuItem):
    """Menu item: evaluates the predictor on the test split."""
    def __init__(self,save):
        super().__init__('test Predictor', 'Menu item is used to test the Predictor',save)

    def exec(self,models,dataset):
        """Show one prediction, evaluate on the test split (batch 30) and plot the result."""
        # predictor 
        pred = models["pred"]
        predTrainSet = dataset["pred"][0]
        predValSet = dataset["pred"][1]
        predTestSet = dataset["pred"][2]
        
        pred.showPrediction(predTrainSet)
        
        
        #test
        pred.evaluateOnTest(predTestSet,30)
        pred.plotTestResults()



class crilTrain(MenuItem):
    """Menu item: runs the whole CRIL loop, learning the tasks one after another with generated replay."""
    def __init__(self,save,fullDs):
        super().__init__("Train CRIL", "learn Tasks sequentially with genrated replay", save)

        self.fullDs = fullDs 

    def exec(self,models,dataset):
        """Build per-task real data, run cril.train_tasks and optionally save the four models.

        See models/cril_flow.py for the loop itself.
        """
        # demonstration-number ranges: small dataset = demos 1-3 train, 4-5 validation, 6-7 test;
        # full dataset = demos 1-80 train, 81-90 validation, 91-100 test
        valStart = 81 if self.fullDs == 1 else 4
        trainEnd = 80 if self.fullDs ==1 else 3 
        valEnd = 90 if self.fullDs == 1 else 5 
        testEnd = 100 if self.fullDs == 1 else 7

        # real (non-generated) data per task, split into train and validation
        reByTask = cril.makeRealByTask(listUtils.TASKS_CONST,1,trainEnd,valStart,valEnd)

        # same structure for the test demos; only the policy and predictor test sets are used later
        testByTask = cril.makeRealByTask(listUtils.TASKS_CONST,valEnd + 1,testEnd,valStart,valEnd)

        # learn the tasks in order; earlier tasks are replayed with generated data
        cril.train_tasks(
            taskOrder= listUtils.TASKS_CONST,
            realByTask= reByTask,
            policy= models["policy"],
            gen = models["GAN"][0],
            disc = models["GAN"][1],
            pred = models["pred"],
            testByTask=testByTask

        )

        # save the four models in the current folder (file names are prefixed with 'kaggle_')
        if(self.save):
            models['policy'].model.save('kaggle_policy_model.keras')
            models['GAN'][0].model.save('kaggle_generator_model.keras')
            models['GAN'][1].model.save('kaggle_critic_model.keras')
            models['pred'].model.save('kaggle_predcitor_model.keras')