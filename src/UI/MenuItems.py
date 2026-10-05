
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
    def __init__(self,name,description,save):
        self.name = name 
        self.description = description 
        self.save = save

    def exec(self,models,dataset):
        print (f"executing menu item{self.name} ") 


# POLOCY MENU ITTEMS
class policyTrain(MenuItem):
    def __init__(self,save):
        super().__init__('Train Policy', 'Menu item is used to test the policy',save)

    def exec(self,models,dataset):
        #get the policyDataset 
        print('Attempting to make the behavior cloning')
        policy = models["policy"]
        behaveTrainSet = dataset["policy"][0]
        behaveValSet = dataset["policy"][1]
        

        print(len(behaveTrainSet.getItem(0)))
        print(type(behaveTrainSet.getItem(0)[0]))
        policy.behavior_cloning(behaveTrainSet,behaveValSet)
        print("training complete")
        
        policy.plotTrainingData()
        if(self.save):
            policy.model.save('policy_model.keras')
        

class policyTest(MenuItem):
    def __init__(self,save):
        super().__init__('Test Policy', 'Menu item is used to train the policy',save)

    def exec(self,models,dataset):
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
    def __init__(self,save):
        super().__init__('train GAN', 'Menu item is used to train the GAN',save)

    def exec(self,models,dataset):
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



        genTrainSet = WGAN.loadDataSet(genTrainSet,1)
        print("datasetLoaded")
        WGAN.plotHistory(WGAN.train_wagangp(gen,disc,genTrainSet,5),5)
        ("Training done ")

        if(self.save):
            gen.model.save('generator_model.keras')
            disc.model.save('critic_model.keras')
       

class ganTest(MenuItem):
    def __init__(self,save):
       
        super().__init__('test GAN', 'test GAN',save)

    def exec(self,models,dataset):
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
    def __init__(self,save):
        super().__init__('train Predictor', 'Menu item is used to train the predictor',save)

    def exec(self,models,dataset):
       
        # predictor 
        pred = models["pred"]
        predTrainSet = dataset["pred"][0]
        predValSet = dataset["pred"][1]
        predTestSet = dataset["pred"][2]

        pred.train(predTrainSet,predValSet,5,30)

        pred.plotTraining()

        pred.showPrediction(predTrainSet)
        if(self.save):
            pred.model.save('predictor_model.keras')




class predTest(MenuItem):
    def __init__(self,save):
        super().__init__('test Predictor', 'Menu item is used to test the Predictor',save)

    def exec(self,models,dataset):
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
    def __init__(self,save,fullDs):
        super().__init__("Train CRIL", "learn Tasks sequentially with genrated replay", save)

        self.fullDs = fullDs 

    def exec(self,models,dataset):
        trainEnd = 80 if self.fullDs ==1 else 3 
        valEnd = 90 if self.fullDs == 1 else 5 
        testEnd = 100 if self.fullDs == 1 else 7

        reByTask = cril.makeRealByTask(listUtils.TASKS_CONST,1,trainEnd,trainEnd+1,valEnd)

        testByTask = cril.makeRealByTask(listUtils.TASKS_CONST,valEnd + 1,testEnd,trainEnd+1,valEnd)

        cril.train_tasks(
            taskOrder= listUtils.TASKS_CONST,
            realByTask= reByTask,
            policy= models["policy"],
            gen = models["GAN"][0],
            disc = models["GAN"][1],
            pred = models["pred"],
            testByTask=testByTask

        )