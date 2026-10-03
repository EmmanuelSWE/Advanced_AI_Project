import utils.meta_wrld_utils as metaUtils
import utils.csv_utils as csvUtils
import utils.image_utils as imgUtils
import utils.dataset_utils as dsUtils
from models.behaviorcloning import CNN
import PIL.Image as Image
import utils.plotter_utils as plotter
import models.wgangp as WGAN




#metaUtils.runAndStoreDemonstrations()


# laod the image
#imageTest = imgUtils.loadImageFromPath('../DATASET_generations/drawer-open-v3/drawer-open-v3_step0_id2.png')


#print(imgUtils.getSize(imageTest))
#print(imageTest.mode)


# create the dataset for generator, policy and predictor
genTrainSet = dsUtils.makeDataset('test',1,3,1)
genValSet = dsUtils.makeDataset('test',4,5,1)
genTestSet = dsUtils.makeDataset('test',6,7,1)
print("generator datasets made")

behaveTrainSet = dsUtils.makeDataset('val',1,3,2)
behaveValSet = dsUtils.makeDataset('val',4,5,2)
behaveTestSet = dsUtils.makeDataset('val',6,7,2)
print("policy datasets made")

predTrainSet = dsUtils.makeDataset('train',1,3,3)
predValSet = dsUtils.makeDataset('train',4,5,3)
predTestSet = dsUtils.makeDataset('train',6,7,3)
print("predictor datasets made")


print('Attempting to make the behavior cloning')
policy = CNN()

policy.behavior_cloning(behaveTrainSet,behaveValSet)
print("training complete now doing testing")
policy.evaluateTraining(behaveTestSet)

policy.plotTrainingData()
policy.plotTestData()
# test it out => 
#img,_ = behaveTestSet.getItem(0)
#Image.open(img)

#policy.identifyActions(img)


## okay now the gan 
gen = WGAN.Generator()
disc = WGAN.Discrimintator()

# load the dataset and train 

genTrainSet = WGAN.loadDataSet(genTrainSet,4)

WGAN.train_wagangp(gen,disc,genTrainSet,5)
