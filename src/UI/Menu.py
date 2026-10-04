
import utils.meta_wrld_utils as metaUtils
import utils.csv_utils as csvUtils
import utils.image_utils as imgUtils
import utils.dataset_utils as dsUtils
from models.behaviorcloning import CNN
import PIL.Image as Image
import utils.plotter_utils as plotter
import models.wgangp as WGAN
import models.predictor as pred
import UI.MenuItems as Items
import tensorflow as tf 

class Menu: 
    def __init__(self,fullDs,loadModels,save,trainModels):
        self.fullDs = fullDs
        self.loadModels = loadModels
        self.trainModels = trainModels
        self.save = save


        print(f"loading menu Items")
        self.menuItems = [Items.policyTrain(save),
                          Items.policyTest(save),
                          Items.ganTrain(save),
                          Items.ganTest(save),
                          Items.predTrain(save),
                          Items.predTest(save)]

        
        trainEnd = 80 if fullDs == 1 else 3 
        valEnd = 90 if fullDs == 1 else 5 
        testEnd = 100 if fullDs == 1 else 7
        
        self.dataSets = { "policy": [
            dsUtils.makeDataset('test',1,trainEnd,2),
            dsUtils.makeDataset('test',trainEnd +1,valEnd,2),
            dsUtils.makeDataset('test',valEnd+1,testEnd,2)
        ],
                         "GAN":[
                             dsUtils.makeDataset('test',1,trainEnd,1),
                             dsUtils.makeDataset('test',trainEnd +1,valEnd,1),
                             dsUtils.makeDataset('test',valEnd+1,testEnd,1)
                         ],
                         "pred": [dsUtils.makeDataset('test',1,trainEnd,3),
                                     dsUtils.makeDataset('test',trainEnd +1,valEnd,3),
                                     dsUtils.makeDataset('test',valEnd+1,testEnd,3)]}


        self.models = {"policy": CNN(),
                       "GAN": [WGAN.Generator(),
                              WGAN.Discrimintator() ],
                       "pred": pred.Predictor() }
        
        if(loadModels):
            self.models["policy"].model = tf.keras.load_model("policy_model.keras")
            self.models["GAN"][0].model = tf.keras.load_model("generator_model.keras")
            self.models["GAN"][1].model = tf.keras.load_model("critic_model.keras")
            self.models["pred"].model = tf.keras.load_model("predictor_model.keras")

        
        print(f"Items Loaded")


    def showOpener(self):
         print(f"HELL USER WELCOME TO THE MENU"
)
    def showMenuItems(self):
        for i, item in  enumerate(self.menuItems):
            print(f"[{i +1}]  --- : {item.name}")
            print(f"[Description] ---: {item.description}")
 
    
    def displayMenu(self):
        if(self.trainModels):
            print("IMMEDIETLEYTRAINING AND SAVING THE MODELS DOING NOTHING ELSE")
            self.menuItems[0].exec(self.models,self.dataSets)
            self.menuItems[2].exec(self.models,self.dataSets)
            self.menuItems[4].exec(self.models,self.dataSets)
            return
        blContinue = True
        while(blContinue):
            self.showOpener()
            self.showMenuItems()
            userCommand = int(input("Please input a Menu Choice"))
            if(userCommand == 0):
                blContinue = False 
                print("Endinf programe")
            elif (1 <= userCommand <= len(self.menuItems)):
                self.menuItems[userCommand -1].exec(self.models,self.dataSets)
            else: 
                print("Selected wrong menu item please try again")
