"""Console menu: owns the models and datasets and runs the menu item the user picks (UI/MenuItems.py)."""

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
    """Menu that holds the shared models and the datasets and runs the chosen MenuItem."""
    def __init__(self,fullDs,loadModels,save,trainModels):
        """Create the models and menu items.

        Args:
            fullDs: 1 = full dataset, 0 = small dataset.
            loadModels: if True, load previously saved .keras models.
            save: if True, menu items save their models after running.
            trainModels: if True, only run CRIL training and exit (no menu).
        """
        self.fullDs = fullDs
        self.loadModels = loadModels
        self.trainModels = trainModels
        self.save = save

        # one entry per model, keyed by name; GAN is a [generator, critic] list
        self.models = {"policy": CNN(),
                       "GAN": [WGAN.Generator(),
                              WGAN.Discrimintator() ],
                       "pred": pred.Predictor() }
        
        # optionally replace the fresh networks with previously saved .keras files from the current folder
        if(loadModels):
            self.models["policy"].model = tf.keras.load_model("policy_model.keras")
            self.models["GAN"][0].model = tf.keras.load_model("generator_model.keras")
            self.models["GAN"][1].model = tf.keras.load_model("critic_model.keras")
            self.models["pred"].model = tf.keras.load_model("predictor_model.keras")

        print(f"loading menu Items")
        # the order of this list is the number shown in the menu (1-based)
        self.menuItems = [Items.policyTrain(save),
                          Items.policyTest(save),
                          Items.ganTrain(save),
                          Items.ganTest(save),
                          Items.predTrain(save),
                          Items.predTest(save),
                          Items.crilTrain(save,fullDs),
                          # last, so the numbers 1-7 of the other items stay the same
                          Items.hyperparamConfig(save)]

    
        
        # datasets are filled in displayMenu(), not here
        self.dataSets= {}
        
        print(f"Items Loaded")


    def showOpener(self):
         """Print the welcome line."""
         print(f"HELL USER WELCOME TO THE MENU"
)
    def showMenuItems(self):
        """Print the number, name and description of every menu item."""
        for i, item in  enumerate(self.menuItems):
            print(f"[{i +1}]  --- : {item.name}")
            print(f"[Description] ---: {item.description}")
 
    
    def displayMenu(self):
        """Run the menu loop until the user types 0.

        In train-only mode it runs CRIL training once and returns instead.
        """
        if(self.trainModels):
            print("IMMEDIETLEYTRAINING AND SAVING THE MODELS DOING NOTHING ELSE")
            self.dataSets = {}
            Items.crilTrain(self.save,self.fullDs).exec(self.models,self.dataSets)
            return
        else :
            # last demonstration number of the train / validation / test splits; the small dataset only uses demos 1-7 to keep runs short
            trainEnd = 40 if self.fullDs == 1 else 3 
            valEnd = 90 if self.fullDs == 1 else 5 
            testEnd = 100 if self.fullDs == 1 else 7
            # the same CSV rows are wrapped three ways, each as [train, val, test]:
            # type 1 = image only (GAN), type 2 = image + action (policy), type 3 = consecutive-frame pairs (predictor)
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
            
            
            
        # loop: show the options, read a number, run that item; 0 quits
        blContinue = True
        while(blContinue):
            self.showOpener()
            self.showMenuItems()
            # int() raises ValueError if the user types something that is not a number
            userCommand = int(input("Please input a Menu Choice"))
            if(userCommand == 0):
                blContinue = False 
                print("Endinf programe")
            elif (1 <= userCommand <= len(self.menuItems)):
                self.menuItems[userCommand -1].exec(self.models,self.dataSets)
            else: 
                print("Selected wrong menu item please try again")
