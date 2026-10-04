import utils.meta_wrld_utils as metaUtils
import utils.csv_utils as csvUtils
import utils.image_utils as imgUtils
import utils.dataset_utils as dsUtils
from models.behaviorcloning import CNN
import PIL.Image as Image
import utils.plotter_utils as plotter
import models.wgangp as WGAN
import models.predictor as pred
import argparse 


import UI.Menu as menu
# get the args for either saving the model or loading the entireDataSet
parser = argparse.ArgumentParser()
parser.add_argument("fullDs", help="load the full dataset or not", type= int, choices= [0,1])
parser.add_argument("-l","--loadModels", action="store_true",help="Load the most recent models")
parser.add_argument("-s","--save",action="store_true", help = "this will save the models that you have training for this")
parser.add_argument("-t","--trainModels", action="store_true", help = " when this is enabled it means that the system will just train and not do anything at all besides train the models and store them")
args = parser.parse_args()

fullDs = args.fullDs
save = args.save if  not args.trainModels else True 
loadModels = args.loadModels 
trainModels = args.trainModels


menu = menu.Menu(fullDs,loadModels,save,trainModels)
menu.displayMenu()
