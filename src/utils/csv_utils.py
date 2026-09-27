import csv
from utils.const_list_utils import DATASET_PATH as dsPath
from utils.const_list_utils import DATASET_NAME_CONST as filePath 
from utils.const_list_utils import HEADING_CONST as heading
from utils.const_list_utils import remove
from pathlib import Path 

def readFile(name):
    with open(name, newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=' ', quotechar='|')
        for row in reader:
            print(row)

def writeToFile(name, content,mode = 'old'): 

    openMode = 'w' if mode == 'new' else  'a'
    with open(name, openMode, newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(content) # conent will be and array of things



def createDataset():
    remove(dsPath)
    Path(dsPath).mkdir(exist_ok=True, parents= True) # create the path 
        # write the heading
    print('emptying file if any content')
        
    print('heading written')
    writeToFile(f'{dsPath}/{filePath}', heading,'new')


def writeToDataset(contents): # contents has to be a 2d array
    print(f'appeding to file {dsPath}/{filePath}')

    for i,line in enumerate(contents):
        writeToFile(f'{dsPath}/{filePath}',line)
    print('done writing to the dataset')
    
