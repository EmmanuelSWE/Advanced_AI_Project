"""CSV helpers: create, append to and read the space-delimited dataset index file.

Each row is: imgID, demonstarionID, task, step, image path, action.
"""
import csv
from utils.const_list_utils import DATASET_PATH as dsPath
from utils.const_list_utils import DATASET_NAME_CONST as filePath 
from utils.const_list_utils import HEADING_CONST as heading
from utils.const_list_utils import remove
from pathlib import Path 
import itertools


# number of header rows; not used below (readFile skips the header with next())
headingCount =1 


def readFile(end, demonID):
    
    """Return the CSV rows whose demonstration number is between demonID and end (inclusive).

    Args: end (last demo number), demonID (first demo number); note that end comes first.
    Returns: list of rows, each a list of strings.
    """
    content= []
    # the CSV is under DATASET_PATH; newline='' is what the csv module asks for when opening files
    with open( f"{dsPath}/{filePath}", newline='') as csvfile:
        # space delimiter; quotechar lets the action text (which contains spaces) stay in one cell
        reader = csv.reader(csvfile,delimiter=' ', quotechar='"')
        # skip the header row
        next(reader)
        print(f"on demon {demonID}")
        counter = 0
        for row in reader:
            # row[4] is the image path, e.g. .../reach-v3_step3_id12.png: take the part after the last '_' ('id12.png'),
            # drop the extension and the 'id' to get the demonstration number (12)
            tokens = row[4].split("_")
            demoimg = tokens[-1].split(".")
            demoNum = int(demoimg[0].strip()[2:])
            # keep the row only if its demonstration number is in range
            if(demonID <= demoNum and demoNum <= end):
                content.append(row)
                #print(row)
       # print(content)
        return content

def writeToFile(name, content,mode = 'old'): 

    """Write one row to a space-delimited CSV file.

    Args: name (file path), content (list of cells), mode ('new' overwrites, anything else appends).
    """
    # conditional expression: 'w' (overwrite) for a new file, otherwise 'a' (append)
    openMode = 'w' if mode == 'new' else  'a'
    with open(name, openMode, newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(content) # conent will be and array of things



def createDataset():
    """Reset the dataset folder (delete and recreate it) and write only the header row."""
    # remove() deletes the whole folder, including any old CSV, so it starts clean
    remove(dsPath)
    Path(dsPath).mkdir(exist_ok=True, parents= True) # create the path 
        # write the heading
    print('emptying file if any content')
        
    print('heading written')
    writeToFile(f'{dsPath}/{filePath}', heading,'new')


def writeToDataset(contents): # contents has to be a 2d array
    """Append every row of `contents` (a list of rows) to the dataset CSV."""
    print(f'appeding to file {dsPath}/{filePath}')

    for i,line in enumerate(contents):
        writeToFile(f'{dsPath}/{filePath}',line)
    print('done writing to the dataset')
    
