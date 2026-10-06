import numpy as np 
import tensorflow as tf
from datasets.dataset import Dataset
import models.wgangp as WGAN
import utils.csv_utils as csvUtils
from collections import defaultdict
import utils.const_list_utils as listUtils
import utils.image_utils as imgUtils
from pathlib import Path
import json
import imageio.v3 as iio 

def make_trajectory(gen,policy,predictor,taskVector,step =30,noise=None):
    task = tf.convert_to_tensor(
        np.asarray(taskVector,dtype=np.float32).reshape(1,3)
    )
    if noise is None:
        noise = tf.random.normal((1,4))

    generated = gen.model(
        {"action": noise, "task": task}, training = False
    )

    image = tf.clip_by_value((generated + 1.0)/2.0, 0.0,1.0)

    trajec = [] 
    for _ in range(step):
        action = policy.model(
            {"image": image, "task": task}, training= False
        )

        nextImage = predictor.model(
            {"image":image, "action":action, "task":task},
            training = False
        )["nextImage"]

        trajec.append({
            "image": image[0].numpy(),
            "action": action[0].numpy(),
            "nextImage": nextImage[0].numpy(),
            "task": task[0].numpy()
        })

        image = nextImage

    return trajec



def pixels(image):
    return (np.clip(image,0,1) * 255).astype(np.uint8)

def combinedDataset(name,realDataset):
    result = Dataset(name)
    result.contents.extend(realDataset.contents)
    return result

def train_tasks(taskOrder, realByTask, policy, gen,disc, pred, testByTask):
    allResults = {}
    fixedNoise = tf.random.normal((1,4), seed=42)

    for taskNum, taskName in enumerate(taskOrder):
        real = realByTask[taskName]

        policyData = combinedDataset("policy",real["policy"])
        predData = combinedDataset("predictor", real["predictor"])
        ganData = combinedDataset("GAN", real["GAN"])

        
        for oldTask in taskOrder[: taskNum]:
                old = realByTask[oldTask]
                taskVector = old["taskVector"]

                for _ in range(old['numTrajec']):
                    trajec = make_trajectory(
                        gen,policy,pred,taskVector=taskVector, step=old["lenTrajec"]
                    )

                    first = trajec[0]
                    ganData.addToDataset((
                        pixels(first["image"]),
                        taskVector
                    ))

                    for step in trajec:
                        image = pixels(step["image"])
                        nextImage = pixels(step["nextImage"])
                        action = step["action"]

                        policyData.addToDataset(((image,action), taskVector))

                        predData.addToDataset((image,action,nextImage,taskVector))
        print(f"training Task {taskNum + 1} : {taskName}")
        policy.behavior_cloning(policyData,real["policyVal"])
        pred.train(predData, real["predVal"],epochs = 30, batch = 1)
        ganBatches = WGAN.loadDataSet(ganData,batch=1)
        WGAN.train_wagangp(gen,disc,ganBatches,epochs=5)
        allResults[taskName] = testLearnedTasks(taskOrder=taskOrder, learnedCount=taskNum + 1 , testByTask=testByTask,policy=policy,pred=pred)

        stageDir = Path(f"/kaggle/working/cril_results/task_{taskNum + 1}") 
        stageDir.mkdir(parents=True,exist_ok=True)

        policy.model.save(f'{stageDir}/policy_model.keras')
        pred.model.save(f'{stageDir}/predictor_model.keras')
        gen.model.save(f'{stageDir}/generator_model.keras')
        disc.model.save(f'{stageDir}/critic_model.keras')

        for learnedTask in taskOrder[:taskNum +1 ]:
            frameDir = stageDir/learnedTask
            frameDir.mkdir(exist_ok=True)

            trajec = make_trajectory(
                gen,policy,pred,
                taskVector=realByTask[learnedTask]["taskVector"],
                step = 30,
                noise= fixedNoise
            )

            iio.imwrite(f"{frameDir}/frame_000.png",pixels(trajec[0]["image"]))

            for i, transition in enumerate(trajec,start=1):
                iio.imwrite(f"{frameDir}/frame_{i:03d}.png",pixels(transition["nextImage"]))

        resultsPath = Path(f"/kaggle/working/cril_results/testMetrics.json")
        with resultsPath.open("w") as file:
            json.dump(allResults,file,indent=2)



    return allResults


def makeRealByTask(taskOrder,demoStart,demoEnd,valStart,valEnd):
    trainRows = csvUtils.readFile(demoEnd,demoStart)
    valRows = csvUtils.readFile(valEnd,valStart)

    result = {} 
    for taskName in taskOrder:
        trajec = defaultdict(list)

        for row in trainRows:
            if row[2] == taskName:
                trajec[row[1]].append(row)
        policyData = Dataset(f"{taskName}_policy")
        predData = Dataset(f"{taskName}_predictor")
        ganData = Dataset(f"{taskName}_gan")

        lengths = []

        for rows in trajec.values():
            rows.sort(key=lambda row:int(row[3]))
            lengths.append(len(rows))

            taskVector = listUtils.task_vector(rows[0])
            ganData.addToDataset((rows[0][4], taskVector))

            for index,row in enumerate(rows):
                image = imgUtils.loadImageFromPath(row[4])
                action = np.fromstring(row[5].strip('[]'),sep=" ",dtype=np.float32)

                policyData.addToDataset(((image,action),taskVector))

                if index + 1 < len(rows):
                    nextRow = rows[index+1]

                    if(int(nextRow[3]) == int(row[3]) + 1):
                        nextImage = imgUtils.loadImageFromPath(nextRow[4])
                        predData.addToDataset((image,action,nextImage,taskVector))

        policyVal = Dataset(f"{taskName}_policyVal")
        predVal = Dataset(f"{taskName}_predVal")

        valTrajec = defaultdict(list)
        for row in valRows:
            if row[2] == taskName:
                valTrajec[row[1]].append(row)

        for rows in valTrajec.values():
            rows.sort(key= lambda row:int(row[3]))
            taskVector = listUtils.task_vector(rows[0])

            for index, row in enumerate(rows):
                image = imgUtils.loadImageFromPath(row[4])
                action = np.fromstring(row[5].strip("[]"),sep= " ", dtype= np.float32)
                policyVal.addToDataset(((image,action),taskVector))

                if index +1 < len(rows):
                    nextRow = rows[index + 1]
                    if(int(nextRow[3]) == int(row[3]) + 1):
                        nextImage = imgUtils.loadImageFromPath(nextRow[4])
                        predVal.addToDataset((image,action,nextImage,taskVector))

        result[taskName] = {
            "policy" : policyData,
            "predictor": predData,
            "GAN": ganData,
            "policyVal": policyVal,
            "predVal": predVal,
            "taskVector": listUtils.task_vector(next(iter(trajec.values()))[0]),
            "numTrajec": len(trajec),
            "lenTrajec": min(lengths) if lengths else 0    
        } 

    return result


def testLearnedTasks(taskOrder, learnedCount,testByTask,policy,pred):
    results = {}

    

    for taskName in taskOrder[:learnedCount]:
        test = testByTask[taskName]
        policyTest = policy.loadDataSet( test['policy'], policy.BATCH_SIZE)
        predTest = pred.loadDataSet(
            test["predictor"],batch = 1
        )
        policyResult = policy.model.evaluate(
            policyTest, return_dict = True, verbose=0
        )
        predResult = pred.model.evaluate(
            predTest, return_dict = True, verbose = 0
        )

        results[taskName] = {
            "policy": policyResult,
            "predictor": predResult
        }

        print(f"Test After task {learnedCount}, {taskName}: {results[taskName]}")
    return results