"""CRIL continual-learning loop.

Builds per-task real data from the CSV, then learns the tasks one after another. Before each new task it
replays earlier tasks with generated data (generator -> policy -> predictor), trains the policy, predictor
and GAN on real + replayed data, and tests on every task learned so far.
"""
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
import gc
import shutil

def make_trajectory(gen,policy,predictor,taskVector,step =30,noise=None):
    """Generator that rolls out an imagined trajectory without the simulator.

    The GAN makes a first frame from noise + task; then for `step` steps the policy picks an action and the
    predictor imagines the next frame. Yields one transition dict per step.
    Args: gen, policy, predictor (wrappers with .model), taskVector (3-number one-hot), step (number of steps),
    noise (optional 1x4 noise; random if None).
    """
    # the model input must be a float32 tensor of shape (1, 3)
    task = tf.convert_to_tensor(
        np.asarray(taskVector,dtype=np.float32).reshape(1,3)
    )
    # no noise given: sample one random 4-number vector
    if noise is None:
        noise = tf.random.normal((1,4))

    # the generator turns (noise, task) into a first frame in [-1, 1]; training=False runs it in inference mode
    generated = gen.model(
        {"action": noise, "task": task}, training = False
    )

    # map the tanh range [-1, 1] to [0, 1], which the policy and predictor expect
    image = tf.clip_by_value((generated + 1.0)/2.0, 0.0,1.0)

    
    # each predicted frame becomes the input of the next step
    for _ in range(step):
        action = policy.model(
            {"image": image, "task": task}, training= False
        )

        nextImage = predictor.model(
            {"image":image, "action":action, "task":task},
            training = False
        )["nextImage"]

        # yield hands back one transition at a time, so the caller streams them instead of holding a full list
        yield ({
            "image": image[0].numpy(),
            "action": action[0].numpy(),
            "nextImage": nextImage[0].numpy(),
            "task": task[0].numpy()
        })

        # feed the predicted frame back in as the current frame
        image = nextImage




def pixels(image):
    """Convert a float image in [0, 1] to uint8 [0, 255] so it can be saved as PNG."""
    return (np.clip(image,0,1) * 255).astype(np.uint8)

def combinedDataset(name,realDataset):
    """Return a new Dataset with the same samples as realDataset.

    The list is copied (the samples are shared), so replay samples added later do not touch the real data.
    """
    result = Dataset(name)
    result.contents.extend(realDataset.contents)
    return result

def train_tasks(taskOrder, realByTask, policy, gen,disc, pred, testByTask):
    """Learn the tasks one after another with generative replay.

    Args:
        taskOrder: task names in learning order.
        realByTask: output of makeRealByTask.
        policy, gen, disc, pred: the model wrappers (CNN, Generator, Discrimintator, Predictor).
        testByTask: test data in the same layout as realByTask.
    Returns: dict task name -> test metrics measured after learning that task.
    """
    # allResults maps task name -> test metrics after that stage
    allResults = {}
    # seeded noise: the same value every run, so result images are comparable across stages
    fixedNoise = tf.random.stateless_normal((1, 4), seed=(42, 0))

    # taskNum is the 0-based position of the task in the learning order
    for taskNum, taskName in enumerate(taskOrder):
        real = realByTask[taskName]

        # start every stage from copies of this task's real data, so replay samples added later do not change realByTask
        policyData = combinedDataset("policy",real["policy"])
        predData = combinedDataset("predictor", real["predictor"])
        ganData = combinedDataset("GAN", real["GAN"])

        
            # Temporary replay frames for this training stage.
        # hardcoded Kaggle path: replay frames are written to disk (only paths are kept in memory) and deleted after the stage
        replayDir = Path(
            f"/kaggle/working/cril_replay/task_{taskNum + 1}"
        )

        # generative replay: for every earlier task, imagine new trajectories with gen + policy + pred and add them to this stage's training data
        for oldTask in taskOrder[:taskNum]:
            old = realByTask[oldTask]
            taskVector = old["taskVector"]

            for demoIndex in range(old["numTrajec"]):
                # one folder per old task and demonstration; :03d zero-pads the number (demo_007);
                # mkdir(parents=True, exist_ok=True) creates nested folders and does not fail if they exist
                frameDir = replayDir / oldTask / f"demo_{demoIndex:03d}"
                frameDir.mkdir(parents=True, exist_ok=True)

                # transitions is a generator: frames are produced lazily, one step at a time
                transitions = make_trajectory(
                    gen, policy, pred,
                    taskVector=taskVector,
                    step=old["lenTrajec"],
                )

                for frameIndex, transition in enumerate(transitions):
                    # file names are zero-padded to 4 digits; the next frame has index + 1
                    imagePath = frameDir / f"frame_{frameIndex:04d}.png"
                    nextPath = frameDir / f"frame_{frameIndex + 1:04d}.png"

                    # Write the starting frame once; each next frame becomes
                    # the current frame of the following transition.
                    if frameIndex == 0:
                        iio.imwrite(imagePath, pixels(transition["image"]))
                        ganData.addToDataset((str(imagePath), taskVector))

                    iio.imwrite(nextPath, pixels(transition["nextImage"]))

                    # Retain paths and the small action vector, not image arrays.
                    action = transition["action"]
                    policyData.addToDataset(
                        ((str(imagePath), action), taskVector)
                    )
                    predData.addToDataset(
                        (str(imagePath), action, str(nextPath), taskVector)
                    )

                # Release the final transition's arrays before the next demonstration.
                if old["lenTrajec"] > 0:
                    del transition
        print(f"training Task {taskNum + 1} : {taskName}")
        # train on real + replayed data; validation uses only the current task's real validation data
        policy.behavior_cloning(policyData,real["policyVal"])
        # predictor: epochs and batch size come from HYPERPARAMS (defaults 5 and 1)
        pred.train(predData, real["predVal"],epochs = listUtils.HYPERPARAMS['cril_pred_epochs'], batch = listUtils.HYPERPARAMS['cril_pred_batch'])
        # GAN data in a streaming pipeline (default batch size 1)
        ganBatches = WGAN.loadDataSet(ganData,batch=listUtils.HYPERPARAMS['cril_gan_batch'])
        # WGAN-GP on this stage's first frames (default 30 epochs)
        WGAN.train_wagangp(gen,disc,ganBatches,epochs=listUtils.HYPERPARAMS['cril_gan_epochs'])

        # checkpoints of this stage go under a hardcoded Kaggle path
        stageDir = Path(f"/kaggle/working/cril_results/task_{taskNum + 1}") 
        stageDir.mkdir(parents=True,exist_ok=True)

        policy.model.save(f'{stageDir}/policy_model.keras')
        pred.model.save(f'{stageDir}/predictor_model.keras')
        gen.model.save(f'{stageDir}/generator_model.keras')
        disc.model.save(f'{stageDir}/critic_model.keras')

        # test every task learned so far to see how earlier tasks hold up
        allResults[taskName] = testLearnedTasks(taskOrder=taskOrder, learnedCount=taskNum + 1 , testByTask=testByTask,policy=policy,pred=pred)


        # save an imagined trajectory per learned task (fixed noise, default 30 steps) to inspect the generated frames
        for learnedTask in taskOrder[:taskNum +1 ]:
            frameDir = stageDir/learnedTask
            frameDir.mkdir(exist_ok=True)

            transitions = make_trajectory(
                gen, policy, pred,
                taskVector=realByTask[learnedTask]["taskVector"],
                step=listUtils.HYPERPARAMS['cril_sample_steps'],
                noise=fixedNoise,
            )

            for i, transition in enumerate(transitions):
                if i == 0:
                    iio.imwrite(
                        frameDir / "frame_000.png",
                        pixels(transition["image"]),
                    )

                iio.imwrite(
                    frameDir / f"frame_{i + 1:03d}.png",
                    pixels(transition["nextImage"]),
                )

        # rewrite the metrics file after each stage; `with` closes the file automatically, indent=2 makes it readable
        resultsPath = Path(f"/kaggle/working/cril_results/testMetrics.json")
        with resultsPath.open("w") as file:
            json.dump(allResults,file,indent=2)

        # free memory: drop the large dataset objects and run garbage collection
        del policyData, predData, ganData, ganBatches
        gc.collect()

        # delete this stage's temporary replay frames
        if replayDir.exists():
            shutil.rmtree(replayDir)



    return allResults


def makeRealByTask(taskOrder,demoStart,demoEnd,valStart,valEnd):
    """Build the per-task data bundles from the CSV.

    Args: taskOrder (task names), demoStart/demoEnd (train demo numbers), valStart/valEnd (validation demo numbers).
    Returns: dict task name -> {policy, predictor, GAN, policyVal, predVal datasets, taskVector, numTrajec
    (number of demos), lenTrajec (length of the shortest demo)}.
    """
    # read the CSV rows of each demo-number range once (readFile takes the end first, then the start);
    # a row is a list of strings: imgID, demo id, task, step, image path, action
    trainRows = csvUtils.readFile(demoEnd,demoStart)
    valRows = csvUtils.readFile(valEnd,valStart)

    # result: task name -> bundle of datasets and info
    result = {} 
    for taskName in taskOrder:
        # group this task's rows by demonstration id; defaultdict(list) creates an empty list for a new key automatically
        trajec = defaultdict(list)

        for row in trainRows:
            # row[2] is the task name and row[1] the demonstration id
            if row[2] == taskName:
                trajec[row[1]].append(row)
        policyData = Dataset(f"{taskName}_policy")
        predData = Dataset(f"{taskName}_predictor")
        ganData = Dataset(f"{taskName}_gan")

        # lengths: number of steps of each demonstration
        lengths = []

        for rows in trajec.values():
            # order the rows of a demo by step number (the lambda is the sort key)
            rows.sort(key=lambda row:int(row[3]))
            lengths.append(len(rows))

            # one-hot task vector from the first row of the demo
            taskVector = listUtils.task_vector(rows[0])
            # GAN data: only the first frame of each demo, with its task vector
            ganData.addToDataset((rows[0][4], taskVector))

            for index,row in enumerate(rows):
                image = row[4]
                # the action is stored as text like [0.1 0.2 0.3 0.4]: strip the brackets and parse the space-separated numbers
                action = np.fromstring(row[5].strip('[]'),sep=" ",dtype=np.float32)

                # policy sample: ((image path, action), task vector)
                policyData.addToDataset(((image,action),taskVector))

                # a predictor sample needs the next frame, so it is only made when the next row is exactly the next step
                if index + 1 < len(rows):
                    nextRow = rows[index+1]

                    if(int(nextRow[3]) == int(row[3]) + 1):
                        nextImage = nextRow[4]
                        predData.addToDataset((image,action,nextImage,taskVector))

        # validation sets are built the same way, without GAN data
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
                image = row[4]
                action = np.fromstring(row[5].strip("[]"),sep= " ", dtype= np.float32)
                policyVal.addToDataset(((image,action),taskVector))

                if index +1 < len(rows):
                    nextRow = rows[index + 1]
                    if(int(nextRow[3]) == int(row[3]) + 1):
                        nextImage = nextRow[4]
                        predVal.addToDataset((image,action,nextImage,taskVector))

        # bundle used by train_tasks and testLearnedTasks
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
    """Evaluate the policy and predictor on the test data of the first `learnedCount` tasks.

    Returns: dict task name -> {'policy': metrics, 'predictor': metrics}.
    """
    results = {}

    

    # test every task learned so far
    for taskName in taskOrder[:learnedCount]:
        test = testByTask[taskName]
        # test sets are streamed with batch size 1
        policyTest = policy.loadDataSet( test['policy'], policy.BATCH_SIZE)
        predTest = pred.loadDataSet(
            test["predictor"],batch = 1
        )
        # evaluate returns a dict of loss and metrics (return_dict=True); verbose=0 hides the progress bar
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