"""Behavior-cloning policy (CNN): maps a camera image plus a one-hot task to a 4-number action.

Trained by supervised regression on the expert demonstrations (behavior cloning).
"""
import tensorflow as tf 
import math 
import numpy as np 
import  matplotlib.pyplot
from datasets.dataset import Dataset
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import utils.image_utils as imgUtils

#making the logs
import logging
# show only ERROR messages from TensorFlow's logger (hides warnings and info)
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

class CNN:
    """Policy network wrapper: holds the Keras model, training history and test results."""
    def __init__(self):

        """Build the CNN: image + task in, 4-number action out."""
        # two named inputs: a 480x480 RGB image and a one-hot task vector of length 3
        image = layers.Input((480,480,3) ,name="image")
        task = layers.Input((3,),name="task")


        #input layer
        # Conv2D(filters, kernel 5, same padding) with leaky ReLU finds features;
        # MaxPool2D(2, 2) halves height and width; three conv + pool blocks shrink 480 -> 60
        x= layers.Conv2D(32,5,padding='same',activation=tf.nn.leaky_relu)(image)

        #x = hidden layer
        x = layers.MaxPool2D(2,2)(x)
        x= layers.Conv2D(64,5,padding='same',activation=tf.nn.leaky_relu)(x)
        x = layers.MaxPool2D(2,2)(x)

        x= layers.Conv2D(64,5,padding='same',activation=tf.nn.leaky_relu)(x)
        x = layers.MaxPool2D(2,2)(x)

        flatten = layers.Flatten()(x)
        # flatten the feature maps and append the task one-hot so the policy knows which task to perform
        t = layers.Concatenate()([flatten,task])
        # hidden Dense layer, then the output layer with 4 numbers (the predicted action)
        y = layers.Dense(300)(t)
        action= layers.Dense(4)(y)

        # functional-API model with dictionary inputs keyed by the Input names above
        self.model = tf.keras.Model(
            inputs = {"image": image, "task" : task},
            outputs = action
        )
        # batch of 1 keeps memory low (480x480 images)
        self.BATCH_SIZE = 1
        self.history = None

        self.results = None

    def behavior_cloning(self,trainSet,valSet):
        
        """Train the policy on trainSet and validate on valSet; stores the Keras history in self.history."""
        dsTrain = self.loadDataSet(trainSet,self.BATCH_SIZE)
        dsVal = self.loadDataSet(valSet,self.BATCH_SIZE)
        # mean squared error loss on the action vector, mean absolute error (MAE) reported as a metric
        self.model.compile(optimizer='adam',loss= tf.keras.losses.MeanSquaredError(), metrics=['MAE'])
        # 5 epochs; the validation set is evaluated after every epoch
        history = self.model.fit(dsTrain, epochs=5, validation_data=dsVal)
        
        self.history = history

    def loadDataSet(self,dataset,batch):
        """Turn a Dataset into a streaming tf.data pipeline of ({image, task}, action) samples.

        Args:
            dataset: Dataset whose contents are ((image, action), task) tuples.
            batch: batch size.
        Returns: a batched tf.data.Dataset.
        """
        # pairs() is a generator: it yields one sample at a time and reads the image file only when needed
        def pairs():
           # nested tuple unpacking: each stored item is ((image, action), task)
           for (i, action),task in dataset.contents:
               # imageArray loads a path (or array) as uint8; dividing by 255 scales pixels to 0-1
               imageAarry = imgUtils.imageArray(i)
               yield(
                   {"image": imageAarry.astype(np.float32)/ 255.0, 
                    "task" :np.array(task, dtype=np.float32)},
                   np.array(action,dtype=np.float32)
               )
        
        # from_generator streams samples lazily, so the whole dataset is never held in memory;
        # output_signature declares the shape and dtype of each yielded value;
        # shuffle(30) mixes samples inside a buffer of 30; batch() groups them
        return tf.data.Dataset.from_generator(
                   pairs,
                   output_signature=(
                       {"image": tf.TensorSpec(shape=(480,480,3), dtype=tf.float32),
                        "task": tf.TensorSpec(shape=(3,), dtype=tf.float32)},
                       tf.TensorSpec(shape=(4,), dtype=tf.float32)
                   )
               ).shuffle(30).batch(batch)

    def identifyActions(self,img,task):

        """Print the policy's predicted action for one image and task (adds a batch dimension of 1)."""
        # expand_dims adds the batch dimension; the image is scaled from 0-255 to 0-1
        inputs = {
            "image": np.expand_dims(np.asanyarray(img, dtype=np.float32) / 255.0, axis = 0),
            "task": np.expand_dims(np.asanyarray(task, dtype=np.float32) , axis = 0)

        }

       # print(img)


        prediction = self.model.predict(inputs)
        print(f"prediction is {prediction}")
        
    def evaluateTraining(self,testSet):
        """Evaluate the policy on testSet; stores the metrics dict in self.results."""
        dsTest = self.loadDataSet(testSet,self.BATCH_SIZE)

        
        results = self.model.evaluate(dsTest,return_dict=True)
        self.results = results
        print(f"Results for policy Are: {results}")
        

    def plotTestData(self):
        """Plot the stored test MAE and loss as bar charts and save policy_Tested.png."""
        print(self.results.keys())
        if(self.results):
            acc = self.results['MAE']
            
            
            loss = self.results['loss']
    
            
            plt.figure(figsize=(8,8))
            plt.subplot(2,1,1)
            plt.bar('Training MAE',[acc])
          
            plt.legend(loc='lower right')
            plt.ylabel('MAE')
            plt.title('Test MAE')
            
            
            plt.subplot(2,1,2)
            plt.bar('Training Loss',[loss])
           
            plt.legend(loc='lower right')
            plt.ylabel('Loss')
            plt.title('Training and vAlidation Loss')
            plt.xlabel('epoch')
            
            # save the image
            plt.tight_layout()
            # delete the previous plot file, then save the new one (os.remove raises if the file does not exist)
            os.remove("policy_Tested.png")
            plt.savefig('policy_Tested.png', dpi=150)
            
            
            
            
            

    
    def plotTrainingData(self):
        """Plot training/validation MAE and loss per epoch and save policy_Trained.png."""
        print(self.history.history.keys())
        if(self.history):
            acc = self.history.history['MAE']
            val_acc = self.history.history['val_MAE']

            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']

            plt.figure(figsize=(8,8))
            plt.subplot(2,1,1)
            plt.plot(acc,label= 'Training Accuracy')
            plt.plot(val_acc, label= 'Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylabel('Accurarcy')
            plt.title('Training and vAlidation Accuracy')


            plt.subplot(2,1,2)
            plt.plot(loss,label= 'Training Loss')
            plt.plot(val_loss, label= 'Validation Loss')
            plt.legend(loc='lower right')
            plt.ylabel('Loss')
            plt.title('Training and vAlidation Loss')
            plt.xlabel('epoch')

            # save the image
            plt.tight_layout()
            # delete the previous plot file, then save the new one (os.remove raises if the file does not exist)
            os.remove("policy_Trained.png")
            plt.savefig('policy_Trained.png', dpi=150)