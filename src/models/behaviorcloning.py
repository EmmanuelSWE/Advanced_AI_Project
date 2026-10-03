import tensorflow as tf 
import math 
import numpy as np 
import  matplotlib.pyplot
from datasets.dataset import Dataset
import matplotlib.pyplot as plt

#making the logs
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

class CNN:
    def __init__(self):
        self.BATCH_SIZE = 10
        self.input = tf.keras.layers.Conv2D(32, (5,5), activation=tf.nn.leaky_relu, padding='same',input_shape=(480,480,3),)
        self.m1 = tf.keras.layers.MaxPool2D(2,2)
        self.hidden1 = tf.keras.layers.Conv2D(64, (5,5),activation=tf.nn.leaky_relu,padding='same')
        self.m2=tf.keras.layers.MaxPool2D(2,2)
        self.hidden2 = tf.keras.layers.Conv2D(64, (5,5),activation=tf.nn.leaky_relu,padding='same')
        self.m3=tf.keras.layers.MaxPool2D(2,2)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden3= tf.keras.layers.Dense(300)
        self.output= tf.keras.layers.Dense(4)
        self.history = None

        self.model = tf.keras.models.Sequential([self.input,self.m1,self.hidden1,self.m2,self.hidden2,self.m3,self.flatten,self.hidden3,self.output])
        self.results = None

    def behavior_cloning(self,trainSet,valSet):
        
        dsTrain = self.loadDataSet(trainSet,self.BATCH_SIZE)
        dsVal = self.loadDataSet(valSet,self.BATCH_SIZE)
        self.model.compile(optimizer='adam',loss= tf.keras.losses.MeanSquaredError(), metrics=['MAE'])
        history = self.model.fit(dsTrain, epochs=5, steps_per_epoch=math.ceil(30/self.BATCH_SIZE),validation_data=dsVal)
        self.history = history

    def loadDataSet(self,dataset,batch):
        def pairs():
           for i, action in dataset.contents:
               yield(
                   np.array(i, dtype=np.float32) / 255.0,
                   np.array(action,dtype=np.float32)
               )
        
        return tf.data.Dataset.from_generator(
                   pairs,
                   output_signature=(
                       tf.TensorSpec(shape=(480,480,3), dtype=tf.float32),
                       tf.TensorSpec(shape=(4,), dtype=tf.float32)
                   )
               ).shuffle(30).batch(batch)

    def identifyActions(self,img):
        img = np.asanyarray(img, dtype=np.float32) / 255
        img = np.expand_dims(img, axis = 0)

       # print(img)


        prediction = self.model.predict(img)
        print(f"prediction is {prediction}")
        
    def evaluateTraining(self,testSet):
        dsTest = self.loadDataSet(testSet,self.BATCH_SIZE)

        
        results = self.model.evaluate(dsTest,return_dict=True)
        self.results = results
        print(f"Results Are: {results}")
        

    def plotTestData(self):
        print(self.results.keys())
        if(self.results):
            acc = self.results['MAE']
            
            
            loss = self.results['loss']
    
            
            plt.figure(figsize=(8,8))
            plt.subplot(2,1,1)
            plt.bar(acc,label= 'Training Accuracy,',height= 10)
          
            plt.legend(loc='lower right')
            plt.ylabel('Accurarcy')
            plt.title('Training and vAlidation Accuracy')
            
            
            plt.subplot(2,1,2)
            plt.bar(loss,label= 'Training Loss',height= 10)
           
            plt.legend(loc='lower right')
            plt.ylabel('Loss')
            plt.title('Training and vAlidation Loss')
            plt.xlabel('epoch')
            
            # save the image
            plt.tight_layout()
            plt.savefig('policy_Tested.png', dpi=150)
            
            
            
            
            

    
    def plotTrainingData(self):
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
            plt.savefig('policy_Trained.png', dpi=150)





