import tensorflow as tf 
import math 
import numpy as np 
import  matplotlib.pyplot
from datasets.dataset import Dataset
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import os

class Predictor:
    def __init__(self):
        image = layers.Input((480,480,3), name= "image")
        action = layers.Input((4,), name= "action")
        task = layers.Input((3,), name = 'task')
        # encoding Image
        x = layers.Conv2D(32,5,strides=2,padding="same",activation=tf.nn.relu)(image)
        x = layers.Conv2D(64,5,strides=2, padding='same',activation=tf.nn.relu)(x)
        x = layers.Conv2D(64,5,strides=2, padding='same',activation=tf.nn.relu)(x)
        #encoding Action
        a = layers.Dense(8)(action)
        a= layers.RepeatVector(60*60)(a)
        a= layers.Reshape((60,60,8))(a)

        #encoding task 
        t = layers.Dense(8)(task)
        t= layers.RepeatVector(60*60)(t)
        t= layers.Reshape((60,60,8))(t)

        shared = layers.Concatenate()([x,a,t])

        #Decode the image
        y = layers.Conv2DTranspose(64,5,strides=2,padding="same",activation=tf.nn.relu)(shared)
        y = layers.Conv2DTranspose(32,5,strides=2,padding="same",activation=tf.nn.relu)(y)
        nextImage = layers.Conv2DTranspose(3,5,strides=2,padding="same", activation='sigmoid',name="nextImage")(y)

        self.model = tf.keras.Model(
            inputs={"image":image, "action": action, "task":task},
            outputs={"nextImage":nextImage}
        )

        print(self.model.summary())

        self.history = None
        self.results = None

    def train(self,dataset,val, epochs,batch):
        dstrain = self.loadDataSet(dataset=dataset,batch=batch)
        dsVal = self.loadDataSet(val,batch=batch)

        self.model.compile(
            optimizer= tf.keras.optimizers.Adam(1e-4),
            loss={"nextImage": tf.keras.losses.MeanSquaredError()},
            metrics= {"nextImage" :[tf.keras.metrics.MeanSquaredError()]}
        )


        self.history = self.model.fit(
            dstrain,
            validation_data= dsVal,
            epochs=epochs
        )
        return self.history


    def plotTestResults(self):
        print(self.results.keys())
        if(self.results):
            acc = self.results['mean_squared_error']
        
        
            loss = self.results['loss']
        
        
            plt.figure(figsize=(8,8))
            plt.subplot(2,1,1)
            plt.bar(acc,label= 'Test mean_squared_error,',height= 10)
        
            plt.legend(loc='lower right')
            plt.ylabel('mean_squared_error')
            plt.title('Testing mean_squared_error')
        
        
            plt.subplot(2,1,2)
            plt.bar(loss,label= 'Training Loss',height= 10)
        
            plt.legend(loc='lower right')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.xlabel('epoch')
        
            # save the image
            plt.tight_layout()
            plt.savefig('predictor_Tested.png', dpi=150)

    def plotTraining(self):
        print(self.history.history.keys())
        if(self.history):
            acc = self.history.history['mean_squared_error']
            val_acc = self.history.history['val_mean_squared_error']
            
            loss = self.history.history['loss']
            val_loss = self.history.history['val_loss']
            
            plt.figure(figsize=(8,8))
            plt.subplot(2,1,1)
            plt.plot(acc,label= 'Training mean_squared_error')
            plt.plot(val_acc, label= 'val_mean_squared_errory')
            plt.legend(loc='lower right')
            plt.ylabel('Accurarcy')
            plt.title('Training and val_mean_squared_error')
            
            
            plt.subplot(2,1,2)
            plt.plot(loss,label= 'Training Loss')
            plt.plot(val_loss, label= 'Validation Loss')
            plt.legend(loc='lower right')
            plt.ylabel('Loss')
            plt.title('Training and vAlidation Loss')
            plt.xlabel('epoch')
            
            # save the image
            plt.tight_layout()
            plt.savefig('predictor_Trained.png', dpi=150)


    def evaluateOnTest(self,testSet, batch):
        dsTest = self.loadDataSet(testSet, batch)

        results = self.model.evaluate(dsTest,return_dict=True)
        self.results = results
        print(f"Results for predictor are: {results}")

    def showPrediction(self,dataset,index=0):
        image,action,actualNext,task = dataset.getItem(index)
        inputs = {
            "image":np.expand_dims(np.asarray(image,dtype=np.float32)/255.0, axis=0),
            "action": np.expand_dims(np.asarray(action,dtype=np.float32), axis=0),
            "task":  np.expand_dims(np.asarray(task,dtype=np.float32), axis=0)

        }

        output = self.model.predict(inputs,verbose=0)
        predictedNext = output["nextImage"][0]

        fig,axes = plt.subplots(1,3,figsize=(15,5))
        axes[0].imshow(image)
        axes[0].set_title("currentframe")
        axes[1].imshow(np.clip(predictedNext,0,1))
        axes[1].set_title("predicted next frame")
        axes[2].imshow(actualNext)
        axes[2].set_title("actual next frame")

        for ax in axes:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('predictor_monitor.png',dpi=150)




    def loadDataSet(self,dataset,batch):
        def samples():
            for image,action,nextImage, task in dataset.contents:
                inputs = {
                    "image": np.asarray(image,np.float32)/255.0,
                    "action":np.asarray(action,np.float32),
                    "task":np.asarray(task,np.float32)
                }
                targets = {
                    "nextImage" :np.asarray(nextImage,np.float32)/255.0,
                }
                yield inputs, targets 


        return tf.data.Dataset.from_generator(
            samples,
            output_signature=(
                {
                    "image": tf.TensorSpec((480,480,3),tf.float32),
                    "action": tf.TensorSpec((4,),tf.float32),
                    "task": tf.TensorSpec((3,),tf.float32)

                },
                {
                    "nextImage": tf.TensorSpec((480,480,3),tf.float32)
                
                }
            )
        ).shuffle(30).batch(batch)

    
def predict_images():
    None 
