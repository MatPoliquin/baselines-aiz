import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
import math
import tensorflow as tf
import gc
from py3nvml.py3nvml import *
from PIL import Image
import os

class TrainingBroadcast():
    def __init__(self):
        self.rewardmean = 0
        self.totaltimesteps = 0
        self.final_dim = (1080, 1920, 3)
        self.playedintro = False
        self.framelist = []
        self.env = None
        self.action = 0
        self.action_meaning = []
        self.action_prob = 0
        self.have_nn_info = False
        self.numNeuralNetParams = 0
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.stats_fontscale = 1.0
        self.stats_pos = (50,50)
        self.nn_pos = (0,450)
        self.env_id = ''
        self.alg = ''
        self.args = []
        self.UpdateNeuralNetFrameCount=0;
        self.final_inited = False;
        self.final = None
        self.total_params = 0
        self.rewardmeanList = []
        self.updateRewardGraph = True
        

        nvmlInit()
        #TODO determine which GPU is used
        self.gpu_handle = nvmlDeviceGetHandleByIndex(0)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        print('=============DIR PATH================')
        print(dir_path)
    

        path = os.path.join(dir_path, '../../data/broadcast_logo.jpg')
        print(path)
        img = Image.open(path)    
        dim = (200,200)
        self.logo = cv2.resize(np.array(img), dim, interpolation=cv2.INTER_AREA)

        #print(self.logo)
        #image.show()
        

    def set_env(self, env):
        #self.env = envNo module named 'tensorflow_datasets'

        #for i in range(0,36):
        #    print(self.env.unwrapped.get_action_meaning(i))
        return

    def set_action_meaning(self, actionlist):
        #self.env = env
        #for i in range(0,36):
        #    print(self.env.unwrapped.get_action_meaning(i))
        
        self.action_meaning = actionlist.copy()
        print('==============ACTION MEANINGS SET================')
        print('Size:%d' % len(self.action_meaning))
        for i in range(0,len(self.action_meaning)):
            print(self.action_meaning[i])
        print('=================================================')

    def set_action_taken(self, action):
        self.action = action

    def set_action_prob(self, logprob):
        self.action_prob = logprob

    def get_neuralnetwork_info(self):
        #print(tf.trainable_variables())
        print('==============TRAINABLE PARAMETERS================')
        self.total_params = 0
        for v in tf.trainable_variables():
            print(v)
            shape = v.get_shape()
            count = 1
            for dim in shape:
                count *= dim.value
            self.total_params += count
        print("Total Params:%d" % self.total_params)


        print(tf.Session.graph_def)
        print('==================================================')
        self.have_nn_info = True

    def playintro(self):
        introarray = []
        intro = np.zeros(shape=self.final_dim, dtype=np.uint8)
        cv2.putText(intro, "INTRO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1 ,2)

        for i in range(0,60):
            introarray.append(intro)

        self.playedintro = True

        return introarray
    
    def add_intro(self):
        #finalarray = []
        #if not self.playedintro:
        #    intro = self.playintro()
        #    finalarray = intro
   
        #finalarray.append(final)
        #finalarray.append(final)

        #self.framelist.clear()
        return

    def addframe(self, ob):
        self.framelist.append(ob)

    def show_probabilities(self, final):
        cv2.putText(final, "Probabilities", (0,200), self.font, 1.0, (255,255,255), 1 ,2)

    def LogRewardMean(self, rew):
        self.rewardmeanList.append(rew)
        self.updateRewardGraph = True

    def DrawTrainingInfo(self, final, PosX, PosY):
        if math.isnan(broadcast.rewardmean):
            broadcast.rewardmean = 0;
        if math.isnan(broadcast.totaltimesteps):
            broadcast.totaltimesteps = 0;

        PosX += 10
        cv2.putText(final, ("Game:             %s" % self.env_id), (PosX,PosY+15), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Algo:             %s" % self.alg), (PosX,PosY+30), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Neural Net:       %s" % self.args['network']), (PosX,PosY+75), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Trainable Params: %s" % self.numNeuralNetParams), (PosX,PosY+100), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Reward mean:      %d" % broadcast.rewardmean), (PosX,PosY+125), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Timesteps:        %d" % broadcast.totaltimesteps), (PosX,PosY+150), self.font, 1.0, (255,255,255), 1 ,2)


        

    def show_inputimage(self, final, img):
        dim = (84,84)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        input = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        final[500:dim[1]+500,0:dim[0]] = input

    def show_actions(self, final):
        cv2.putText(final, ("A:%s" % self.action_meaning[self.action]), (0,600), self.font, 0.5, (255,255,255), 1 ,2)

    
    def DrawHardwareInfo(self, final, posX, posY):
        #num_gpus = nvmlDeviceGetCount()
        #print("NUM GPUS: %d" % nvmlDeviceGetCount())


        #nv_util = nvmlDeviceGetUtilizationRates(self.gpu_handle)
        #print("UTILS: %d" % nv_util.gpu)

        #self.clear_screen(posX, posY, 100, 200)

        cv2.putText(final, ("INTEL E5 2667 v3 8C/16T"), (posX, posY), self.font, 1.0, (0,255,255), 1 ,2)
        cv2.putText(final, ("NVIDIA P106-100 6GB"), (posX, posY + 15), self.font, 1.0, (0,255,255), 1 ,2)
        cv2.putText(final, ("PCIE 1.0 16X"), (posX, posY + 30), self.font, 1.0, (0,255,255), 1 ,2)
        cv2.putText(final, ("32 GB RAM"), (posX, posY + 45), self.font, 1.0, (0, 255,255), 1 ,2)


    def DrawInputLayer(self, final, img, posX, posY):
        cv2.putText(final, ("Input"), (posX, posY), self.font, 2.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("84x84 greyscale"), (posX, posY + 15), self.font, 1.0, (0,255,0), 1 ,2)
        cv2.putText(final, ("Last 4 frames"), (posX, posY + 30), self.font, 1.0, (0, 255,0), 1 ,2)

        dim = (84,84)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        input = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        #final[500:dim[1]+500,0:dim[0]] = input

        imgPosX = posX
        imgPosY = posY + 50
        final[imgPosY:imgPosY+dim[1],imgPosX:imgPosX+dim[0]] = input
        imgPosY += dim[1] + 50
        cv2.putText(final, ("[...]"), (posX, imgPosY), self.font, 1.0, (0,255,0), 1 ,2)
        #final[imgPosY:imgPosY+dim[1],imgPosX:imgPosX+dim[0]] = input
        #imgPosY += dim[1] + 50
        #final[imgPosY:imgPosY+dim[1],imgPosX:imgPosX+dim[0]] = input
        #imgPosY += dim[1] + 50
        #final[imgPosY:imgPosY+dim[1],imgPosX:imgPosX+dim[0]] = input
    
    def DrawConvNetLayer(self, final, posX, posY, numFilters, filterSize, name, param_name, input_dim):
        cv2.putText(final, ("%s" % name), (posX, posY), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("%d filters" % (numFilters)), (posX, posY + 15), self.font, 1.0, (0,255,0), 1 ,2)
        cv2.putText(final, ("%dx%d" % (filterSize, filterSize)), (posX, posY + 30), self.font, 1.0, (0,255,0), 1 ,2)
        
        #cv2.rectangle(final, (posX + 100,posY + 480), (180, 560), (255,255,255))

        #test = tf.get_variable(param_name)
        #print(test)
        #print(test.shape)

        with tf.variable_scope(param_name, reuse=True) as conv_scope:
            hello = tf.get_variable('w', shape=[filterSize,filterSize,input_dim,numFilters])
            #print(hello)
            weights0 = tf.transpose(hello, [3,0,1,2])
            weights = weights0.eval()
            #print(weights.shape)

            num_channels = hello.shape.dims[3]
            filter_w = hello.shape.dims[0]
            filter_h = hello.shape.dims[1]

            x = posX
            y = posY + 40

            for channel in range(0,10):
                img = weights[channel,:,:,0]
                #print(img.eval())
                #img = img * 255.0

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                #img2 = []
                img2 = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

                dim = (filter_w*4,filter_h*4)
                upscaled = cv2.resize(img2, dim, interpolation=cv2.INTER_NEAREST)

          
                

                final[y:y+dim[1], x:x+dim[0]] = upscaled
                y+=dim[1] + 5

            
            cv2.putText(final, ("[...]"), (posX, y + 40), self.font, 1.0, (0,255,0), 1 ,2)
        

    def DrawHiddenLayer(self, final, posX, posY):
        cv2.putText(final, ("Hidden Layer"), (posX, posY), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("512 units"), (posX, posY + 30), self.font, 1.0, (0,255,0), 1 ,2)
        cv2.putText(final, ("Fully Connected"), (posX, posY + 15), self.font, 1.0, (0,255,0), 1 ,2)

        with tf.variable_scope('ppo2_model/pi/fc1', reuse=tf.AUTO_REUSE) as conv_scope:
            hello = tf.get_variable('w', shape=[3136,512])
            weights0 = tf.transpose(hello, [1,0])
            weights = weights0.eval()
            x = posX
            y = posY + 40
            for channel in range(0,3):
                #print(channel)
                img = np.reshape(weights[channel,:], (56,56))
            
                '''
                for img_x in range(0,img.shape[0]):
                    for img_y in range(0,img.shape[1]):
                        if img[img_x, img_y] > 0:
                            img[img_x, img_y] = 255
                        else:
                            img[img_x, img_y] = 0
                '''
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img2 = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
                dim = (56*2,56*2)
                upscaled = cv2.resize(img2, dim, interpolation=cv2.INTER_NEAREST)

                #x = (channel % 3) * (56 * 4)
                #y = int(channel / 3) * (56 * 4)
                
                final[y:y+dim[1], x:x+dim[0]] = upscaled
                y+=dim[1] + 5

            #del hello
            #del weights0
            #del weights
            #del img
            #del img2
            #gc.collect()

        cv2.putText(final, ("[...]"), (posX, y + 40), self.font, 1.0, (0,255,0), 1 ,2)
        

    def DrawOutputLayer(self, final, posX, posY):
        cv2.putText(final, ("Output"), (posX, posY), self.font, 2.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("36 Actions"), (posX, posY + 15), self.font, 1.0, (0,255,0), 1 ,2)
        cv2.putText(final, ("Prob:%f" % self.action_prob), (posX, posY + 30), self.font, 1.0, (0,255,0), 1 ,2)
        for i in range(0,len(self.action_meaning)):
            color = (255,255,255)
            if self.action == i:
                color = (0,255,0)
            cv2.putText(final, ("%s" % self.action_meaning[i]), (posX, posY + 55 + i*15), self.font, 1.0, color, 1 ,2)

        
        


    def DrawNeuralNetwork(self, final, img, posX, posY):

        #Input layer
        self.DrawInputLayer(final, img, posX, posY)


        if self.UpdateNeuralNetFrameCount == 0:
            #Conv net layer 1
            self.DrawConvNetLayer(final, posX + 150, posY, 32, 8, "Conv Layer 1", 'ppo2_model/pi/c1',4)

            #Conv net layer 2
            self.DrawConvNetLayer(final, posX + 300, posY, 64, 4, "Conv Layer 2", 'ppo2_model/pi/c2',32)
        
            #Conv net layer 3
            self.DrawConvNetLayer(final, posX + 450, posY, 64, 3, "Conv Layer 3", 'ppo2_model/pi/c3',64)
        
            #Hidden layer
            self.DrawHiddenLayer(final, posX + 600, posY)

            print('UPDATE CONV NETS!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            self.UpdateNeuralNetFrameCount = 1000

        
        #Output layer
        self.DrawOutputLayer(final, posX + 750, posY)

        #Draw Enclosing Rectangle
        #cv2.rectangle(final, (posX-10, posY), (700, 800), (0,255,0), 2)

        self.UpdateNeuralNetFrameCount -= 1

    def clear_screen(self, posX, posY, dimX, dimY):

        dim = (dimX, dimY, 3)
        clear = np.zeros(shape=dim, dtype=np.uint8)

        #print(posX)
        #print(posY)
        #print(dim[0])
        #print(dim[1])
        self.final[posX:posX+dim[0],posY:posY+dim[1]] = clear

    def DrawRewardGraph(self, posX, posY, width, height):
        fig = plt.figure(0)

        #mpl.rcParams['lines.linewidth'] = 4
        #mpl.rcParams['lines.color'] = 'g'
  
        

        #plt.style.

        plt.plot(self.rewardmeanList, color="green")

        #plt.style.context('dark_background')

        fig.set_facecolor('black')

        numYData = len(self.rewardmeanList)
        plt.xlim([0,numYData])
        #plt.tight_layout()
  
        plt.grid(True)
        plt.rc('grid', color='w', linestyle='solid')


        #fig.patch.set_visible(False)

        

        fig.set_size_inches(width/80, height/80, forward=True)

        
        

        #plt.box(False)

        #plt.xlabel('Timesteps') 
        #plt.ylabel('Reward') 
        #plt.title('Reward')
        #plt.show()
 
        #self.rewardmeanList.append(10000)

        ax = plt.gca()
        ax.set_facecolor("black")

        ax.tick_params(axis='x', colors='green')
        ax.tick_params(axis='y', colors='green')

        #for child in ax.get_children():
        #    if isinstance(child, mpl.spines.Spine):
        #        child.set_color('white')


        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        #for spine in plt.gca().spines.values():
        #    spine.set_visible(False)

        #ax.xaxis.label.set_color('red')
        #ax.set_facecolor("black")
        #ax.set_color("green")
        #ax.tick_params(axis='x', colors='blue')
        #ax.tick_params(axis='y', colors='red')

        #ax.lines.set_color('g')

        
        #draw buffer
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer, size = fig.canvas.print_to_buffer()
        image = np.fromstring(buffer, dtype='uint8').reshape(height, width, 4)
        self.final[posY:posY+height,posX:posX+width] = image[:,:,0:3]

        plt.close()

    def DrawGPUGraph(self, posX, posY, width, height):
        #x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32] 
        # corresponding y axis values 
        #y = [2,4,1]

        fig = plt.figure(0, facecolor='b')
  
        # plotting the points  
        plt.plot(self.rewardmeanList)

        numYData = len(self.rewardmeanList)
        plt.xlim([0,numYData])

        plt.axes.set_facecolor('b')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        #fig.patch.set_facecolor('black')
        # naming the x axis 
        #plt.xlabel('Timesteps') 
        # naming the y axis 
        #plt.ylabel('Reward') 
  
        # giving a title to my graph 
        #plt.title('Reward')
  
        # function to show the plot 
        #plt.show()

        #fig, ax = plt.subplots()

        
        fig.canvas.draw()

        

        #fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        #image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        buffer, size = fig.canvas.print_to_buffer()
        image = np.fromstring(buffer, dtype='uint8').reshape(height, width, 4)
        #print(buffer)
        self.final[posY:posY+height,posX:posX+width] = image[:,:,0:3]


        plt.close()

    def DrawLogo(self, posX, posY, width, height):

        self.final[posY:posY+height,posX:posX+width] = self.logo

    def set_gameframe(self, img):

        if not self.have_nn_info:
            self.get_neuralnetwork_info()


        h, w, c = img.shape
        dim = (w * 4,h * 4)
        upscaled = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

        if not self.final_inited:
            self.final = np.zeros(shape=self.final_dim, dtype=np.uint8)
            print('final inited!!!!!!!!!!!!!!!!!!!!!')
            self.final_inited = True

        start_x = self.final_dim[1] - (w*4) -1
        start_y = self.final_dim[0] - (h*4) -1
        self.final[start_y:dim[1]+start_y,start_x:dim[0]+start_x] = upscaled


        self.DrawLogo(0,0,self.logo.shape[0], self.logo.shape[1])
        self.DrawHardwareInfo(self.final, 200, 15)
        cv2.rectangle(self.final, (0, 0), (425, 200), (0,0,255), 4)

        self.DrawTrainingInfo(self.final, 0, 220)
        cv2.rectangle(self.final, (0, 215), (425, 425), (0,255,0), 4)
        
        self.DrawNeuralNetwork(self.final, img, 0, 475)

        #y_data =[5] * 32

        #print(y_data)

        if self.updateRewardGraph:
            self.DrawRewardGraph(400,50,430,300)
            self.updateRewardGraph = False

        
        return self.final


broadcast = TrainingBroadcast()

