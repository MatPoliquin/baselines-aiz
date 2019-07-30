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
from cpuinfo import get_cpu_info
import psutil

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
        self.gpuUtilStat = [0] * 200
        self.pcieUtilStat = [0] * 200
        self.UpdatePerfStatsFrameCount=0;
        self.cpuUsage = [0] * 200
        self.audio_rate = 0
        

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

        #for key, value in get_cpu_info().items():
        #    print("{0}: {1}".format(key, value))
        

    def set_env(self, env):
        #self.env = envNo module named 'tensorflow_datasets'

        #for i in range(0,36):
        #    print(self.env.unwrapped.get_action_meaning(i))
        return
    
    def set_audio_rate(self, audio_rate):
        self.audio_rate = audio_rate

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
        #print('added frame: %d' % len(self.framelist))

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
        cv2.putText(final, ("GAME:                 %s" % self.env_id), (PosX,PosY+15), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("ALGO:                 %s" % self.alg), (PosX,PosY+30), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("NEURAL NET:          %s" % self.args['network']), (PosX,PosY+45), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("TRAINABLE PARAMS:   %d float32" % self.total_params), (PosX,PosY+60), self.font, 1.0, (255,255,255), 1 ,2)
        #cv2.putText(final, ("REWARD MEAN:        %d" % broadcast.rewardmean), (PosX,PosY+75), self.font, 1.0, (255,255,255), 1 ,2)

        self.clear_screen(PosX + 100, PosY+75, 200, 15)
        cv2.putText(final, ("TIMESTEPS:          %d" % broadcast.totaltimesteps), (PosX,PosY+90), self.font, 1.0, (255,255,255), 1 ,2)


        

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
        cv2.putText(final, ("PCIE 1.1 16X"), (posX, posY + 30), self.font, 1.0, (0,255,255), 1 ,2)
        cv2.putText(final, ("32 GB DDR4"), (posX, posY + 45), self.font, 1.0, (0, 255,255), 1 ,2)

        posY += 100
        cv2.putText(final, ("OPENAI BASELINES 0.1.6"), (posX, posY), self.font, 1.0, (255,255,0), 1 ,2)
        cv2.putText(final, ("TENSORFLOW 1.14"), (posX, posY + 15), self.font, 1.0, (255,255,0), 1 ,2)
        cv2.putText(final, ("CUDA 10"), (posX, posY + 30), self.font, 1.0, (255,255,0), 1 ,2)
   


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
        #cv2.putText(final, ("Prob:%f" % self.action_prob), (posX, posY + 30), self.font, 1.0, (0,255,0), 1 ,2)
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
            self.UpdateNeuralNetFrameCount = 4000

        
        #Output layer
        self.DrawOutputLayer(final, posX + 750, posY)

        #Draw Enclosing Rectangle
        #cv2.rectangle(final, (posX-10, posY), (700, 800), (0,255,0), 2)

        self.UpdateNeuralNetFrameCount -= 1

    def clear_screen(self, posX, posY, width, height):

        #dim = (dimX, dimY, 3)
        #clear = np.zeros(shape=dim, dtype=np.uint8)

        #print(posX)
        #print(posY)
        #print(dim[0])
        #print(dim[1])
        self.final[posY:posY+height,posX:posX+width] = [0, 0, 0]

    def DrawRewardGraph(self, posX, posY, width, height):
        fig = plt.figure(0)

        plt.plot(self.rewardmeanList, color=(0,1,0))

        plt.title(("Reward Mean: %d" % broadcast.rewardmean), color=(0,1,0))

        fig.set_facecolor('black')

        numYData = len(self.rewardmeanList)
        plt.xlim([0,numYData])
        plt.tight_layout()
  
        plt.grid(True)
        plt.rc('grid', color='w', linestyle='solid')


        fig.set_size_inches(width/80, height/80, forward=True)

        ax = plt.gca()
        ax.set_facecolor("black")

        ax.tick_params(axis='x', colors='green')
        ax.tick_params(axis='y', colors='green')

        ax.get_xaxis().set_ticks([])


        #draw buffer
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer, size = fig.canvas.print_to_buffer()
        image = np.fromstring(buffer, dtype='uint8').reshape(height, width, 4)
        self.final[posY:posY+height,posX:posX+width] = image[:,:,0:3]

        plt.close()

    def DrawStatGraph(self, posX, posY, width, height, y_data, y_limit, title, color):
        fig = plt.figure(0)

        plt.plot(y_data, color=color)

        fig.set_facecolor('black')

        plt.title(title, color=color)

        numYData = len(y_data)
        plt.xlim([0,numYData])

        if y_limit:
            plt.ylim([0,y_limit])
        #plt.tight_layout()
  
        plt.grid(True)
        plt.rc('grid', color='w', linestyle='solid')


        fig.set_size_inches(width/80, height/80, forward=True)

        ax = plt.gca()
        ax.set_facecolor("black")


        ax.tick_params(axis='x', colors='green')
        ax.tick_params(axis='y', colors='green')

        ax.get_xaxis().set_ticks([])


        #draw buffer
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer, size = fig.canvas.print_to_buffer()
        image = np.fromstring(buffer, dtype='uint8').reshape(height, width, 4)
        self.final[posY:posY+height,posX:posX+width] = image[:,:,0:3]

        plt.close()

    def DrawLogo(self, posX, posY, width, height):

        self.final[posY:posY+height,posX:posX+width] = self.logo

    def DrawPerformanceStats(self):
        nv_util = nvmlDeviceGetUtilizationRates(self.gpu_handle)
        self.gpuUtilStat.append(nv_util.gpu)
        pcie_tx = nvmlDeviceGetPcieThroughput(self.gpu_handle,NVML_PCIE_UTIL_TX_BYTES)
        self.pcieUtilStat.append(pcie_tx)
        mem = nvmlDeviceGetMemoryInfo(self.gpu_handle)
        #psutil.cpu_percent(interval=1)
        cpuStat = psutil.cpu_percent()
        self.cpuUsage.append(cpuStat)

        



        

        test = 1
        self.clear_screen(950,30, 200, 65)
        cv2.putText(self.final, ("VRAM: %.0f MB" % (mem.used / (1024.0 * 1024.0))), (950, 45), self.font, 1.0, (255,255,255), 1 ,2)
        #cv2.putText(self.final, ("RAM:%d" % test), (950, 70), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(self.final, ("PCIE: %.3f MB/s" % (pcie_tx / 1024.0)), (950, 95), self.font, 1.0, (255,255,255), 1 ,2)

        #test1 = pcie_tx
        #print(test1 / 1024)

        #numGPUStats = len(gpuUtilStat)
        
        #keep at 200 samples
        self.gpuUtilStat = self.gpuUtilStat[1:len(self.gpuUtilStat)]
        self.pcieUtilStat = self.pcieUtilStat[1:len(self.pcieUtilStat)]
        self.cpuUsage = self.cpuUsage[1:len(self.cpuUsage)]

        if self.UpdatePerfStatsFrameCount == 0:
            self.DrawStatGraph(1200, 0, 300, 150, self.gpuUtilStat, 100, 'GPU USAGE', (0,1.0,0))
            self.DrawStatGraph(1500, 0, 300, 150, self.cpuUsage, 100, 'CPU USAGE',(0,0,1.0))
            self.UpdatePerfStatsFrameCount = 120

        self.UpdatePerfStatsFrameCount -= 1


    def set_gameframe(self, img, baseFrame):

        if not self.have_nn_info:
            self.get_neuralnetwork_info()

        #print('Hello')

        #print(self.framelist[0].shape)

        #if len(self.framelist) == 0:
        #    self.final = np.zeros(shape=self.final_dim, dtype=np.uint8)
        #    return self.final


        #print('Have frame list item!!!!!!!!!!!')

        #plt.imshow(img, cmap='gray', interpolation='nearest')
        #plt.show()


        h, w, c = img.shape
        #print(img.shape)
        dim = (w * 4,h * 4)
        upscaled = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

        if not self.final_inited:
            self.final = np.zeros(shape=self.final_dim, dtype=np.uint8)
            print('final inited!!!!!!!!!!!!!!!!!!!!!')
            self.final_inited = True

        start_x = self.final_dim[1] - (w*4) -1
        start_y = self.final_dim[0] - (h*4) -1
        self.final[start_y:dim[1]+start_y,start_x:dim[0]+start_x] = upscaled

        #We only update info every 4 frames
        if not baseFrame:
            #plt.imshow(self.final, cmap='gray', interpolation='nearest')
            #plt.show()
            return self.final


        self.DrawLogo(0,0,self.logo.shape[0], self.logo.shape[1])
        self.DrawHardwareInfo(self.final, 200, 45)
        cv2.rectangle(self.final, (0, 0), (425, 200), (0,0,255), 4)

        self.DrawTrainingInfo(self.final, 0, 220)
        cv2.rectangle(self.final, (0, 215), (425, 425), (0,255,0), 4)
        
        self.DrawNeuralNetwork(self.final, img, 0, 475)


        if self.updateRewardGraph:
            self.DrawRewardGraph(460,50,400,300)
            self.updateRewardGraph = False


        # Draw Performance Stats
        self.DrawPerformanceStats()

        #self.framelist.clear()
        
        #plt.imshow(self.final, cmap='gray', interpolation='nearest')
        #plt.show()
        return self.final


broadcast = TrainingBroadcast()

