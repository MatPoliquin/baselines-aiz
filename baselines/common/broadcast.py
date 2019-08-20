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
from baselines.common.aiz import aiz
from baselines.common.aiz_nn import AIZ_NeuralNet
import time
import datetime
import string

class TrainingBroadcast():
    def __init__(self):
        self.rewardmean = 0
        self.totaltimesteps = 0
        self.fps = 0
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
        self.UpdateNeuralNetFrameCount=0
        self.final_inited = False
        self.final = None
        self.total_params = 0
        self.rewardmeanList = []
        self.updateRewardGraph = True
        self.UpdatePerfStatsFrameCount=0
        self.audio_rate = 0
        self.neuralNet = AIZ_NeuralNet()
        self.reward = 0
        self.frameRewardList = [0] * 200
        self.frameListUpdateCount = 0
        self.explained_variance = 0
        self.policy_entropy = 0
        self.policy_loss = 0
        self.learning_rate = 0
        self.gamma = 0
        

        dir_path = os.path.dirname(os.path.realpath(__file__))
        print('=============DIR PATH================')
        print(dir_path)
    

        path = os.path.join(dir_path, '../../data/broadcast_logo.jpg')
        print(path)
        img = Image.open(path)    
        dim = (170,170)
        self.logo = cv2.resize(np.array(img), dim, interpolation=cv2.INTER_AREA)

        aiz.PrintInfo()

    #def __del__(self):
    #    del aiz
    
    def Shutdown(self):
        aiz.Shutdown()

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
        #print('==============ACTION MEANINGS SET================')
        #print('Size:%d' % len(self.action_meaning))
        #for i in range(0,len(self.action_meaning)):
        #    print(self.action_meaning[i])
        #print('=================================================')


        self.neuralNet.set_action_meaning(actionlist)

    def set_action_taken(self, action):
        self.action = action
        self.neuralNet.set_action_taken(action)

    def set_action_prob(self, logprob):
        self.action_prob = logprob
        self.neuralNet.set_action_prob(logprob)

        #odds = np.exp(logprob)
        #prob = odds / (1 + odds)

        #print(prob, logprob)

    def set_reward(self, rew):
        self.reward = rew
        self.frameRewardList.append(rew)
        self.frameRewardList = self.frameRewardList[1:len(self.frameRewardList)]
        #if rew != 0:
        #    print(rew)

        self.frameListUpdated = True


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

    def clear_screen(self, posX, posY, width, height, color = (0,0,0)):
        self.final[posY:posY+height,posX:posX+width] = color

    def LogRewardMean(self, rew):
        self.rewardmeanList.append(rew)
        self.updateRewardGraph = True

    def DrawMainInfo(self, final, PosX, PosY):
        

        #cv2.rectangle(self.final, (PosX, PosY), (275, 150), (0,255,0), 4)

        #PosX += 10
        env_name = self.env_id
        
        PosY += 15
        cv2.putText(final, ("ENV:      %s" % env_name.upper()), (PosX,PosY), self.font, 1.0, (255,255,255), 1 ,2)
        PosY += 15
        cv2.putText(final, ("DATE:     %s" % time.strftime("%d/%m/%Y")), (PosX,PosY), self.font, 1.0, (255,255,255), 1 ,2)

        #PosY += 30
        #cv2.putText(final, ("HOW TO REPLICATE"), (PosX,PosY), self.font, 1.0, (255,255,255), 1 ,2)
        #PosY += 15
        #cv2.putText(final, ("videogames.ai/How-To-Replicate-Experiments"), (PosX,PosY), self.font, 0.5, (255,255,255), 1 ,2)

    def DrawMainStats(self, final, PosX, PosY):
        #Posx += 400
        if math.isnan(broadcast.rewardmean):
            broadcast.rewardmean = 0;
        if math.isnan(broadcast.totaltimesteps):
            broadcast.totaltimesteps = 0;

        self.clear_screen(PosX, PosY, 200, 250, (0,0,0))

        PosY += 15
        cv2.putText(final, ("REWARD MEAN:"), (PosX,PosY), self.font, 1.0, (255,255,255), 1 ,2)
        PosY += 30
        cv2.putText(final, ("%d" % broadcast.rewardmean), (PosX,PosY), self.font, 2.0, (255,255,255), 2 ,2)
        PosY += 30         
        cv2.putText(final, ("TIMESTEPS:"), (PosX,PosY), self.font, 1.0, (255,255,255), 1 ,2)
        PosY += 30
        cv2.putText(final, ("%d" % broadcast.totaltimesteps), (PosX,PosY), self.font, 2.0, (255,255,255), 2 ,2)
        play_time = datetime.timedelta(seconds=int(broadcast.totaltimesteps * (1/15)))
        PosY += 30
        cv2.putText(final, ("= %s TOTAL PLAY TIME" % play_time), (PosX,PosY), self.font, 1.0, (255,255,255), 1 ,2)
      
        

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


        #nv_util = nvmlDeviceGetUtilizationRates(self.gpu_handle)z
        #print("UTILS: %d" % nv_util.gpu)

        saved_y = posY

        self.clear_screen(posX, posY + 60, 200, 50, (0,0,0))

        cv2.putText(final, "HARDWARE", (posX, posY), self.font, 1.5, (0,255,255), 2 ,2)
        posY += 30
        cv2.putText(final, ("%s" % aiz.cpu.name), (posX, posY), self.font, 1.0, (0,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("%d cores / %d threads" % (aiz.cpu.num_cores,aiz.cpu.num_threads)), (posX, posY), self.font, 1.0, (0,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("%s" % aiz.gpus[0].name), (posX, posY), self.font, 1.0, (0,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("PCIE %d.0 %dx :%d %%" % (aiz.gpus[0].pcie_gen,aiz.gpus[0].pcie_width, aiz.gpus[0].pcieUtilStat[0])), (posX, posY), self.font, 1.0, (0,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("%d/%d MB VRAM" % (aiz.gpus[0].vramUsage/1024/1024, aiz.gpus[0].memory/1024/1024)), (posX, posY), self.font, 1.0, (0, 255,255), 1 ,2)
        posY += 15
        #cv2.putText(final, ("TIMESTEPS/S: %d" % broadcast.fps), (posX ,posY), self.font, 1.0, (255,255,255), 1 ,2)
        
        #posY += 100
        posY = saved_y
        posX += 320
        cv2.putText(final, "SOFTWARE", (posX, posY), self.font, 1.5, (0,255,255), 2 ,2)
        posY += 30
        cv2.putText(final, ("OPENAI BASELINES 0.1.6"), (posX, posY), self.font, 1.0, (255,255,0), 1 ,2)
        posY += 15
        cv2.putText(final, ("TENSORFLOW %s" % tf.__version__), (posX, posY), self.font, 1.0, (255,255,0), 1 ,2)
        posY += 15
        cv2.putText(final, ("CUDA 10"), (posX, posY), self.font, 1.0, (255,255,0), 1 ,2)
   
    def DrawAlgoDetails(self, final, img, posX, posY):
        
        
        saved_posY = posY
        saved_posX = posX


        cv2.putText(final, "REWARD FUNCTION:", (posX,posY+15), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, "RETURNS A SCORE [-1, 1]", (posX,posY+30), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, "AT EVERY TIMESTEPS", (posX,posY+45), self.font, 1.0, (255,255,255), 1 ,2)
        self.DrawFrameRewardHistogram(posX,posY+60,250,150)


         #cv2.putText(final, ("REWARD:     %s" % self.reward), (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        posY = saved_posY + 15
        posX += 350

        self.clear_screen(posX, posY - 15, 300, 200, (0,0,0))
        alg_name = self.alg
        cv2.putText(final, ("ALGO:                 %s" % alg_name.upper()), (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("NEURAL NET TYPE:    %s" % (self.args['network']).upper()), (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("TRAINABLE PARAMS:   %s float32" % format(self.total_params, ',d')), (posX,posY), self.font, 1.0, (0,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("OPTIMIZER:            ADAM"), (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        posY += 30
        cv2.putText(final, "HYPER PARAMS:", (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("  LEARNING RATE: %f" % self.learning_rate), (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("  GAMMA: %f" % self.gamma), (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        
        posY += 30
        cv2.putText(final, "STATS:", (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("   EXPLAINED VARIANCE: %f" % self.explained_variance), (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("   POLICY ENTROPY: %f" % self.policy_entropy), (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)
        posY += 15
        cv2.putText(final, ("   POLICY LOSS: %f" % self.policy_loss), (posX,posY), self.font, 1.0, (255,255,255), 1 ,2)


        posY = saved_posY
        posX = saved_posX
       
        self.neuralNet.Draw(self.final, img, posX, posY + 300)
    

    def DrawRewardGraph(self, posX, posY, width, height):
        fig = plt.figure(0)

        plt.plot(self.rewardmeanList, color=(0,1,0))

        #plt.title(("Reward Mean: %d" % broadcast.rewardmean), color=(0,1,0))

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

    def DrawLogo(self, posX, posY, width, height):

        self.final[posY:posY+height,posX:posX+width] = self.logo

    def DrawPerformanceStats(self, posX, posY):

       

        if self.UpdatePerfStatsFrameCount == 0:
            aiz.DrawStatGraph(self.final, posX, posY, 300, 150, aiz.cpu.usage, None, 100, 'CPU USAGE',(0,1,1.0))
            #aiz.DrawStatGraph(self.final, 1150, 0, 300, 150, aiz.gpus[0].pcieUtilStat, None, 100, 'PCIE USAGE', (0,1.0,0))
            aiz.DrawStatGraph(self.final, posX + 350, posY, 300, 150, aiz.gpus[0].utilStat, aiz.gpus[0].pcieUtilStat, 100, 'GPU(green)/PCIE(blue) USAGE', (0,1.0,0))
            #aiz.DrawStatGraph(self.final, 1500, 0, 300, 150, aiz.gpus[0].utilStat, None, 100, 'GPU USAGE', (0,1.0,0))
            
            
            self.UpdatePerfStatsFrameCount = 30

        self.UpdatePerfStatsFrameCount -= 1

    def DrawFrameRewardHistogram(self, posX, posY, width, height):
        fig = plt.figure(0)

        plt.plot(self.frameRewardList, color=(0,1,0))
        #plt.hist(self.frameRewardList, bins=1, rwidth=0.8)
        #plt.bar(self.frameRewardList, 50)

        #plt.title(("Reward Mean: %d" % broadcast.rewardmean), color=(0,1,0))

        fig.set_facecolor('black')

        numYData = len(self.frameRewardList)
        plt.xlim([0,numYData])
        plt.ylim([-1,1])
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

        self.frameListUpdated = False


    def CalculateGameScreenDimensions(self, img):
        h, w, c = img.shape
        #print(img.shape)
        #Adjust everything to 240 x 180 resolition
        #xScale = (240.0 / w) * 4
        #yScale = 0.75 * 4
        #dim = (int(w * xScale), int(240 * yScale))
        dim = (240 * 5, 180 * 5)
        return dim


    def set_gameframe(self, img, baseFrame):

        if not self.have_nn_info:
            self.get_neuralnetwork_info()

        
        dim = self.CalculateGameScreenDimensions(img)

        start_x = self.final_dim[1] - int(dim[0]) -1
        start_y = self.final_dim[0] - int(dim[1]) -1

        upscaled = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

        if not self.final_inited:
            self.final = np.zeros(shape=self.final_dim, dtype=np.uint8)
            print('final inited!!!!!!!!!!!!!!!!!!!!!')
            self.final_inited = True

        

       

        #We only update info every 4 frames
        if not baseFrame:
            #plt.imshow(self.final, cmap='gray', interpolation='nearest')
            #plt.show()
            return self.final


        # Draw Performance Stats
        machine_y = self.final_dim[0] - self.logo.shape[1]
        self.DrawPerformanceStats(0, machine_y - 175)

        self.DrawLogo(0, machine_y, self.logo.shape[0], self.logo.shape[1])
        self.DrawHardwareInfo(self.final, self.logo.shape[0], machine_y + 30)
        #cv2.rectangle(self.final, (0, 0), (425, 200), (0,0,255), 4)

        
        self.DrawMainStats(self.final, start_x + 650, 0)
        self.DrawMainInfo(self.final, start_x + 850, 0)
        
        
    

        
        if self.frameListUpdateCount == 0:
            self.DrawAlgoDetails(self.final, img, 0, 0)
            self.frameListUpdateCount = 1

        self.frameListUpdateCount -= 1


        if self.updateRewardGraph:
            self.DrawRewardGraph(start_x, 0, 500, 150)
            self.updateRewardGraph = False


        
        #cv2.putText(self.final, ("Showing Env 0 out of 8 parallel environments"), (start_x, start_y), self.font, 1.0, (0,255,0), 1 ,2)


        #cv2.putText(final, ("HOW TO REPLICATE"), (PosX,PosY), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(self.final, ("videogames.ai/How-To-Replicate").upper(), (self.logo.shape[0],self.final_dim[0]-20), self.font, 1.0, (255,255,255), 1 ,2)


        # =============== DRAW GAME FRAME ================
        
        self.final[start_y:dim[1]+start_y,start_x:dim[0]+start_x] = upscaled
      
        return self.final


broadcast = TrainingBroadcast()

