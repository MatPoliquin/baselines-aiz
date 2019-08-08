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
from _thread import start_new_thread
from time import sleep
#import Queue
import threading

def SampleThreadFunc(input):
    print('Thread Started!!!!!!!!!!!!!!!!!!!!!!!!')
    while(True):
        input.SamplePerformanceStats()
        sleep(0.15)


class AIZGPU():
    def __init__(self):
        self.handle = 0
        self.name = 'GPU NAME'
        self.memory = 0
        self.pcie_gen = 0
        self.pcie_width = 0
        self.utilStat = [0] * 200
        self.pcieUtilStat = [0] * 200
        self.vramUsage = 0
        #self.utilStat2 = 10

        #print(self.utilStat)


       

class AIZCPU():
    def __init__(self):
        self.name = ''
        self.num_cores = 0
        self.num_threads = 0
        self.memory = 0

        self.usage = [0] * 200



class AIZManager():
    def __init__(self):
        self.gpus = []
        self.cpu = None

        nvmlInit()
        
        #Scan GPUs
        num_gpus = nvmlDeviceGetCount()
        print('NUM GPUS:%d' % num_gpus)
        for i in range(0,num_gpus):
            self.gpus.append(AIZGPU())
            new_gpu = self.gpus[i]
            new_gpu.handle = nvmlDeviceGetHandleByIndex(i)
            new_gpu.name = nvmlDeviceGetName(new_gpu.handle)
            new_gpu.memory = nvmlDeviceGetMemoryInfo(new_gpu.handle).total
            new_gpu.pcie_gen = nvmlDeviceGetMaxPcieLinkGeneration(new_gpu.handle)
            new_gpu.pcie_width = nvmlDeviceGetMaxPcieLinkWidth(new_gpu.handle)


        #Scan CPUs
        self.cpu = AIZCPU()
        #cpu_items = get_cpu_info().items()
        #self.cpu.name = cpu_items['brand']
        self.cpu.num_cores = psutil.cpu_count(logical=False)
        self.cpu.num_threads = psutil.cpu_count()
        self.cpu.memory = int(psutil.virtual_memory().total) / 1024 / 1024


        for key, value in get_cpu_info().items():
            if key == 'brand':
                self.cpu.name = value
                break


        #self.PerformanceStats()
        self.thread = threading.Thread(target=SampleThreadFunc, args=[(self)])
        self.thread.start()

    def __del__(self):
        self.thread.terminate()
        print('AI-Z Shutdown!')


    def PrintInfo(self):
        print('GPU info')
        print("%s" % self.gpus[0].name)
        print("%d MB" % (self.gpus[0].memory / 1024 / 1024))
        print("PCIE %d.0 %dx" % (self.gpus[0].pcie_gen,self.gpus[0].pcie_width))

        print('CPU Info')
        print(self.cpu.name)
        print("%d Cores / %d Thread" % (self.cpu.num_cores, self.cpu.num_threads))
        print("%d MB RAM" % self.cpu.memory)

    
    def DrawStatGraph(self, dest, posX, posY, width, height, y_data, y_limit, title, color):
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
        dest[posY:posY+height,posX:posX+width] = image[:,:,0:3]

        plt.close()


    def SamplePerformanceStats(self):
        #print(self.gpus[0].name)

        nv_util = nvmlDeviceGetUtilizationRates(self.gpus[0].handle)
        self.gpus[0].utilStat.append(nv_util.gpu)
        pcie_tx = nvmlDeviceGetPcieThroughput(self.gpus[0].handle,NVML_PCIE_UTIL_TX_BYTES)
        self.gpus[0].pcieUtilStat.append(pcie_tx / 1024.0)
        self.gpus[0].vramUsage = nvmlDeviceGetMemoryInfo(self.gpus[0].handle).used
     

        #psutil.cpu_percent(interval=1)
        cpuStat = psutil.cpu_percent()
        self.cpu.usage.append(cpuStat)

        self.gpus[0].utilStat = self.gpus[0].utilStat[1:len(self.gpus[0].utilStat)]
        self.gpus[0].pcieUtilStat = self.gpus[0].pcieUtilStat[1:len(self.gpus[0].pcieUtilStat)]
        self.cpu.usage = self.cpu.usage[1:len(self.cpu.usage)]

        

        
    def set_tf(self, tf):
        #self.env = envNo module named 'tensorflow_datasets'

        #for i in range(0,36):
        #    print(self.env.unwrapped.get_action_meaning(i))
        self.tf = None
        return
    

aiz = AIZManager()

