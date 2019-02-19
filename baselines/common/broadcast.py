import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


class TrainingBroadcast():
    def __init__(self):
        self.rewardmean = 0
        self.totaltimesteps = 0
        self.final_dim = (1080, 1920, 3)
        self.playedintro = False
        self.framelist = []


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
        cv2.putText(final, "Probabilities", (0,200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1 ,2)

    def show_stats(self, final):
        if math.isnan(broadcast.rewardmean):
            broadcast.rewardmean = 0;
        if math.isnan(broadcast.totaltimesteps):
            broadcast.totaltimesteps = 0;

        cv2.putText(final, ("Reward:%d" % broadcast.rewardmean), (0,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Timesteps:%d" % broadcast.totaltimesteps), (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1 ,2)

    def show_inputimage(self, final, img):
        dim = (84,84)
        input = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        final[500:dim[1]+500,0:dim[0]] = input

    def set_gameframe(self, img):
        h, w, c = img.shape
   
        dim = (w * 4,h * 4)
        upscaled = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

        final = np.zeros(shape=self.final_dim, dtype=np.uint8)

        start_x = self.final_dim[1] - (w*4) -1

        final[0:dim[1],start_x:dim[0]+start_x] = upscaled

        self.show_stats(final)
        self.show_probabilities(final)
        self.show_inputimage(final,img)

        return final


broadcast = TrainingBroadcast()

