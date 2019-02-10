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


    def playintro(self):
        introarray = []
        intro = np.zeros(shape=self.final_dim, dtype=np.uint8)
        cv2.putText(intro, "INTRO", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1 ,2)

        for i in range(0,60):
            introarray.append(intro)

        self.playedintro = True

        return introarray

    def set_gameframe(self, img):
        h, w, c = img.shape
        #print(img_nhwc.shape)

        dim = (w * 4,h * 4)
        upscaled = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

        
        final = np.zeros(shape=self.final_dim, dtype=np.uint8)

        #upscaled.cop copyto(final.rowRange(0,dim[1]).colRange(0,dim[0]))
        final[0:dim[1],0:dim[0]] = upscaled

        if math.isnan(broadcast.rewardmean):
            broadcast.rewardmean = 0;
        if math.isnan(broadcast.totaltimesteps):
            broadcast.totaltimesteps = 0;

        cv2.putText(final, ("Reward:%d" % broadcast.rewardmean), (1000,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Timesteps:%d" % broadcast.totaltimesteps), (1000,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1 ,2)

        finalarray = []
        if not self.playedintro:
            intro = self.playintro()
            finalarray = intro
   
        finalarray.append(final)
        #finalarray.append(final)

        return finalarray


broadcast = TrainingBroadcast()

