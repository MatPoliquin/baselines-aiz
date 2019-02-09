import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


class TrainingBroadcast():
    def __init__(self):
        self.rewardmean = 0
        self.totaltimesteps = 0

    def set_gameframe(self, img):
        h, w, c = img.shape
        #print(img_nhwc.shape)

        dim = (w * 4,h * 4)
        upscaled = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

        final_dim = (1080, 1920, 3)
        final = np.zeros(shape=final_dim, dtype=np.uint8)

        #upscaled.cop copyto(final.rowRange(0,dim[1]).colRange(0,dim[0]))
        final[0:dim[1],0:dim[0]] = upscaled

        if math.isnan(broadcast.rewardmean):
            broadcast.rewardmean = 0;
        if math.isnan(broadcast.totaltimesteps):
            broadcast.totaltimesteps = 0;

        cv2.putText(final, ("Reward:%d" % broadcast.rewardmean), (1000,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Timesteps:%d" % broadcast.totaltimesteps), (1000,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 1 ,2)
   
        #return hello
        return final


broadcast = TrainingBroadcast()

