import cv2
import matplotlib.pyplot as plt
import math



class TrainingBroadcast():
    def __init__(self):
        self.rewardmean = 0
        self.totaltimesteps = 0


broadcast = TrainingBroadcast()

def set_for_broadcast(img):
    h, w, c = img.shape
    #print(img_nhwc.shape)

    dim = (w * 4,h * 4)
    upscaled = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

    if math.isnan(broadcast.rewardmean):
        broadcast.rewardmean = 0;
    if math.isnan(broadcast.totaltimesteps):
        broadcast.totaltimesteps = 0;


    text = "Text:%d, %d" % (broadcast.rewardmean, broadcast.totaltimesteps);

    cv2.putText(upscaled, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
   
    #return hello
    return upscaled