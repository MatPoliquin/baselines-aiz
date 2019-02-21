import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import math
import tensorflow as tf

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

    def set_env(self, env):
        #self.env = env
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

    def set_action_taken(self, action):
        self.action = action

    def set_action_prob(self, logprob):
        self.action_prob = logprob

    def get_neuralnetwork_info(self):
        print(tf.trainable_variables())
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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        input = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        final[500:dim[1]+500,0:dim[0]] = input

    def show_actions(self, final):
        cv2.putText(final, ("A:%s" % self.action_meaning[self.action]), (0,600), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 ,2)

    def show_neuralnetwork(self, final, img):
        #Input layer
        cv2.putText(final, ("Input: Stack of 4"), (0,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("84x84 pixels greyscale"), (0,475), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 ,2)

        dim = (84,84)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        input = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        final[500:dim[1]+500,0:dim[0]] = input
        final[600:dim[1]+600,0:dim[0]] = input
        final[700:dim[1]+700,0:dim[0]] = input
        final[800:dim[1]+800,0:dim[0]] = input

        #Conv net

        #Hidden layer

        #Output layer
        cv2.putText(final, ("Output"), (500,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("36 Actions"), (500,465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("Prob:%f" % self.action_prob), (500,480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 ,2)
        for i in range(0,len(self.action_meaning)):
            color = (255,255,255)
            if self.action == i:
                color = (0,255,0)
            cv2.putText(final, ("%s" % self.action_meaning[i]), (600,500 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 ,2)

    def set_gameframe(self, img):

        if not self.have_nn_info:
            self.get_neuralnetwork_info()


        h, w, c = img.shape
   
        dim = (w * 4,h * 4)
        upscaled = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

        final = np.zeros(shape=self.final_dim, dtype=np.uint8)

        start_x = self.final_dim[1] - (w*4) -1

        final[0:dim[1],start_x:dim[0]+start_x] = upscaled

        self.show_stats(final)
        self.show_probabilities(final)
        #self.show_inputimage(final,img)
        #self.show_actions(final)
        self.show_neuralnetwork(final, img)

        

        return final


broadcast = TrainingBroadcast()

