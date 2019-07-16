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
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.stats_fontscale = 1.0
        self.stats_pos = (50,50)
        self.nn_pos = (0,450)


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
        print('=================================================')

    def set_action_taken(self, action):
        self.action = action

    def set_action_prob(self, logprob):
        self.action_prob = logprob

    def get_neuralnetwork_info(self):
        #print(tf.trainable_variables())
        print('==============TRAINABLE PARAMETERS================')
        for s in tf.trainable_variables():
            print(s)

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

    def show_stats(self, final):
        if math.isnan(broadcast.rewardmean):
            broadcast.rewardmean = 0;
        if math.isnan(broadcast.totaltimesteps):
            broadcast.totaltimesteps = 0;

        cv2.putText(final, ("Algo: PPO2"), (0,50), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Neural Net: CNN"), (0,100), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Reward mean:%d" % broadcast.rewardmean), (0,150), self.font, 1.0, (255,255,255), 1 ,2)
        cv2.putText(final, ("Timesteps:%d" % broadcast.totaltimesteps), (0,200), self.font, 1.0, (255,255,255), 1 ,2)

    def show_inputimage(self, final, img):
        dim = (84,84)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        input = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        final[500:dim[1]+500,0:dim[0]] = input

    def show_actions(self, final):
        cv2.putText(final, ("A:%s" % self.action_meaning[self.action]), (0,600), self.font, 0.5, (255,255,255), 1 ,2)

    


    def show_neuralnetwork(self, final, img):
        #Input layer
        cv2.putText(final, ("Input"), (0,450), self.font, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("84x84 pixels greyscale"), (0,475), self.font, 0.5, (255,255,255), 1 ,2)

        dim = (84,84)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        input = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
        final[500:dim[1]+500,0:dim[0]] = input
        final[600:dim[1]+600,0:dim[0]] = input
        final[700:dim[1]+700,0:dim[0]] = input
        final[800:dim[1]+800,0:dim[0]] = input

        #Conv net layer 1
        cv2.putText(final, ("Convnet 1"), (100,450), self.font, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("32 filters"), (100,465), self.font, 0.5, (255,255,255), 1 ,2)
        #cv2.putText(final, ("8x8"), (100,480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 ,2)
        
        cv2.rectangle(final, (100,480), (180, 560), (255,255,255))

        #Conv net layer 2
        cv2.putText(final, ("Convnet 2"), (200,450), self.font, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("64 filters"), (200,465), self.font, 0.5, (255,255,255), 1 ,2)
        #cv2.putText(final, ("4x4"), (200,480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 ,2)

        cv2.rectangle(final, (200, 450), (240, 490), (255,255,255))

        #Conv net layer 3
        cv2.putText(final, ("Convnet 3"), (300,450), self.font, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("64 filters"), (300,465), self.font, 0.5, (255,255,255), 1 ,2)
        #cv2.putText(final, ("3x3"), (300,480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 ,2)

        cv2.rectangle(final, (300, 450), (340, 480), (255,255,255))

        #Hidden layer
        cv2.putText(final, ("Hidden Layer"), (400,450), self.font, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("512 units"), (400,465), self.font, 0.5, (255,255,255), 1 ,2)

        cv2.circle(final, (400,480), 50, (255,255,255,255))

        #Output layer
        cv2.putText(final, ("Output"), (600,450), self.font, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("36 Actions"), (600,465), self.font, 0.5, (255,255,255), 1 ,2)
        cv2.putText(final, ("Prob:%f" % self.action_prob), (600,480), self.font, 0.5, (255,255,255), 1 ,2)
        for i in range(0,len(self.action_meaning)):
            color = (255,255,255)
            if self.action == i:
                color = (0,255,0)
            cv2.putText(final, ("%s" % self.action_meaning[i]), (600,500 + i*15), self.font, 0.5, color, 1 ,2)

    def show_nn_weights(self, final):
        #Show convnet 1
        with tf.variable_scope('ppo2_model/pi/c1', reuse=True) as conv_scope:
            hello = tf.get_variable('w', shape=[8,8,4,32])
            #print(hello)
            weights0 = tf.transpose(hello, [3,0,1,2])
            weights = weights0.eval()
            #print(weights.shape)

            num_channels = hello.shape.dims[3]
            filter_w = hello.shape.dims[0]
            filter_h = hello.shape.dims[1]
            #print(num_channels)
            x = 0
            #for channel in range(0,num_channels):
            for channel in range(20,30):
                #print(channel)
                img = weights[channel,:,:,0]
                #print(img.eval())
                #img = img * 255.0

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                #img2 = []
                img2 = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

                dim = (filter_w * 10,filter_h * 10)
                upscaled = cv2.resize(img2, dim, interpolation=cv2.INTER_NEAREST)

                x=x+80
                #print(channel)
                #print(x)
                y=0

                final[x:x+dim[0], y:y+dim[1]] = np.uint8(upscaled)

    def show_nn_weights2(self, final):
        #Show convnet 1
        with tf.variable_scope('ppo2_model/pi/fc1', reuse=True) as conv_scope:
            hello = tf.get_variable('w', shape=[3136,512])
            #print(hello)
            weights0 = tf.transpose(hello, [1,0])
            weights = weights0.eval()
            #print(weights.shape)

            #num_channels = hello.shape.dims[3]
            #filter_w = hello.shape.dims[0]
            #filter_h = hello.shape.dims[1]
            #print(num_channels)
            x = 0
            #for channel in range(0,num_channels):
            for channel in range(0,20):
                #print(channel)
                img0 = weights[channel,:]
                img = np.reshape(img0, (56,56))
                #print(img.eval())
                #img = img * 255.0

                

                img = (img + 1) / 2

                #print(img)

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                #img2 = []
                img2 = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

                dim = (56*4,56*4)
                upscaled = cv2.resize(img2, dim, interpolation=cv2.INTER_NEAREST)

                x = (channel % 3) * (56 * 4)
                y = int(channel / 3) * (56 * 4)


                #x=x+(56+3)
                #print(channel)
                #print(x)
                #y=(channel / 10) * (56 + 3)

                final[x:x+dim[0], y:y+dim[1]] = np.uint8(upscaled)
        #Show convnet 2
        #with tf.variable_scope('ppo2_model/pi/c2', reuse=True) as conv_scope:
        #    hello = tf.get_variable('w', shape=[4,4,32,64])
        #    print(hello)
        #    weights = hello.eval()
        #    print(weights.shape)

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
        self.show_inputimage(final,img)
        self.show_actions(final)
        self.show_neuralnetwork(final, img)
        self.show_nn_weights2(final)

        

        return final


broadcast = TrainingBroadcast()

