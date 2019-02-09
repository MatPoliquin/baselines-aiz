import cv2
import matplotlib.pyplot as plt




def set_for_broadcast(img):
    h, w, c = img.shape
    #print(img_nhwc.shape)

    dim = (w * 4,h * 4)
    upscaled = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')

    hello = plt.imshow(upscaled)
    print(hello)

    return hello