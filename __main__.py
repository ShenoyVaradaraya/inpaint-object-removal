from skimage.io import imread, imsave
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
import os
from PIL import ImageTk, Image

from inpainter import inpainting


drawing = False
mode = True
target_contour_list = []
fname = ""
root = Tk()

def mask_builder(target_contour_list, image):
    nb_lines, nb_columns = image.shape[0], image.shape[1]
    mask = np.zeros((nb_lines, nb_columns))

    array_contour = np.array(target_contour_list)
    cv2.fillPoly(mask, pts=[array_contour], color=1.0)

    return mask


def draw_contours(event,x,y,flags,param):
    global fname
    image = cv2.imread(fname)
    global ix,iy,drawing, mode

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                target_contour_list.append([x, y])
                for c in target_contour_list:
                    cv2.circle(image,(c[0],c[1]),1,(0,0,255),-1)
                    cv2.imshow("Mask Builder", image)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.circle(image,(x,y),1,(0,0,255),-1)

def main():

    global fname
    fname = openfilename()
    head, tail = os.path.split(fname)
    filename = os.path.splitext(tail)
    img = Image.open(fname)
     
    print(fname)
    img = ImageTk.PhotoImage(img)
  
    panel = Label(root, image = img)
     
    image = cv2.imread(fname)
    panel.image = image
    panel.grid(row = 2)
    cv2.namedWindow('building_mask')
    cv2.imshow('building_mask',image)
    cv2.setMouseCallback('building_mask',draw_contours)
    cv2.waitKey(0)
    mask = mask_builder(target_contour_list,image)
    cv2.imshow("mask",mask)
    cv2.waitKey(0)

    output_image = inpainting(
        image,
        mask,
        patch_size=10
    ).driver()
    cv2.imwrite("output_"+filename[0]+".jpg", output_image)

def openfilename():
 
    filename = filedialog.askopenfilename(title ='"pen')
    return filename

 
root.title("Image Loader")
 
root.geometry("550x300+300+150")

root.resizable(width = True, height = True)
 
btn = Button(root, text ='open image', command = main).grid(
                                        row = 1, columnspan = 4)
root.mainloop()



if __name__ == '__main__':
    main()
