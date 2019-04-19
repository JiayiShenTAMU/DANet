#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tkinter
import os

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
#from keras.models import load_model
import numpy
#import keras

HEIGHT = 600
WIDTH = 500
top = Tk()
top.title = 'cifar10'
top.geometry('2400x2500')
canvas1 = Canvas(top, width=WIDTH,height=HEIGHT, bd=0,bg='white')
canvas1.grid(row=1, column=0)

canvas2 = Canvas(top, width=WIDTH,height=HEIGHT, bd=0,bg='white')
canvas2.grid(row=1, column=1)

def showImg():
    File = askopenfilename(title='Open Image')
    e.set(File)

    load = Image.open(e.get())
    w, h = load.size
    load = load.resize((WIDTH,HEIGHT))
    imgfile = ImageTk.PhotoImage(load )

    canvas1.image = imgfile  # <--- keep reference of your image
    canvas1.create_image(2,2,anchor='nw',image=imgfile)
    global name
    name = e.get()


e = StringVar()

submit_button = Button(top, text ='Open', command = showImg)
submit_button.grid(row=0, column=0)

status = os.system('sh ../danet/629.sh')
print (status)

label_name = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def Predict():
    #File2 = askopenfilename(title='Open Image')
    #e2.set(File2)

    Str = name.split('/')
    Str2 = Str[-1].split('.')

    load2 = Image.open("../datasets/ISIC/danet_vis/" + Str2[0] + ".png")
    w2, h2 = load2.size
    load2 = load2.resize((WIDTH,HEIGHT))
    imgfile2 = ImageTk.PhotoImage(load2 )

    canvas2.image = imgfile2  # <--- keep reference of your image
    canvas2.create_image(2,2,anchor='nw',image=imgfile2)


    '''
    img=Image.open(e.get())
    img=img.resize((32, 32))
    imgArray = numpy.array(img)
    imgArray = imgArray.reshape(1, 32 * 32 * 3)
    imgArray = imgArray.astype('float32')
    imgArray /= 255.0
    #model=load_model('mlp_cifar10.h5')

    #clsimg=model.predict_classes(imgArray)
    clsimg = 1
    textvar = "The object is : %s" %(label_name[int(clsimg)])
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar+'\n')
    t1.update()
    '''
e2 = StringVar()
submit_button = Button(top, text ='Segmentation', command = Predict)
submit_button.grid(row=0, column=1)

l1=Label(top,text='Please <Open> a RGB image, then press <Segmentation> ')
l1.grid(row=2)



#t1=Text(top,bd=0, width=20,height=10,font='Fixdsys -14')
#t1.grid(row=1, column=1)
top.mainloop()
