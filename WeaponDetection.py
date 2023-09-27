from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
import matplotlib.pyplot as plt
import winsound

main = tkinter.Tk()
main.title("Weapon Detection in Real-Time CCTV Videos Using Deep Learning")
main.geometry("1300x900")

global model, classes, layer_names, output_layers, colors, filename, dataset

def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def uploadDataset():
    global dataset
    text.delete('1.0', END)
    dataset = filedialog.askdirectory(initialdir = ".")
    X_train = np.load('model/X.txt.npy')
    Y_train = np.load('model/Y.txt.npy')
    X_train = X_train.astype('float32')
    X_train = X_train/255
    test = X_train[3]
    indices = np.arange(X_train.shape[0])   
    np.random.shuffle(indices)
    X_train = X_train[indices]
    Y_train = Y_train[indices]
    text.insert(END,"Total images found in dataset: "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total classes found in dataset is KNIVES & GUNS")
    text.update_idletasks()
    cv2.imshow("Sample Loaded Image",cv2.resize(test,(300,300)))
    cv2.waitKey(0)


def loadModel():
    text.delete('1.0', END)
    global model, classes, layer_names, output_layers, colors
    model = cv2.dnn.readNet("model/yolov4_training_2000.weights", "model/yolov4.cfg")
    classes = ['Pistol']
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    pathlabel.config(text="Weapon Detection Model Loaded")
    text.insert(END,"Yolov4 Weapon Detection Model Loaded\n\n")
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    data = data['accuracy']
    acc = data[1]
    precision = data[2]
    recall = data[3]
    fscore = data[4]
    text.insert(END,"Yolov4 Accuracy  : "+str(acc)+"\n")
    text.insert(END,"Yolov4 Precision : "+str(precision)+"\n")
    text.insert(END,"Yolov4 Recall    : "+str(recall)+"\n")
    text.insert(END,"Yolov4 FSCORE    : "+str(fscore)+"\n")

def detectWeapon():
    global model, classes, layer_names, output_layers, colors, filename
    img = cv2.imread(filename)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    outs = model.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0:
                print(str(class_id)+" "+str(confidence))
            if confidence > 0.8:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(str(indexes)+" "+str(len(boxes)))
    if indexes == 0:
        text.insert(END,"weapon detected in image\n")
    flag = 0
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            beep()
            flag = 1           
    if flag == 0:
        cv2.putText(img, "Non Pistol", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(0)

def uploadImage():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "testImages")
    pathlabel.config(text=filename+" loaded")
    text.insert(END,filename+" loaded\n")

def detectVideoWeapon():
    global model, classes, layer_names, output_layers, colors, filename
    filename = askopenfilename(initialdir = "Videos")
    cap = cv2.VideoCapture(filename)
    while True:
        _, img = cap.read()
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        model.setInput(blob)
        outs = model.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        if indexes == 0: print("weapon detected in frame")
        font = cv2.FONT_HERSHEY_PLAIN
        flag = 0
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                flag = 1
                beep()
        if flag == 0:
            cv2.putText(img, "Non Pistol", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)               
        cv2.imshow("Image", img)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break   
    cap.release()
    cv2.destroyAllWindows()


def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['accuracy']
    print(accuracy)
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('Weapon Detection Training Accuracy & Loss Graph')
    plt.show()
    

font = ('times', 16, 'bold')
title = Label(main, text='Weapon Detection in Real-Time CCTV Videos Using Deep Learning',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')

datasetButton = Button(main, text="Upload Weapon Dataset", command=uploadDataset)
datasetButton.place(x=50,y=100)
datasetButton.config(font=font1)

loadButton = Button(main, text="Load Yolov4 Weapon Detection Model", command=loadModel)
loadButton.place(x=50,y=150)
loadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=440,y=150)

uploadButton = Button(main, text="Upload Image", command=uploadImage)
uploadButton.place(x=50,y=200)
uploadButton.config(font=font1)

detectButton = Button(main, text="Detect Weapon from Image", command=detectWeapon)
detectButton.place(x=50,y=250)
detectButton.config(font=font1)

videoButton = Button(main, text="Detect Weapon from Video", command=detectVideoWeapon)
videoButton.place(x=50,y=300)
videoButton.config(font=font1)

graphButton = Button(main, text="Weapon Detection Training Accuracy-Loss Graph", command=graph)
graphButton.place(x=350,y=300)
graphButton.config(font=font1)

text=Text(main,height=10,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=400)
text.config(font=font1)

main.config(bg='chocolate1')
main.mainloop()
