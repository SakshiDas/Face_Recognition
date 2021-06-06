import cv2
import os
import numpy as np
video= cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

import tkinter as tk
from tkinter import messagebox

def generate_data():
    name=username.get()

    skip=0
    face_training=[]
    path='C:/Users/Admin/Desktop/tkinter project/data/'

    while True:
        boolean ,frame = video.read()

        if boolean == False:
            continue

        
        face = classifier.detectMultiScale(frame, 1.3, 5)
        if len(face)==0:
            continue

        #arrange in decsending order of faces incase of multiple faces
        face = sorted(face, key = lambda l:l[2]*l[3], reverse = True)
        for (x,y,b,h) in face:
            cv2.rectangle(frame, (x,y), (x+b,y+h), (0,0,225),3)

            #cropping face part
            offset=10
            cropped_face = frame[y-offset:y+h+offset,x-offset:x+b+offset]

            #resize image spthat all training data is of same size
            cropped_face = cv2.resize(cropped_face,(100,100))

            
            #storing every 10th face
            skip+=1
            if (skip%10==0):
                face_training.append(cropped_face)
                print(len(face_training))

        cv2.imshow('video stream', frame)
        cv2.imshow("face", cropped_face)


        #if user press e then exit
        press = cv2.waitKey(1) & 0xff
        if press == ord('e'):
            break

    # Converting our face_training list array into a numpy array
    face_training = np.asarray(face_training)
    face_training = face_training.reshape((face_training.shape[0],-1))
    print(face_training.shape)

    # Saving training data into file system
    np.save(path+name+'.npy',face_training)
    messagebox.showinfo("Face Recognition", "Data Successfully saved at "+path+name+'.npy')

    video.release()
    cv2.destroyAllWindows()

    #KNN
def distance(v1, v2):
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]


def testing_data():
    #video streaming
    video= cv2.VideoCapture(0)
    classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


    skip=0
    face_train=[]
    labels=[]
    path='C:/Users/Admin/Desktop/tkinter project/data/'

    class_id = 0 #labels
    names = {} #maping of names with labels



    # loading training data
    for t in os.listdir(path):
        if t.endswith('.npy'):
            #Create a mapping btw class_id and name
            names[class_id] = t[:-4]
            print("Loaded "+t)
            data_item = np.load(path+t) #giving file name plus path to load
            face_train.append(data_item)

            #Create Labels for the class
            target = class_id*np.ones((data_item.shape[0],))
            class_id += 1
            labels.append(target)

    face_dataset = np.concatenate(face_train,axis=0)
    face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

    print(face_dataset.shape)
    print(face_labels.shape)

    trainset = np.concatenate((face_dataset,face_labels),axis=1)
    print(trainset.shape)


    #testing

    while True:
        boolean ,frame = video.read()

        if boolean == False:
            continue

        
        face = classifier.detectMultiScale(frame, 1.3, 5)
        if len(face)==0:
            continue

        #arrange in decsending order of faces incase of multiple faces
        face = sorted(face, key = lambda l:l[2]*l[3], reverse = True)
        for (x,y,b,h) in face:
            cv2.rectangle(frame, (x,y), (x+b,y+h), (0,0,225),3)

            #cropping face part
            offset=10
            cropped_face = frame[y-offset:y+h+offset,x-offset:x+b+offset]

            #resize image spthat all training data is of same size
            cropped_face = cv2.resize(cropped_face,(100,100))

            pred = knn(trainset, cropped_face.flatten())

            pred_name = names[int(pred)]
            cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+b,y+h),(0,255,255),2)

        cv2.imshow("Faces",frame)

        key = cv2.waitKey(1) & 0xFF
        if key==ord('e'):
            break
    video.release()
    cv2.destroyAllWindows()
    




#Tkinter
root = tk.Tk()
root.title("Facial recognition")
root.geometry('1366x786+0+0')
root.config(bg='#804da9')
#Frames
loginFrame=tk.Frame(root, bg='#13857D', width=900, height=400, relief='ridge', bd=22)
regFrame=tk.Frame(root, bg='#13857D', width=900, height=400, relief='ridge', bd=22)
userMenuFrame = tk.Frame(root, bg='#13857D', width=900, height=400, relief='ridge', bd=22)

#Define Frame List
frameList=[loginFrame,regFrame,userMenuFrame]
#Configure all Frames
for frame in frameList:
	frame.place(x=240, y=140)
	frame.configure(bg="#ffeeff")
	
def raiseFrame(frame):
	frame.tkraise()

def regFrameRaiseFrame():
	raiseFrame(regFrame)
def logFrameRaiseFrame():
	raiseFrame(loginFrame)
#Tkinter Vars
#Stores user's name when registering
username = tk.StringVar()
#Stores user's name when they have logged in
loggedInUser = tk.StringVar()


tk.Label(loginFrame,text="Face Recognition",font=("Helvetica", 30, "bold"),bg="#ffeeff").place(x=275,y=100)
loginButton = tk.Button(loginFrame,text="Training",command=regFrameRaiseFrame,bg="#804da9",fg = 'white',font=("Helvetica", 20, "bold"), bd=5)
loginButton.place(x=250,y=200)
regButton = tk.Button(loginFrame,text="Testing",bg="#804da9",fg = 'white',font=("Helvetica", 20, "bold"), command=testing_data,bd=5)
regButton.place(x=500,y=200)

#=============================================TRAINING FACES===========================================================================
tk.Label(regFrame,text="Register Your Face Here",font=("Helvetica", 30, "bold"),bg="#ffeeff").place(x=220,y=75)
tk.Label(regFrame,text="Name: ",font=("Helvetica", 17, "bold"),bg="#ffeeff").place(x=250,y=170)
nameEntry=tk.Entry(regFrame,textvariable=username,font=("Helvetica", 17, "bold")).place(x=360, y=170, width=250)

registerButton = tk.Button(regFrame,text="Proceed",bg="#804da9",fg = 'white',font=("Helvetica", 17, "bold"),command=generate_data)
registerButton.place(x=260,y=240)
goBack = tk.Button(regFrame,text="Go Back",bg="#804da9",fg = 'white',font=("Helvetica", 17, "bold"),command=logFrameRaiseFrame)
goBack.place(x=520,y=240)



raiseFrame(loginFrame)
root.mainloop()