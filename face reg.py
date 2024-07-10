#Loading Libarires

import cv2
import numpy as np
from keras.models import load_model


#Functions


# For the Face Detection
def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs



# Setiing up and loading models

faceProto = "D:\\projects @\\face reg\\opencv_face_detector.pbtxt"
faceModel = "D:\\projects @\\face reg\\opencv_face_detector_uint8.pb"

ageProto ="D:\\projects @\\face reg\\age_deploy.prototxt"
ageModel = "D:\\projects @\\face reg\\age_net.caffemodel"

genderProto = "D:\\projects @\\face reg\\gender_deploy.prototxt"
genderModel = "D:\\projects @\\face reg\\gender_net.caffemodel"

expression_model =load_model("D:\\projects @\\face reg\\model_file_30epochs (2).h5")

faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


#Setting Up Labels and Mean values

labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']




# Capturing the the frame from the web cam

video=cv2.VideoCapture(0)

padding=20

while True:
    ret,frame=video.read()
    frame,bboxs=faceBox(faceNet,frame)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for bbox in bboxs:
        # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        
        #For the Gender
        
        # Using the deep learing model of cnn to predict gender
        
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]

        ##For the Age
        
        # Using the deep learing model of cnn to predict gender
        
        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]

        #For the Expression
        
        # Using the keras and tensorflow with a pretrained facial expression recognizer to detect emotions
        
        try:
            sub_face_img=gray[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            resized=cv2.resize(sub_face_img,(48,48))
            normalize=resized/255.0
            reshaped=np.reshape(normalize, (1, 48, 48, 1))
            result=expression_model.predict(reshaped)
            labelE=np.argmax(result, axis=1)[0]
        except:
            print("frame Out of bounds")
        
        
        label="{},{}".format(gender,age)
        
        # Setting up the frame for the output and putting age, gender and emotions
        
        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
        cv2.putText(frame, labels_dict[labelE], (bbox[0], bbox[3]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
