import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import face_recognition
from PIL import Image, ImageDraw


def face_recognitions():
    webcam_video_stream = cv2.VideoCapture(0)

 
    modi_image = face_recognition.load_image_file('D:/face-recognition/images/samples/modi.jpg')
    modi_face_encodings = face_recognition.face_encodings(modi_image)[0]
    
    trump_image = face_recognition.load_image_file('D:/face-recognition/images/samples/trump.jpg')
    trump_face_encodings = face_recognition.face_encodings(trump_image)[0]
    
    abhi_image = face_recognition.load_image_file('D:/face-recognition/images/samples/anush.jpeg')
    abhi_face_encodings = face_recognition.face_encodings(abhi_image)[0]

    known_face_encodings = [modi_face_encodings, trump_face_encodings, abhi_face_encodings]
    known_face_names = ["Narendra Modi", "Donald Trump", "Anushka"]
    

    all_face_locations = []
    all_face_encodings = []
    all_face_names = []
    
    while True:
        ret,current_frame = webcam_video_stream.read()

        current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)

        all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
        
        all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)
    
 
        for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
  
            top_pos,right_pos,bottom_pos,left_pos = current_face_location
     
            top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4

            all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
           

            name_of_person = 'Unknown face'
  
            if True in all_matches:
                first_match_index = all_matches.index(True)
                name_of_person = known_face_names[first_match_index]
            
       
            cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
            
  
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
        
     
        cv2.imshow("Webcam Video",current_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam_video_stream.release()
    cv2.destroyAllWindows()        
def mood_detections():
        webcam_video_stream = cv2.VideoCapture(0)


        face_exp_model = model_from_json(open("D:/face-recognition/dataset/facial_expression_model_structure.json","r").read())
        face_exp_model.load_weights('D:/face-recognition/dataset/facial_expression_model_weights.h5')
        
        emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        
        
        
        all_face_locations = []
        
        while True:
            
            ret,current_frame = webcam_video_stream.read()
            
            current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
            
            
            all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
            
            
            for index,current_face_location in enumerate(all_face_locations):
                
                top_pos,right_pos,bottom_pos,left_pos = current_face_location
                
                top_pos = top_pos*4
                right_pos = right_pos*4
                bottom_pos = bottom_pos*4
                left_pos = left_pos*4
               
                print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
               
                
                current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
                
                
                cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
                
              
                current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
               
                current_face_image = cv2.resize(current_face_image, (48, 48))
               
                img_pixels = image.img_to_array(current_face_image)
               
                img_pixels = np.expand_dims(img_pixels, axis = 0)
               
                img_pixels /= 255 
                
                
                exp_predictions = face_exp_model.predict(img_pixels) 
               
                max_index = np.argmax(exp_predictions[0])
                
                emotion_label = emotions_label[max_index]
                
               
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(current_frame, emotion_label, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
                
           
            cv2.imshow("Webcam Video",current_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
        webcam_video_stream.release()
        cv2.destroyAllWindows()  

def face_detections():
    web_stream=cv2.VideoCapture(0)

    all_face_location=[]
    while True:
        ret,current_frame= web_stream.read()
        small_current_frame= cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
        all_face_location= face_recognition.face_locations(small_current_frame,model='hog')
        for index,current_image_location in enumerate(all_face_location):
            top,right,bottom,left=current_image_location
            top=top*4
            right=right*4
            bottom=bottom*4
            left=left*4
            print("pos of face {} is {}".format((index+1),current_image_location))
            cv2.rectangle(current_frame,(left,top),(right,bottom),(0,255,0),2)
        cv2.imshow("webcan video",current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    web_stream.release()
    cv2.destroyAllWindows()   
def age_and_gender():
    webcam_video_stream = cv2.VideoCapture(0)


    all_face_locations = []

    while True:
        ret,current_frame = webcam_video_stream.read()

        current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)

        all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
        
     
        for index,current_face_location in enumerate(all_face_locations):
      
            top_pos,right_pos,bottom_pos,left_pos = current_face_location
  
            top_pos = top_pos*4
            right_pos = right_pos*4
            bottom_pos = bottom_pos*4
            left_pos = left_pos*4
   
            current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
            
            AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

            current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
   
            gender_label_list = ['Male', 'Female']

            gender_protext = "D:/face-recognition/dataset/gender_deploy.prototxt"
            gender_caffemodel = "D:/face-recognition/dataset/gender_net.caffemodel"
 
            gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)

            gender_cov_net.setInput(current_face_image_blob)
            gender_predictions = gender_cov_net.forward()

            gender = gender_label_list[gender_predictions[0].argmax()]
            

            age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

            age_protext = "D:/face-recognition/dataset/age_deploy.prototxt"
            age_caffemodel = "D:/face-recognition/dataset/age_net.caffemodel"

            age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)

            age_cov_net.setInput(current_face_image_blob)

            age_predictions = age_cov_net.forward()

            age = age_label_list[age_predictions[0].argmax()]
                  

            cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
                
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame, gender+" "+age+"yrs", (left_pos,bottom_pos+20), font, 0.5, (0,255,0),1)
        

        cv2.imshow("Webcam Video",current_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam_video_stream.release()
    cv2.destroyAllWindows()        
def make_up():
    webcam_video_stream = cv2.VideoCapture(0)

    all_face_locations = []
    
    while True:

        ret,current_frame = webcam_video_stream.read()
  
        face_landmarks_list =  face_recognition.face_landmarks(current_frame)
        
  
        pil_image = Image.fromarray(current_frame)
 
        d = ImageDraw.Draw(pil_image)
        
      
        index=0
        while index < len(face_landmarks_list):
      
            for face_landmarks in face_landmarks_list:
              
          
                d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
                d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
                d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
                d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
     
                d.polygon(face_landmarks['top_lip'], fill=(0, 0, 200, 100))
                d.polygon(face_landmarks['bottom_lip'], fill=(0, 0, 200, 100))
                d.line(face_landmarks['top_lip'], fill=(150, 150, 150, 64), width=2)
                d.line(face_landmarks['bottom_lip'], fill=(150, 150, 150, 64), width=2)
          
                d.polygon(face_landmarks['left_eye'], fill=(0, 255, 0, 100))
                d.polygon(face_landmarks['right_eye'], fill=(0, 255, 0, 100))
            
                d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=1)
                d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=1)
    
        
            index +=1

        rgb_image = pil_image.convert('RGB') 
        rgb_open_cv_image = np.array(pil_image)
        
        bgr_open_cv_image = cv2.cvtColor(rgb_open_cv_image, cv2.COLOR_RGB2BGR)
        bgr_open_cv_image = bgr_open_cv_image[:, :, ::-1].copy()

        cv2.imshow("Webcam Video",bgr_open_cv_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam_video_stream.release()
    cv2.destroyAllWindows()       
def land_mark():
    webcam_video_stream = cv2.VideoCapture(0)

    all_face_locations = []
    
    while True:
  
        ret,current_frame = webcam_video_stream.read()
        face_landmarks_list =  face_recognition.face_landmarks(current_frame)
        pil_image = Image.fromarray(current_frame)
    
        d = ImageDraw.Draw(pil_image)
        
        index=0
        while index < len(face_landmarks_list):
    
            for face_landmarks in face_landmarks_list:
              
                
                #join each face landmark points
                d.line(face_landmarks['chin'],fill=(255,255,255), width=2)
                d.line(face_landmarks['left_eyebrow'],fill=(255,255,255), width=2)
                d.line(face_landmarks['right_eyebrow'],fill=(255,255,255), width=2)
                d.line(face_landmarks['nose_bridge'],fill=(255,255,255), width=2)
                d.line(face_landmarks['nose_tip'],fill=(255,255,255), width=2)
                d.line(face_landmarks['left_eye'],fill=(255,255,255), width=2)
                d.line(face_landmarks['right_eye'],fill=(255,255,255), width=2)
                d.line(face_landmarks['top_lip'],fill=(255,255,255), width=2)
                d.line(face_landmarks['bottom_lip'],fill=(255,255,255), width=2)
        
            index +=1  
        rgb_image = pil_image.convert('RGB') 
        rgb_open_cv_image = np.array(pil_image)
        
        # Convert RGB to BGR 
        bgr_open_cv_image = cv2.cvtColor(rgb_open_cv_image, cv2.COLOR_RGB2BGR)
        bgr_open_cv_image = bgr_open_cv_image[:, :, ::-1].copy()
    
        cv2.imshow("Webcam Video",bgr_open_cv_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam_video_stream.release()
    cv2.destroyAllWindows()       
print("\n\n*******************FACE RECOGNITION AND MOOD DETECTION****************\n")
print("                                         By: Anushka Gupta:19001003018\n")
print("                                             Jessica Mishra:19001003051\n")
print("enter the no. corresponding to option you want to chose:\n")
print("1:face recognition\n2:mood detection\n3:age and gender detection\n4:make up\n5:land marks\n")
print("0:exit")
print("\nENTER 'q' TO EXIT THE WEBCAM")
a= int(input("enter no."))

while a!=0:
    if a==1:
        face_recognitions()
    elif a==2:
        mood_detections()
    elif a==3:
        age_and_gender()
    elif a==4:
        make_up()
    elif a==5:
        land_mark()
    else:
        print('\n---*---Try again---*---\n')
    print("enter the no. corresponding to option you want to chose:\n")
    print("1:face recognition\n2:mood detection\n3:age and gender detection\n4:make up\n5:land marks\n")
    print("0:exit")
    print("\nENTER 'q' TO EXIT THE WEBCAM")
    a=int(input("enter no."))
    
    