import cv2
import numpy as np
from keras.models import model_from_json
import webbrowser
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


song_dict = {
    "Angry": "https://www.youtube.com/watch?v=zzwRbKI2pn4&ab_channel=CarryMinati",
    "Disgusted": "https://www.youtube.com/watch?v=zWq2TT3ieGE&ab_channel=NewYearsDayVEVO",
    "Fearful": "https://www.youtube.com/watch?v=b5BNUa_op2o&ab_channel=UniversalMelody",
    "Happy": "https://www.youtube.com/watch?v=ZbZSe6N_BXs&ab_channel=PharrellWilliamsVEVO",
    "Neutral": "https://www.youtube.com/watch?v=P5GPsO8yK_E",
    "Sad": "https://www.youtube.com/watch?v=rNnazxtNEKI&ab_channel=ParthDodiya",
    "Surprised": "https://www.youtube.com/watch?v=GWY3XVUcGtM&ab_channel=LBOLyrics"
}


# load json and create model
json_file = open('face_emotion_ditection_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
face_emotion_ditection_model = model_from_json(loaded_model_json)

# load weights into new model
face_emotion_ditection_model.load_weights("face_emotion_ditection_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# cap = cv2.VideoCapture("home\\motiur\\emotion_sample6.mp4")


counter = 0
prev_emotion = None

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = face_emotion_ditection_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        
        emotion = emotion_dict[maxindex]
        cv2.putText(frame, emotion+ " counter: " + str(counter), (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        if emotion == prev_emotion:
            counter += 1
            print(str(counter) + " times " + prev_emotion + " detected, waiting sometimes for better accuracy")
        else:
            counter = 1
            prev_emotion = emotion

        if counter == 50:
            webbrowser.open(song_dict[emotion])
            text = "We detect successfully your emotion : " +emotion +", now watch your designer song! "
            print(text)
            # Language in which you want to convert
            language = 'en'
            # Create a gTTS object
            tts = gTTS(text=text, lang=language, slow=False)
            # Save the audio file
            tts.save("/home/motiur/Documents/hello/output.mp3")
            # load audio file
            audio = AudioSegment.from_mp3("/home/motiur/Documents/hello/output.mp3")
            # play audio file
            play(audio)
            cap.release()
            cv2.destroyAllWindows()
            break
        

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
