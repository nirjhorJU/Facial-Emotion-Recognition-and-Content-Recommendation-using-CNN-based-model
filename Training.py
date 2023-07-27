import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
        

face_emotion_detection_model = Sequential()

face_emotion_detection_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
face_emotion_detection_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
face_emotion_detection_model.add(MaxPooling2D(pool_size=(2, 2)))
face_emotion_detection_model.add(Dropout(0.25))

face_emotion_detection_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
face_emotion_detection_model.add(MaxPooling2D(pool_size=(2, 2)))
face_emotion_detection_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
face_emotion_detection_model.add(MaxPooling2D(pool_size=(2, 2)))
face_emotion_detection_model.add(Dropout(0.25))

face_emotion_detection_model.add(Flatten())
face_emotion_detection_model.add(Dense(1024, activation='relu'))
face_emotion_detection_model.add(Dropout(0.5))
face_emotion_detection_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

face_emotion_detection_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the neural network/model
face_emotion_detection_model_info = face_emotion_detection_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# save model structure in json file
model_json = face_emotion_detection_model.to_json()
with open("face_emotion_detection_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
face_emotion_detection_model.save_weights('face_emotion_detection_model.h5')

# Plot model loss and accuracy during training
fig, axs = plt.subplots(1,2,figsize=(15,5))
axs[0].plot(face_emotion_detection_model_info.history["loss"], label="Training Loss")
axs[0].plot(face_emotion_detection_model_info.history["val_loss"], label="Validation Loss")
axs[0].set_title("Model Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()

axs[1].plot(face_emotion_detection_model_info.history["accuracy"], label="Training Accuracy")
axs[1].plot(face_emotion_detection_model_info.history["val_accuracy"], label="Validation Accuracy")
axs[1].set_title("Model Accuracy")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
axs[1].legend()
plt.show()