import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modelni yuklash
model = load_model("keras_model.h5", compile=False)

# Class nomlarini olish
with open("labels.txt", "r") as f:
    class_names = f.read().splitlines()

# Kamerani yoqish
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = camera.read()
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    image = cv2.resize(frame, (224, 224))
    image = np.asarray(image, dtype=np.float32)
    image = image.reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence = prediction[0][index]

    cv2.putText(
        frame,
        f"{class_name}: {confidence*100:.1f}%",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("FACE_similarity_AI - Python", frame)

    if cv2.waitKey(1) == 27:  # ESC bosilsa chiqadi
        break

camera.release()
cv2.destroyAllWindows()
