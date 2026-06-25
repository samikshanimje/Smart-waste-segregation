import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("waste_classifier.h5")

# Class labels (same order as training folders)
classes = ["metal_ewaste", "organic_wet", "paper", "plastic"]

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'C' to capture image")
print("Press 'Q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    # Capture image
    if key == ord('c'):
        img = cv2.resize(frame, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]

        label = classes[class_index]

        print(f"Prediction: {label} ({confidence*100:.2f}%)")

        cv2.putText(frame, f"{label} ({confidence*100:.2f}%)",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Prediction", frame)
        cv2.waitKey(3000)

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()