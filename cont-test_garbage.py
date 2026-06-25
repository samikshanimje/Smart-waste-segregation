import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("waste_classifier.h5")

# Class names (must match folder names used in training)
classes = ["metal_ewaste", "organic_wet", "paper", "plastic"]

# Open laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize image to match model input
    img = cv2.resize(frame, (224, 224))

    # Normalize image
    img = img / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = prediction[0][class_index]

    label = classes[class_index]

    # Display prediction
    text = f"{label} ({confidence*100:.2f}%)"

    cv2.putText(frame, text, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Waste Detection Camera", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()