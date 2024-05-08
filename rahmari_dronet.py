import cv2
import numpy as np
from tensorflow.keras.models import model_from_json


def main():
    
    with open('model_struct.json', 'r') as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)

    model.load_weights('model_weights.h5')

    model.compile(loss='mse', optimizer='sgd')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(grayed, (200, 200))

        norm = resized / 255.0

        image = np.expand_dims(norm, axis=[0, -1])

        prediction = model.predict(image)

        cv2.putText(frame, f"Prediction: {'collision' if prediction[1][0] > 0.8 else 'no collision'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        print('collision' if prediction[1][0] > 0.8 else 'no collision')
        cv2.imshow('DroNet', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
 