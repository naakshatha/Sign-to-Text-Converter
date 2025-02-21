import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
try:
    model_dict = pickle.load(open('C:/Users/akshu/Desktop/data/model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Label dictionary for gestures
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


def debug_features(data_aux, label=None):
    """Helper to log features for debugging."""
    print(f"\n[Debug] Features for gesture {label or 'Unknown'}:")
    print(f"Normalized features: {data_aux}")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Warning: Failed to capture frame. Retrying...")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmark positions
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize landmarks relative to bounding box
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)

            # Normalize all landmarks
            for i in range(len(x_)):
                normalized_x = (x_[i] - min_x) / (max_x - min_x) if max_x != min_x else 0
                normalized_y = (y_[i] - min_y) / (max_y - min_y) if max_y != min_y else 0
                data_aux.append(normalized_x)
                data_aux.append(normalized_y)

        debug_features(data_aux)  # Log features for this frame

        # Draw bounding box and landmarks for debugging
        x1 = int(min_x * W) - 10
        y1 = int(min_y * H) - 10
        x2 = int(max_x * W) + 10
        y2 = int(max_y * H) + 10

        # Visualize landmarks as small circles for debugging
        for i in range(0, len(x_), 2):
            x_circ = int(x_[i] * W)
            y_circ = int(y_[i] * H)
            cv2.circle(frame, (x_circ, y_circ), 5, (255, 0, 0), -1)  # Red dots for landmarks

        try:
            # Reshape data for prediction
            data_aux_array = np.asarray(data_aux).reshape(1, -1)
            print(f"Input Features Shape: {data_aux_array.shape}")

            # Predict gesture
            prediction = model.predict(data_aux_array)
            print(f"Prediction Index: {prediction[0]}")

            # If model supports probabilities, print them
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(data_aux_array)
                print(f"Prediction Probabilities: {probabilities}")

            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")
            print(f"Predicted Character: {predicted_character}")
        except ValueError as ve:
            print(f"ValueError during prediction: {ve}")
            predicted_character = "Error"
        except Exception as e:
            print(f"General Error during prediction: {e}")
            predicted_character = "Error"

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
