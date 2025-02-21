import os
import cv2

# Directory to store collected images
DATA_DIR = 'C:/Users/akshu/Desktop/data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Configuration
number_of_classes = 26
dataset_size = 100

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change index if necessary
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Collect data for each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}. Press "q" to start.')

    # Wait for the user to signal readiness
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Collect images for the class
    counter = 0
    print(f'Starting image capture for class {j}...')
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        # Display the frame
        cv2.putText(frame, f'Capturing {counter + 1}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Save the image
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture stopped by user")
            break

print("Data collection completed.")

# Release resources
cap.release()
cv2.destroyAllWindows()
