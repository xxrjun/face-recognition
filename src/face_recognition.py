from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
from face_comparison import compare_face
from face_detection import face_detect, face_detect_bgr
from feature_extraction import feature_extract
from utils import detector, sess, db_path
import winsound


def recognize_image():
    """
    Recognize faces in an image selected by the user using a file dialog.

    :return: None
    """

    print('Recognizing image...')

    # Open a file dialog to select the image file
    img_path = filedialog.askopenfilename()

    # Detect faces in the image
    img_rgb, detections = face_detect(img_path, detector)

    # Extract features from the detected faces
    position, landmarks, embeddings = feature_extract(img_rgb, detections, sess)

    # Set a threshold for face recognition
    threshold = 1

    # Compare the embeddings of the detected faces with the faces in the database
    for i, embedding in enumerate(embeddings):
        name, distance, total_result = compare_face(embedding, threshold, db_path)

        # Draw a rectangle around the recognized face and put the name of the person
        cv2.rectangle(img_rgb, (position[i][0], position[i][1]), (position[i][2], position[i][3]), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_rgb, name + ', ' + str(distance), (position[i][0] + 10, position[i][1] - 10), font, 0.8,
                    (0, 255, 0), 2)

        print(total_result)

    # Display the image with recognized faces
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb / 255)
    _ = plt.axis('off')
    plt.show()


def recognize_video():
    """
    Recognize faces in a video selected by the user using a file dialog.

    :return: None
    """

    print('Recognizing video...')

    # Open a file dialog to select the video file
    video_path = filedialog.askopenfilename()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    found = False
    count = 0

    while True and not found:  # read every frame to play video
        ret, frame = cap.read()  # return two values(boolean, next frame)
        if ret:
            # Detect faces in the frame
            img_rgb, detections = face_detect_bgr(frame, detector)

            if count == fps // 3:
                count = 0

                # Extract features from the detected faces
                position, landmarks, embeddings = feature_extract(img_rgb, detections, sess)

                threshold = 1
                for i, embedding in enumerate(embeddings):
                    name, distance, total_result = compare_face(embedding, threshold, db_path)
                    if distance < threshold:
                        # Draw a rectangle around the recognized face and put the name of the person
                        cv2.rectangle(img_rgb, (position[i][0], position[i][1]), (position[i][2], position[i][3]),
                                      (0, 255, 0), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img_rgb, name + ', ' + str(distance), (position[i][0] + 10, position[i][1] - 10),
                                    font, 0.8, (0, 255, 0), 2)
                        print(name, distance)
                        print('Found the person in the video!')

                        # Show the image with the recognized face
                        plt.figure(figsize=(10, 10))
                        plt.imshow(img_rgb / 255)
                        _ = plt.axis('off')
                        winsound.Beep(600, 1000)
                        plt.show()

                        found = True
                        break

            # Resize the frame and show the video
            frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            cv2.imshow('video', frame)

            count += 1

        else:
            print('No found the person in the video!')
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
