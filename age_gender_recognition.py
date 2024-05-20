import cv2  # Import OpenCV for image processing.
import os  # Import OS for operating system related functionalities.

# Change the working directory to the location where the models are stored.
os.chdir(r'C:\Users\Ahmed\Documents\Portofolio\Age_and_gender_recognition\models')

# Define a function to detect faces using a deep learning model.
def detectFace(net, frame, confidence_threshold=0.8):
    frameOpencvDNN = frame.copy()  # Make a copy of the frame for processing.
    print(frameOpencvDNN.shape)  # Print the shape of the frame.
    frameHeight = frameOpencvDNN.shape[0]  # Get the height of the frame.
    frameWidth = frameOpencvDNN.shape[1]  # Get the width of the frame.
    
    # Create a blob from the frame to pass it through the network.
    blob = cv2.dnn.blobFromImage(frameOpencvDNN, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
    net.setInput(blob)  # Set the input to the network.
    detections = net.forward()  # Perform a forward pass to get the detections.
    
    faceBoxes = []  # Initialize an empty list to store face bounding boxes.
    for i in range(detections.shape[2]):  # Loop through all detections.
        confidence = detections[0, 0, i, 2]  # Get the confidence of the detection.
        if confidence > confidence_threshold:  # If confidence is above the threshold.
            x1 = int(detections[0, 0, i, 3] * frameWidth)  # Calculate the x1 coordinate.
            y1 = int(detections[0, 0, i, 4] * frameHeight)  # Calculate the y1 coordinate.
            x2 = int(detections[0, 0, i, 5] * frameWidth)  # Calculate the x2 coordinate.
            y2 = int(detections[0, 0, i, 6] * frameHeight)  # Calculate the y2 coordinate.
            faceBoxes.append([x1, y1, x2, y2])  # Append the face bounding box to the list.
            # Draw a rectangle around the detected face.
            cv2.rectangle(frameOpencvDNN, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDNN, faceBoxes  # Return the processed frame and face boxes.

# File paths for the models.
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
ageProto = 'age_deploy.prototxt'
ageModel = 'age_net.caffemodel'
genderProto = 'gender_deploy.prototxt'
genderModel = 'gender_net.caffemodel'

# Lists of age and gender labels.
genderList = ['Male', 'Female']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load the pre-trained models.
faceNet = cv2.dnn.readNet(faceModel, faceProto)  # Load the face detection model.
ageNet = cv2.dnn.readNet(ageModel, ageProto)  # Load the age prediction model.
genderNet = cv2.dnn.readNet(genderModel, genderProto)  # Load the gender prediction model.

video = cv2.VideoCapture(0)  # Start video capture from the default camera.
padding = 20  # Define padding for face bounding boxes.

while cv2.waitKey(1) < 0:  # Continuously capture frames from the camera.
    hasFrame, frame = video.read()  # Read a frame from the video.
    if not hasFrame:  # If there are no frames, exit the loop.
        cv2.waitKey()
        break
    
    # Detect faces in the frame.
    resultImg, faceBoxes = detectFace(faceNet, frame)
    
    if not faceBoxes:  # If no faces are detected, print a message.
        print("No face detected")
    
    for faceBox in faceBoxes:  # Loop through all detected faces.
        # Extract the face region with padding.
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1), 
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]
        
        # Create a blob from the face region.
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
        
        # Predict gender.
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Predict age.
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        # Put text for gender and age on the result image.
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Show the result image.
        cv2.imshow("Detecting age and Gender", resultImg)
        
        # Exit loop if 'q' is pressed.
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()  # Destroy all OpenCV windows.
