import cv2
import os
import numpy as np
from deepface import DeepFace

# --- 1. Define Constants & Setup ---

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"

# Directory to save screenshots
OUTPUT_DIR = "recognition_screenshots"

# Number of screenshots to capture
MAX_SCREENSHOTS = 3

# --- 2. Create Output Directory ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# --- 3. Build DeepFace Model (Warm-up) ---
print("Warming up DeepFace model...")
print("This may take a few minutes as it downloads the model...")

# Find the first image in the known_faces directory for the warm-up
model_built = False
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        first_face_path = os.path.join(KNOWN_FACES_DIR, filename)
        try:
            # Verify the first image against itself to build the model
            DeepFace.verify(img1_path=first_face_path, 
                            img2_path=first_face_path, 
                            enforce_detection=False,
                            model_name='VGG-Face') # You can change the model
            model_built = True
            print("DeepFace model is ready.")
            break
        except Exception as e:
            print(f"Error during warm-up: {e}")
            break
            
if not model_built and len(os.listdir(KNOWN_FACES_DIR)) > 0:
    print("Could not build model. Is 'known_faces' empty or is there another issue?")
    exit()
elif len(os.listdir(KNOWN_FACES_DIR)) == 0:
    print(f"Error: The '{KNOWN_FACES_DIR}' directory is empty.")
    print("Please add at least one image of a known face to continue.")
    exit()


# --- 4. Load OpenCV Face Detector ---
# We use OpenCV's fast Haar Cascade for *detection*
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Error: Could not load Haar Cascade file.")
    exit()

# --- 5. Initialize Webcam ---
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

screenshot_count = 0
print("Starting video stream... Press 's' to save a screenshot. Press 'q' to quit.")

# --- 6. Process Video Stream (Main Loop) ---
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
        
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    # This finds the (x, y, w, h) coordinates of all faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through each face found
    for (x, y, w, h) in faces:
        
        # Crop the face from the frame
        # Add some padding to capture the whole face
        padding = 10
        face_crop = frame[y-padding:y+h+padding, x-padding:x+w+padding]

        name = "Unknown"
        
        if face_crop.size == 0:
            continue # Skip if crop is empty

        try:
            # Use DeepFace.find() to recognize the face
            # It compares the 'face_crop' against all images in 'KNOWN_FACES_DIR'
            # We use 'dfs' (DataFrames) as the variable name
            dfs = DeepFace.find(img_path=face_crop,
                                db_path=KNOWN_FACES_DIR,
                                enforce_detection=False,
                                detector_backend='skip', # We already detected
                                model_name='VGG-Face', # Must match warm-up
                                silent=True) 

            # DeepFace.find returns a list of dataframes. We check the first one.
            if not dfs[0].empty:
                # Get the 'identity' (file path) of the best match
                best_match_path = dfs[0].iloc[0]['identity']
                
                # Get the name from the filename
                name = os.path.splitext(os.path.basename(best_match_path))[0]

        except Exception as e:
            # This can happen if DeepFace fails to process the crop
            # print(f"Error during recognition: {e}")
            name = "Unknown" # Stay unknown if error

        # --- Draw the Results ---
        # Draw a box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (x, y+h - 35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (x + 6, y+h - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition (DeepFace)', frame)

    # --- 7. Add Controls ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('s'):
        if screenshot_count < MAX_SCREENSHOTS:
            screenshot_count += 1
            filename = f"recognition_{screenshot_count}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(save_path, frame)
            print(f"Saved screenshot: {save_path}")
            if screenshot_count == MAX_SCREENSHOTS:
                print(f"All {MAX_SCREENSHOTS} screenshots captured.")
        else:
            print("Already saved max screenshots.")

# --- 8. Release Resources ---
video_capture.release()
cv2.destroyAllWindows()
print("Video stream stopped.")