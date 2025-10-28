import face_recognition
import cv2
import os
import numpy as np

# --- 1. Define Constants & Setup ---

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"

# Directory to save screenshots
OUTPUT_DIR = "recognition_screenshots"

# Number of screenshots to capture
MAX_SCREENSHOTS = 3

# Frame resize scale for faster processing (0.25 = 1/4 size)
RESIZE_SCALE = 0.25 

# --- 2. Create Output Directory ---

# Check if the output directory exists, if not, create it
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# --- 3. Load Known Faces ---
print("Loading known faces...")

known_face_encodings = []
known_face_names = []

# Loop through each file in the 'known_faces' directory
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # Load the image
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        known_image = face_recognition.load_image_file(image_path)
        
        # Get face encodings (assume one face per image)
        try:
            encoding = face_recognition.face_encodings(known_image)[0]
            # Get the name from the filename (e.g., "your_name.jpg" -> "your_name")
            name = os.path.splitext(filename)[0]
            
            known_face_encodings.append(encoding)
            known_face_names.append(name)
        except IndexError:
            print(f"Warning: No face found in {filename}. Skipping.")

print(f"Loaded {len(known_face_names)} known faces.")

# --- 4. Initialize Webcam ---

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

screenshot_count = 0
print("Starting video stream... Press 's' to save a screenshot. Press 'q' to quit.")

# --- 5. Process Video Stream (Main Loop) ---
while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    # --- 6. Display the Results ---
    
    # Scale factor for drawing boxes (inverse of RESIZE_SCALE)
    scale_up = int(1 / RESIZE_SCALE) 
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= scale_up
        right *= scale_up
        bottom *= scale_up
        left *= scale_up

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition', frame)

    # --- 7. Add Controls ---

    key = cv2.waitKey(1) & 0xFF

    # Hit 'q' to quit
    if key == ord('q'):
        break

    # Hit 's' to save a screenshot
    if key == ord('s'):
        if screenshot_count < MAX_SCREENSHOTS:
            screenshot_count += 1
            # Create the full save path
            filename = f"recognition_{screenshot_count}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            # Save the current frame
            cv2.imwrite(save_path, frame)
            print(f"Saved screenshot: {save_path}")
            
            if screenshot_count == MAX_SCREENSHOTS:
                print(f"All {MAX_SCREENSHOTS} screenshots captured. Press 'q' to quit.")
        else:
            print(f"Already saved {MAX_SCREENSHOTS} screenshots. Press 'q' to quit.")

# --- 8. Release Resources ---
video_capture.release()
cv2.destroyAllWindows()
print("Video stream stopped.")