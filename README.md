# Real-Time Face Recognition Lab (COS30082)

This project is a Python application for a university assignment. It uses a live webcam feed to perform real-time face detection and recognition, identifying "known" faces from a local folder.

This implementation uses the `DeepFace` library, which provides high-accuracy, modern face recognition models (such as VGG-Face) with a TensorFlow backend.

## Technology Used

* **Python**
* **OpenCV** (for webcam access, face detection, and drawing on the screen)
* **DeepFace** (for the core face recognition/verification)
* **TensorFlow/Keras** (as the backend for DeepFace)
* **NumPy**

## Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/cos30082-face-recognition.git](https://github.com/your-username/cos30082-face-recognition.git)
    cd cos30082-face-recognition
    ```

2.  **(Recommended) Create a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    All dependencies are listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **First-Run Notice:**
    The first time you execute the script, `DeepFace` will automatically download the pre-trained model files (e.g., `VGG-Face`). This requires an internet connection and may take a few minutes. This is a one-time setup.

## Usage

1.  **Add Known Faces:**
    Place one or more images (e.g., `.jpg`, `.png`) of the people you want to recognize into the `known_faces/` folder.
    * The script will use the filename as the person's name (e.g., `john_smith.jpg` will be recognized as "john_smith").

2.  **Run the application:**
    ```bash
    python face_recognition_lab.py
    ```

3.  **Controls:**
    * A window will open showing your webcam feed.
    * Faces found in the `known_faces` folder will be labeled with their name.
    * Other faces will be labeled as "Unknown".
    * Press the **'s'** key to save a screenshot to the `recognition_screenshots/` folder (up to 3 screenshots).
    * Press the **'q'** key to close the application.

## Project Structure