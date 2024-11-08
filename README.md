# DRIVEGAURD
Driver Monitoring & Parking System (Demo video's are attached as mp4 files)

Abstract
   Every day, countless accidents happen due to driver drowsiness, mental strain, and long hours of driving without breaks. Fatigued drivers often continue on the road, even in heavy traffic, unable to find a place to rest—resulting in injuries and tragic fatalities.

Our solution offers a comprehensive, affordable, and accessible Driver Monitoring & Parking System designed for everyone—not just luxury car owners. This product provides real-time driver monitoring for fatigue detection, emotion analysis, and head pose tracking using a webcam feed. To further ensure safety and convenience, it also includes a parking space management system, allowing users to easily locate and book nearby parking spots.

With this system, we aim to make roads safer by reducing the risks associated with driver fatigue and offering accessible parking options for all drivers.

Features
Driver Monitoring
    Detects eye closure to identify signs of drowsiness, with a customizable alert threshold.
    Plays an alert sound when drowsiness is detected. If the issue persists, an automated call is initiated using Twilio.
    Detects emotions using DeepFace, displaying the driver’s dominant emotion.
    Tracks head pose (left, right, up, down) using MediaPipe to monitor where the driver is looking.
    
Parking System
    Displays available parking spaces for multiple locations.
    Allows users to book or release parking spaces dynamically, with real-time updates.
Requirements
    To run the application, install the necessary dependencies:
Libraries
    pip install streamlit opencv-python-headless numpy deepface mediapipe pygame twilio
Usage
    Run the Streamlit application with the following command:
                streamlit run app.py
Flowchart
![drive_guard_architecture 1](https://github.com/user-attachments/assets/8ba223d9-012d-4d43-9309-9ea07ec84949)


Code Explanation
Driver Monitoring Page
    The driver monitoring page includes:

Drowsiness Detection: Detects closed eyes using OpenCV’s CascadeClassifier for eyes and triggers an alarm after a set duration (adjustable through the sidebar).
Alert System: Plays an alert sound if drowsiness is detected. A Twilio call is made if multiple consecutive alerts occur.
Emotion Detection: Uses DeepFace to analyze the webcam feed and detect the driver’s dominant emotion.
Head Pose Estimation: Uses MediaPipe’s FaceMesh and OpenCV's solvePnP to determine the driver’s head orientation.

Code Snippets
eye_cascade: Initializes the Haar Cascade classifier for detecting eyes.
make_twilio_call(): Initiates a call using Twilio when an alert threshold is reached.
analyze_emotion_from_frame(): Detects emotions using the DeepFace.analyze() function.

Parking System Page
The parking system page includes:
Parking Space Display: Shows available spaces for each location.
Booking and Releasing Spaces: Users can book or release spaces for selected locations. This updates the session state to track available parking spaces.

Example Code Snippets:
st.session_state.parking_data: Stores initial parking data in the session state.
if page == "Parking System": Contains the layout for the parking system and logic for booking or releasing a parking space.
Files and Configurations
Alert Sound: Users can upload custom alert sounds via the sidebar.
Twilio Integration: Twilio account_sid and auth_token are used for initiating calls.
