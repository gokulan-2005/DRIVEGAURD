import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import time
import mediapipe as mp
import pygame
from twilio.rest import Client

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Driver Monitoring & Parking System", layout="wide")
pygame.mixer.init()

account_sid = 'AC10ac5972ff1efcadbec7e525e34c0ee5'
auth_token = '642328b333502e99e43b08c120dc3e0b'
twilio_client = Client(account_sid, auth_token)
to_phone_number = '+919942225405'
from_phone_number = '+12568261613'

if 'parking_data' not in st.session_state:
    st.session_state.parking_data = {
        'Erode': {'Total Spaces': 50, 'Available Spaces': 20},
        'Tiruppur': {'Total Spaces': 30, 'Available Spaces': 10},
        'Coimbatore': {'Total Spaces': 40, 'Available Spaces': 25},
    }

def make_twilio_call():
    call = twilio_client.calls.create(
        to=to_phone_number,
        from_=from_phone_number,
        url="http://demo.twilio.com/docs/voice.xml"
    )
    print(f"Call initiated with SID: {call.sid}")

def analyze_emotion_from_frame(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'])
        emotion = analysis[0]['dominant_emotion']
        return emotion
    except Exception as e:
        return "Error"

st.sidebar.title("Settings")
alert_file = st.sidebar.file_uploader("Upload Custom Alert Sound", type=["mp3"])
CLOSED_THRESHOLD = st.sidebar.slider("Eye Closure Alert Threshold (seconds)", 1, 5, 3)

alarm_count = 0

page = st.sidebar.radio("Select Page", ["Driver Monitoring", "Parking System"])

if page == "Driver Monitoring":
    st.title("Driver Monitoring System")

    if st.sidebar.button("Start Monitoring"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            frame_placeholder = st.empty()
            eye_closed_start_time = None
            eye_closed_duration = 0
            stop_button = st.button("Stop Monitoring", key="stop_button")

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Error: Failed to capture image.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

                if len(eyes) == 0:
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()
                    else:
                        eye_closed_duration = time.time() - eye_closed_start_time

                    if eye_closed_duration >= CLOSED_THRESHOLD:
                        cv2.putText(frame, "You are fatigued!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        alarm_count += 1
                        if alert_file:
                            alert_sound = alert_file
                        else:
                            alert_sound = 'alert1.mp3'

                        pygame.mixer.music.load(alert_sound)
                        pygame.mixer.music.play()

                        if alarm_count >= 3:
                            make_twilio_call()
                            alarm_count = 0
                else:
                    eye_closed_start_time = None
                    eye_closed_duration = 0
                    alarm_count = 0

                emotion_label = analyze_emotion_from_frame(frame)

                img_h, img_w, img_c = frame.shape
                face_3d = []
                face_2d = []
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        for idx, lm in enumerate(face_landmarks.landmark):
                            if idx in [33, 263, 1, 61, 291, 199]:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                           [0, focal_length, img_h / 2],
                                           [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
                    if y < -10:
                        head_pose_text = "Looking Right"
                    elif y > 10:
                        head_pose_text = "Looking Left"
                    elif x < -10:
                        head_pose_text = "Looking Down"
                    elif x > 10:
                        head_pose_text = "Looking Up"
                    else:
                        head_pose_text = "Looking Forward"

                    cv2.putText(frame, f"Emotion: {emotion_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Head Pose: {head_pose_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                if stop_button:
                    break

            cap.release()
            cv2.destroyAllWindows()

elif page == "Parking System":
    st.title("Nearby Parking Lots")

    cols = st.columns(3)
    for i, (location, data) in enumerate(st.session_state.parking_data.items()):
        with cols[i]:
            st.header(location)
            st.write(f"*Total Spaces*: {data['Total Spaces']}")
            st.write(f"*Available Spaces*: {data['Available Spaces']}")
            st.progress(data['Available Spaces'] / data['Total Spaces'])
            if st.button(f"Select {location}"):
                st.session_state.selected_location = location
                st.session_state.page = 'booking'
                st.rerun()

    if 'selected_location' in st.session_state:
        selected_location = st.session_state.selected_location
        st.title(f"Booking - {selected_location}")

        location_data = st.session_state.parking_data[selected_location]
        total_spaces = location_data['Total Spaces']
        available_spaces = location_data['Available Spaces']

        st.write(f"*Total Spaces:* {total_spaces}")
        st.write(f"*Available Spaces:* {available_spaces}")
        st.progress(available_spaces / total_spaces)

        if available_spaces > 0 and st.button("Book a Space"):
            st.session_state.parking_data[selected_location]['Available Spaces'] -= 1
            st.success("Your parking space has been booked!")
            st.balloons()
        elif st.button("Leave the Space"):
            if available_spaces < total_spaces:
                st.session_state.parking_data[selected_location]['Available Spaces'] += 1
                st.success("You've left the parking space.")
            else:
                st.warning("All parking spaces are empty.")

        if st.button("Return to Parking Overview"):
            del st.session_state.selected_location
            st.rerun()
