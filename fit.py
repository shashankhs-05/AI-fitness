import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define exercises and corresponding pose landmarks
exercises = {
    'pushup': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW],
    'squat': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE],
    'lunge': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE]
}

# Define a function to estimate exercise based on detected pose landmarks
def estimate_exercise(pose_landmarks):
    for exercise, pose in exercises.items():
        if all(pose_landmarks[landmark].visibility > 0.5 for landmark in pose):
            return exercise
    return 'unknown'

# Initialize Mediapipe pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Start video capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Read frame from video capture
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB and process it with Mediapipe Pose.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = Falseq
        results = pose.process(image)

        # Draw pose landmarks on image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Estimate exercise
            exercise = estimate_exercise(results.pose_landmarks.landmark)
            cv2.putText(image, exercise, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Exercise Estimation', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Release the video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
