import cv2
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
drawSpecific = mp.solutions.pose
mp_pose = mp.solutions.pose

def distanceCalculate(p1, p2):
    """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis


pushUpStart = 0
pushUpCount = 0

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture("/home/code/Videos/records/2.mp4")
cv2.namedWindow("MediaPipe Pose",0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)
    image_height, image_width, _ = image.shape

    nosePoint = (int(results.pose_landmarks.landmark[0].x*image_width), int(results.pose_landmarks.landmark[0].y*image_height))
    leftWrist = (int(results.pose_landmarks.landmark[15].x*image_width), int(results.pose_landmarks.landmark[15].y*image_height))
    rightWrist = (int(results.pose_landmarks.landmark[16].x*image_width), int(results.pose_landmarks.landmark[16].y*image_height))
    leftShoulder = (int(results.pose_landmarks.landmark[11].x*image_width), int(results.pose_landmarks.landmark[11].y*image_height))
    rightShoulder = (int(results.pose_landmarks.landmark[12].x*image_width), int(results.pose_landmarks.landmark[12].y*image_height))

  

    
    if distanceCalculate(rightShoulder,rightWrist)<130:
        pushUpStart = 1
    elif pushUpStart and distanceCalculate(rightShoulder,rightWrist)>250:
        pushUpCount = pushUpCount + 1
        pushUpStart = 0

    print(pushUpCount)

    font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
    org = (50, 100)
  
# fontScale
    fontScale = 2
   
# Blue color in BGR
    color = (255, 0, 0)
  
# Line thickness of 2 px
    thickness = 3

    image = cv2.putText(image, "Push-up count:  " + str(pushUpCount), org, font, fontScale, color, thickness, cv2.LINE_AA)

      
          

                



 
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(1) & 0xFF == 27:
      break
    time.sleep(0.01)
cap.release()
