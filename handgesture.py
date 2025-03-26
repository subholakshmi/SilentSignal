import cv2
import mediapipe as mp
import pyttsx3
import time
engine = pyttsx3.init()
engine.setProperty('rate', 150)  
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    max_num_hands=1 
)
def classify_gesture(hand_landmarks, img_shape):
    tip_ids = [4, 8, 12, 16, 20]  
    fingers = []
    height, width, _ = img_shape
    for i in range(1, 5): 
        finger_tip = hand_landmarks.landmark[tip_ids[i]]
        finger_pip = hand_landmarks.landmark[tip_ids[i] - 2]
        if i == 1:  
            finger_up = finger_tip.y < finger_pip.y and abs(finger_tip.x - finger_pip.x) < 0.05
        else:
            finger_up = finger_tip.y < finger_pip.y

        fingers.append(finger_up)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    if hand_landmarks.landmark[17].x < hand_landmarks.landmark[5].x:  
        thumb_up = thumb_tip.x > thumb_ip.x
    else: 
        thumb_up = thumb_tip.x < thumb_ip.x
    
    fingers.insert(0, thumb_up)  
    extended_fingers = sum(fingers)
    if extended_fingers == 5:
        return "Hello 👋", "Hello!"
    elif extended_fingers == 0:
        return "Good Night 🌙", "Good night, sleep well!"
    elif fingers == [1, 0, 0, 0, 0]:  # Only thumb up
        return "Thumbs Up 👍", "Good job!"
    elif fingers == [0, 1, 1, 0, 0]:  # Peace sign
        return "V Sign ✌️", "Good Morning!"
    elif fingers == [0, 1, 0, 0, 0]:  # Pointing
        return "Pointing 👆", "Look there!"
    elif fingers == [0, 1, 1, 1, 0]:  # Three fingers
        return "Three fingers 🖖", "OK!"
    else:
        return "Unknown Gesture", None

def speak(text):
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)

previous_speech = ""
gesture_count = {}
stable_threshold = 15  
last_detection_time = time.time()
cooldown_period = 3  

while True:
    success, img = cap.read()
    if not success:
        print("Failed to Capture Image")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    detected_gesture = "No Hand Detected"
    detected_speech = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            detected_gesture, detected_speech = classify_gesture(hand_landmarks, img.shape)

    cv2.putText(img, f"Gesture: {detected_gesture}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "Press 'q' to quit", (20, img.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if detected_gesture not in gesture_count:
        gesture_count[detected_gesture] = 0
    gesture_count[detected_gesture] += 1

    for gesture in list(gesture_count.keys()):
        if gesture != detected_gesture:
            gesture_count.pop(gesture)

    current_time = time.time()
    if (detected_speech and detected_gesture != "Unknown Gesture" and 
        detected_gesture != "No Hand Detected"):
        if (gesture_count[detected_gesture] >= stable_threshold and 
            detected_speech != previous_speech and
            current_time - last_detection_time > cooldown_period):
            
            speak(detected_speech)
            previous_speech = detected_speech
            gesture_count.clear() 
            last_detection_time = current_time

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping detection...")
        speak("Stopping detection")
        break

cap.release()
cv2.destroyAllWindows()