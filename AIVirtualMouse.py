import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui
import math

# Hand Detector Class
class HandDetector:
    def __init__(self, mode=True, maxHands=2, detectionCon=0.7, trackCon=0.7):  # mode=True changed here
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            self.results = self.hands.process(imgRGB)
        except Exception:
            self.results = None

        if self.results and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findDistance(self, p1, p2, img=None, draw=True):
        if len(self.lmList) > max(p1, p2):
            x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
            x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
            length = math.hypot(x2 - x1, y2 - y1)

            if img is not None and draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)

            return length
        return None


def main():
    wCam, hCam = 640, 480
    frameR = 100  # Frame reduction for bounding box
    smoothening = 5  # Smaller means smoother
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = HandDetector(mode=True, maxHands=1)  # mode=True added here
    screenW, screenH = pyautogui.size()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) == 21:
            # Landmarks for index and middle finger tips
            x1, y1 = lmList[8][1], lmList[8][2]   # Index finger tip
            x2, y2 = lmList[12][1], lmList[12][2] # Middle finger tip

            # Check which fingers are up (thumb excluded for simplicity)
            fingers = []
            # Thumb logic simplified: compare tip and IP joint x to detect if thumb is open (works better on right hand)
            if lmList[4][1] > lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # For other fingers (index, middle, ring, pinky)
            for tipId in [8, 12, 16, 20]:
                if lmList[tipId][2] < lmList[tipId - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            index_up = fingers[1]
            middle_up = fingers[2]

            # Moving mode - index finger up, middle finger down
            if index_up == 1 and middle_up == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, screenW))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, screenH))

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                pyautogui.moveTo(screenW - clocX, clocY)

                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # Click mode - index and middle fingers up and close together
            elif index_up == 1 and middle_up == 1:
                length = detector.findDistance(8, 12, img)
                if length is not None and length < 40:
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 0, 255), cv2.FILLED)
                    pyautogui.click()
                    time.sleep(0.3)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("AI Virtual Mouse", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui
import math

# Hand Detector Class
class HandDetector:
    def __init__(self, mode=True, maxHands=2, detectionCon=0.7, trackCon=0.7):  # mode=True changed here
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            self.results = self.hands.process(imgRGB)
        except Exception:
            self.results = None

        if self.results and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findDistance(self, p1, p2, img=None, draw=True):
        if len(self.lmList) > max(p1, p2):
            x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
            x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
            length = math.hypot(x2 - x1, y2 - y1)

            if img is not None and draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)

            return length
        return None


def main():
    wCam, hCam = 640, 480
    frameR = 100  # Frame reduction for bounding box
    smoothening = 5  # Smaller means smoother
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = HandDetector(mode=True, maxHands=1)  # mode=True added here
    screenW, screenH = pyautogui.size()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) == 21:
            # Landmarks for index and middle finger tips
            x1, y1 = lmList[8][1], lmList[8][2]   # Index finger tip
            x2, y2 = lmList[12][1], lmList[12][2] # Middle finger tip

            # Check which fingers are up (thumb excluded for simplicity)
            fingers = []
            # Thumb logic simplified: compare tip and IP joint x to detect if thumb is open (works better on right hand)
            if lmList[4][1] > lmList[3][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # For other fingers (index, middle, ring, pinky)
            for tipId in [8, 12, 16, 20]:
                if lmList[tipId][2] < lmList[tipId - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            index_up = fingers[1]
            middle_up = fingers[2]

            # Moving mode - index finger up, middle finger down
            if index_up == 1 and middle_up == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, screenW))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, screenH))

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                pyautogui.moveTo(screenW - clocX, clocY)

                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # Click mode - index and middle fingers up and close together
            elif index_up == 1 and middle_up == 1:
                length = detector.findDistance(8, 12, img)
                if length is not None and length < 40:
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 0, 255), cv2.FILLED)
                    pyautogui.click()
                    time.sleep(0.3)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("AI Virtual Mouse", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
