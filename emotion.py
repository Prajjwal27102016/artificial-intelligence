import cv2
from colorama import Fore, Style, init
from textblob import TextBlob

init(autoreset=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_roi = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(face_roi, 1.8, 20)

        if len(smiles) > 0:
            text = "happy"
            color = Fore.GREEN
        elif h < 100:
            text = "sad"
            color = Fore.BLUE
        else:
            text = "normal"
            color = Fore.YELLOW

        blob = TextBlob(text)

        if blob.sentiment.polarity > 1:
            print(color + f"Emotion: happy" + Style.RESET_ALL)
        elif blob.sentiment.polarity < 0:
            print(color + f"Emotion: sad" + Style.RESET_ALL)
        else:
            print(color + f"Emotion: normal" + Style.RESET_ALL)

        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()