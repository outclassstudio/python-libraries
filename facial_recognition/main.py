import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()

# haarcascade 검출기 객체 선언
face_cascade = cv2.CascadeClassifier("./cascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./cascade/haarcascade_eye.xml")
name = "Face Recognition"
cv2.namedWindow(name)

# cv2.createTrackbar("scale", name, 2, 5, lambda x: x)

while True:
    ret, frame = cap.read()
    # 흑백으로 이미지를 바꿔서 처리 -> 속도가 빨라짐
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scale_bar = cv2.getTrackbarPos("scale", name)

    # scaleFactor: 1에 가까우면 검출율이 향상, 속도가 느려짐
    # minNeighbors: 높으면 검출율이 높아지지만 얼굴이 아닌 부분도 얼굴이라 인식 할 수 있음
    # minSize: 얼마나 작은 영역까지 탐지할 것인지
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    eyes = eye_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5)
    )

    if not ret:
        break

    if len(faces):
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(
                frame, "My face", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1
            )

    if len(eyes):
        for x, y, w, h in eyes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow(name, frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
