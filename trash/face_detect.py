import cv2


def detect():
    face_cascade = cv2.CascadeClassifier('/home/roit/wkpgs/Anaconda3/envs/pt/lib/python3.6/site-packages/cv2/data/haarcascade_frontalcatface.xml')
    eye_cascade = cv2.CascadeClassifier('/home/roit/wkpgs/Anaconda3/envs/pt/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')

    camera = cv2.VideoCapture(0)
    while (True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 检测眼睛
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('camera', frame)

        if cv2.waitKey(1000 // 12) & 0xff == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()