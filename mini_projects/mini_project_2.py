import cv2

# import os

# print("뭔데", os.getcwd())

CARD = "./images/card_2.png"

img = cv2.imread(CARD)
copied = img.copy()
gray = cv2.cvtColor(copied, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierachy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

idx = 1
for ctr in contours:
    if cv2.contourArea(ctr) > 25000:
        x, y, width, height = cv2.boundingRect(ctr)
        cv2.rectangle(
            copied, (x, y), (x + width, y + height), (0, 255, 0), 1, cv2.LINE_AA
        )
        crop = img[y : y + height, x : x + width]
        cv2.imshow(f"{idx}번째 카드", crop)
        idx += 1

cv2.imshow("contour", copied)
cv2.waitKey(0)
cv2.destroyAllWindows()
