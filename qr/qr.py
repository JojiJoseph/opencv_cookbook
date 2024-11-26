import cv2

detector = cv2.QRCodeDetector()

img = cv2.imread("./qr.png") # https://www.the-qrcode-generator.com/
retval, points, _ = detector.detectAndDecode(img)

points = points.astype(int).reshape((-1,1,2))
img = cv2.drawContours(img,[points], -1, (255,0,0), 2)
cv2.putText(img, retval, (10,15), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)
cv2.imshow("QR", img)
cv2.waitKey()

print("Detected Value: ", retval)