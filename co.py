import cv2


image_path = "resim.jpg"


image = cv2.imread(image_path)
if image is None:
    print(" Resim bulunamadı. Yolunu tekrar kontrol etmeliyim.")
    exit()  # Resim yoksa devam edemem

# İnsanlar daha rahat görebilmek için griye çeviriyor
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# OpenCV'nin kullanıldığı kısım
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Şimdi resmi tarayıp insanları bulalım
boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))

# Bounding box yapılan kod parcası
for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Kaç kişi bulduğunu söyleyen kod parçası
print(f"Bu resimdeki insan sayısı: {len(boxes)}")

# Sonucu göstermeme yarayan kod parcası
cv2.imshow("İnsan Tespiti", image)
cv2.waitKey(0)  
cv2.destroyAllWindows()  
