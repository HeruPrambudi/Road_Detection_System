import cv2
import numpy as np

# Fungsi callback untuk trackbar
def nothing(x):
    pass

# Membuat jendela
cv2.namedWindow('image')

# Membuat trackbar untuk nilai H, S, dan V
cv2.createTrackbar('H_low', 'image', 0, 179, nothing)
cv2.createTrackbar('S_low', 'image', 0, 255, nothing)
cv2.createTrackbar('V_low', 'image', 0, 255, nothing)
cv2.createTrackbar('H_high', 'image', 0, 179, nothing)
cv2.createTrackbar('S_high', 'image', 0, 255, nothing)
cv2.createTrackbar('V_high', 'image', 0, 255, nothing)

# Inisialisasi nilai trackbar
cv2.setTrackbarPos('H_low', 'image', 0)
cv2.setTrackbarPos('S_low', 'image', 0)
cv2.setTrackbarPos('V_low', 'image', 0)
cv2.setTrackbarPos('H_high', 'image', 179)
cv2.setTrackbarPos('S_high', 'image', 255)
cv2.setTrackbarPos('V_high', 'image', 255)

# Membaca gambar
img = cv2.imread('malam1.jpg')

while True:
    # Konversi gambar ke HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mendapatkan nilai trackbar saat ini
    h_low = cv2.getTrackbarPos('H_low', 'image')
    s_low = cv2.getTrackbarPos('S_low', 'image')
    v_low = cv2.getTrackbarPos('V_low', 'image')
    h_high = cv2.getTrackbarPos('H_high', 'image')
    s_high = cv2.getTrackbarPos('S_high', 'image')
    v_high = cv2.getTrackbarPos('V_high', 'image')

    # Membuat lower dan upper threshold dari nilai HSV yang didapat dari trackbar
    lower_hsv = np.array([h_low, s_low, v_low])
    upper_hsv = np.array([h_high, s_high, v_high])

    # Thresholding gambar menggunakan nilai HSV yang diatur
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(img, img, mask=mask)

    # Menampilkan gambar
    cv2.imshow('image', np.hstack([img, result]))

    # Exit dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
