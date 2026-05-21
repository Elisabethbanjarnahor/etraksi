import cv2
import matplotlib.pyplot as plt
# Muat gambar
image = cv2.imread('dog2.jpg') # Use an image with distinct features (e.g., a building or a car)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Deteksi titik kunci SIFT
sift = cv2.SIFT_create()
keypoints = sift.detect(gray, None)
# Gambar poin kunci pada gambar
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None,
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# Tampilkan gambar dengan poin kunci
plt.figure(figsize=(10, 5))
plt.title('SIFT Keypoints')
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.show()
