import numpy as np
import cv2
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

# Memuat gambar dari direktori Colab
query_img = cv2.imread('test1.jpg')
train_img = cv2.imread('test11.jpg')

# Konversi gambar ke grayscale (hitam putih)
query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# Inisialisasi detektor ORB
orb = cv2.ORB_create()

# Mendeteksi titik kunci (Keypoints) dan Deskriptor
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

# Proses pencocokan (Matching)
matcher = cv2.BFMatcher()
matches = matcher.match(queryDescriptors, trainDescriptors)

# Menggambar hasil pencocokan (20 kecocokan pertama)
final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:20], None)
final_img = cv2.resize(final_img, (1000, 650))

# Menampilkan hasil
plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title("Feature Matches")
plt.axis('off')
plt.show()
