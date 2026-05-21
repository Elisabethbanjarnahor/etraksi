import cv2
import matplotlib.pyplot as plt
# Muat gambar
image = cv2.imread('cat1.jpg', cv2.IMREAD_GRAYSCALE)
# Terapkan detektor tepi Canny
edges = cv2.Canny(image, 100, 200)
# Tampilkan gambar asli dan tepinya
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Edges')
plt.imshow(edges, cmap='gray')
plt.show()
