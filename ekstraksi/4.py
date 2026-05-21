from skimage import io, feature
import matplotlib.pyplot as plt
# Muat gambar
image = io.imread('14.jpg', as_gray=True) # Use an image with varied textures (e.g., fabric, wood, or leaves)
# Terapkan Pola Biner Lokal (LBP)
lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
# Tampilkan gambar asli dan hasil LBP
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('LBP')
plt.imshow(lbp, cmap='gray')
plt.show()
