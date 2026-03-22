import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Dosyayı oku
with rasterio.open('data/Site04_final_adj_5mpp_surf.tif') as src:
    heightmap = src.read(1).astype(float)

# Geçersiz değerleri temizle
heightmap[heightmap < -9000] = np.nan

# Ekranda göster
plt.figure(figsize=(10, 8))
plt.imshow(heightmap, cmap='gray')
plt.colorbar(label='Yükseklik (metre)')
plt.title('Shackleton Krateri — Ay Yüzeyi Yükseklik Haritası')
plt.savefig('gorsel/harita.png')
plt.show()