import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Konfigurasi
input_base = "C:/python_skripsi/aug-khusus"
output_base = "C:/python_skripsi/aug-khusus/aug-lagi"
target_total = 270

# Generator augmentasi
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop untuk dua kategori
for category in ["Sesuai", "TidakSesuai"]:
    input_dir = os.path.join(input_base, category)
    output_dir = os.path.join(output_base, category.replace(" ", ""))  # Hindari spasi

    os.makedirs(output_dir, exist_ok=True)

    images = [img for img in os.listdir(input_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(images)

    print(f"[{category}] Gambar sumber: {total_images}")

    # Augmentasi yang dibutuhkan per gambar
    if total_images == 0:
        continue

    augment_per_image = (target_total - total_images) // total_images
    sisa_augment = (target_total - total_images) % total_images

    count = 0
    for idx, img_name in enumerate(images):
        img_path = os.path.join(input_dir, img_name)
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Simpan gambar asli
        new_name = f"img_{count+1:04d}.jpg"
        img.save(os.path.join(output_dir, new_name))
        count += 1

        # Hitung berapa augmentasi untuk gambar ini
        aug_count = augment_per_image + (1 if idx < sisa_augment else 0)

        i = 0
        for batch in datagen.flow(x, batch_size=1):
            new_name = f"img_{count+1:04d}.jpg"
            array_to_img(batch[0]).save(os.path.join(output_dir, new_name))
            count += 1
            i += 1
            if i >= aug_count:
                break

    print(f"[{category}] Total gambar setelah augmentasi: {count}")
