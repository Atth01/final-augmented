# augment_tidaksesuaihari.py
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

input_base = "tidaksesuaihari"
output_base = "augmented/tidaksesuaihari"
jumlah_augmentasi = 30

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode="nearest"
)

for kategori in ['sesuai', 'tidaksesuai']:
    input_folder = os.path.join(input_base, kategori)
    output_folder = os.path.join(output_base, kategori)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_folder, filename)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_folder,
                                      save_prefix=f"{kategori}_aug", save_format='jpg'):
                i += 1
                if i >= jumlah_augmentasi:
                    break
