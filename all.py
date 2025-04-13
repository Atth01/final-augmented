import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

def augment_category(category_folder, output_folder, target_count=200):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for label in ['sesuai', 'tidaksesuai']:
        input_path = os.path.join(category_folder, label)
        output_path = os.path.join(output_folder, label)
        os.makedirs(output_path, exist_ok=True)

        images = [img for img in os.listdir(input_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        current_count = len(images)
        images_to_generate = target_count - current_count
        aug_per_image = max(1, images_to_generate // current_count) if current_count > 0 else 0

        print(f"\n[{category_folder.split('_')[-1].upper()} - {label.upper()}] "
              f"Mengaugmentasi {current_count} gambar × {aug_per_image} → ~{images_to_generate} gambar")

        total_generated = 0
        for img_name in tqdm(images):
            img_path = os.path.join(input_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            x = img.reshape((1,) + img.shape)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_path,
                                      save_prefix="aug", save_format='jpg'):
                i += 1
                total_generated += 1
                if i >= aug_per_image or total_generated >= images_to_generate:
                    break
            if total_generated >= images_to_generate:
                break

# Jalankan augmentasi untuk ketiga kategori
augment_category("almamater", "augmented_almamater")
augment_category("dasi", "augmented_dasi")
augment_category("tidaksesuaihari", "augmented_tidaksesuai_hari")
