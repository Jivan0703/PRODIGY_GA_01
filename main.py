import keras_cv
import matplotlib.pyplot as plt

# Enable mixed precision and XLA for performance boost
from tensorflow import keras
keras.mixed_precision.set_global_policy("mixed_float16")

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)
images = model.text_to_image("Teddy bears conducting machine learning research", batch_size=3)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()

plot_images(images)
