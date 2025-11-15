from PIL import Image
import random
import os
import torchvision.transforms as t
from torchvision.transforms import ToPILImage
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def visualize(image_path, tensor_image):
    image = Image.open(image_path)
    gray_image = image.convert('L')
    to_pil = ToPILImage()
    final_image = to_pil(tensor_image)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Оригинал")
    axes[0].axis("off")

    axes[1].imshow(gray_image, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(final_image, cmap="gray")
    axes[2].set_title("После преобразований")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("visualize.jpg", dpi=1600, bbox_inches='tight')



def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = t.Compose([
        t.Grayscale(num_output_channels=1),
        t.Resize((100, 100)),
        t.ToTensor(),
        t.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor_image = transform(image)
    to_pil = ToPILImage()
    final_image = to_pil(tensor_image)
    final_image.save(f'images/{image}.jpg')
    return tensor_image


# Photo
rand_folder = random.randint(1, 7)
folder = f'data/DATASET/train/{rand_folder}'

images = [f for f in os.listdir(folder)
          if f.lower().endswith(('.jpg'))]

random_image = random.choice(images)
image_path = os.path.join(folder, random_image)

photo = Image.open(image_path)
photo_data = list(photo.getdata())
# print(photo.size)
# print(photo.mode)

# photo

# pytorch

transform = t.Compose([
    t.Grayscale(num_output_channels=1),
    t.Resize((100, 100)),
    t.ToTensor(),
    t.Normalize(mean=[0.5], std=[0.5])
])

transform_image = transform(photo)

# print(transform_image)

data = pd.read_csv('data/train_labels.csv')

result_wonder = (data['label'] == 1).sum()
result_scared = (data['label'] == 2).sum()
result_disgust = (data['label'] == 3).sum()
result_happy = (data['label'] == 4).sum()
result_sad = (data['label'] == 5).sum()
result_angry = (data['label'] == 6).sum()
result_neutral = (data['label'] == 7).sum()

x = ['wonder', 'scared', 'disgust', 'happy', 'sad', 'angry', 'neutral']
y = [result_wonder, result_scared, result_disgust, result_happy, result_sad, result_angry, result_neutral]

plt.bar(x, y, label = 'Count')
plt.xlabel('Emotions')
plt.ylabel('quantity')
plt.title('Data')
plt.legend()
# plt.savefig("plot.jpg", dpi=1600, bbox_inches='tight')

# visualize(image_path, transform_image)
# preprocess_image(image_path)