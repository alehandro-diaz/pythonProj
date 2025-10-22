from PIL import Image
import random
import os
import torch
import torchvision.transforms as t

# Photo
rand_folder = random.randint(1, 7)
folder = f'data/DATASET/train/{rand_folder}'

images = [f for f in os.listdir(folder)
          if f.lower().endswith(('.jpg'))]

random_image = random.choice(images)
image_path = os.path.join(folder, random_image)

photo = Image.open(image_path)
photo_data = list(photo.getdata())
print(photo.size)
print(photo.mode)
# print(photo_data)

# photo

# pytorch
transform = t.Compose([
    t.Grayscale(num_output_channels=1),
    t.Resize((128, 128)),
    t.ToTensor(),
    t.Normalize(mean=[0.5], std=[0.5])
])

transform_image = transform(photo)
print(transform_image)