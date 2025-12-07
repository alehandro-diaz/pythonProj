import torch.nn as nn
import torch.optim as optim
import torch as torch
import epochs as epo
import Data as dt
import plots as plt

#определение устройства pytorcg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#создание простой модели
class SimpleFcModel(nn.Module):
    #инитиализация
    def __init__(self, num_classes, img_size):
        super().__init__()

        #присвоение размера картинок
        H, W = img_size
        input_dim = 1 * H * W
        self.flatten = nn.Flatten() #преобразование в вектор
        self.fc1 = nn.Linear(input_dim, 512) # линейный слой. На вход принимает набор чисел с размерностью 1* H * W. На выход отдает 512 параметров
        self.relu = nn.ReLU() #слой активатор. Все отрицательные числа приравнивает к 0, а положительные остаються положительными
        self.fc2 = nn.Linear(512, num_classes) #тоже самое. Только на вход принимает 512 парметров, а на выходе даёт num_classes(На выходе будут 7)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

#Создание свёрточной модели нейросети    
class SimpleCNNModel(nn.Module):
    # инициализация
    def __init__(self, img_size, num_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),#Делает карту принаков. Размер тензора (32, 100, 100)
            nn.BatchNorm2d(32),
            nn.ReLU(),#Слой активатор, все отрицательные веса становяться нулями
            nn.MaxPool2d(2, 2), #меняет размер тензора, размер становиться (16, 50, 50)
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 5, 1, 2), # здесь размер тензора становиться (64, 50, 50)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # а тут (32, 25, 25)
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),

            nn.Conv2d(2048, 4096, 3, 1, 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(),

            nn.Conv2d(4096, 8192, 3, 1, 1),
            nn.BatchNorm2d(8192),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        # H = img_size[0] // 8
        # W = img_size[1] // 8
        # dim = 4096 * H * W 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(8192, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


#инициализируем размер изображения и саму модель
img_size = (100, 100)
#model = SimpleFcModel(7, img_size).to(device)
model = SimpleCNNModel(img_size, 7).to(device)

 #добавляем оптимизаторы
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

#переменные для кол-ва эпох и оценки
n_epoch = 50
best_f1 = 0.0

#история для создания графика
history = {
    "train_loss": [], "test_loss": [],
    "train_f1": [], "test_f1": [],
    "train_acc": [], "test_acc": [],
    "test_f1_per_class": []
}

#сам цикд обучения и цикл оценки
for epoch in range(n_epoch):
    print(f"\nEPOCH {epoch+1}/{n_epoch}")
    train_loss, train_acc, train_f1, train_f1_per_class = epo.train_epoch(model, dt.train_loader, optimizer, criterion, device)
    
    test_loss, test_acc, test_f1, test_f1_per_class = epo.evaluate_epoch(model, dt.test_loader, criterion, device)
    
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_f1"].append(train_f1)
    history["test_f1"].append(test_f1)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)
    history["test_f1_per_class"].append(test_f1_per_class)

    if test_f1 > best_f1:
        best_f1 = test_f1
        best_epoch = epoch
        torch.save(model.state_dict(), "models/best_model_dropout4.pth")
        print(f"Epoch: {epoch+1}, F1: {best_f1:.3f}")

print(f"Best model: Epoch {best_epoch+1}, F1: {best_f1:.3f}")

#сохранение графика
plt.plot_history(history)