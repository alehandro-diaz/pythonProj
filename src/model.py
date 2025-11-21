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
    
class SimpleCNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feauter_extractor = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classfier = nn.Sequential(
            nn.Flatten(),
            #nn.Linear(, 7)
        )

    def forward(self, x):
        x = self.feauter_extractor(x)
        x = self.classfier(x)
        return x


#инициализируем размер изображения и саму модель
img_size = (100, 100)
model = SimpleFcModel(7, img_size).to(device)

 #добавляем оптимизаторы
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#переменные для кол-ва эпох и оценки
n_epoch = 50
best_f1 = 0.0

#история для создания графика
history = {
    "train_loss": [], "test_loss": [],
    "train_f1": [], "test_f1": [],
    "train_acc": [], "test_acc": []
}

#сам цикд обучения и цикл оценки
for epoch in range(n_epoch):
    print(f"\nEPOCH {epoch+1}/{n_epoch}")
    train_loss, train_acc, train_f1 = epo.train_epoch(model, dt.train_loader, optimizer, criterion, device)
    print(f"Train loss: {train_loss}")
    print(f"Train accuracy: {train_acc}")
    print(f"Train f1: {train_f1}")
    test_loss, test_acc, test_f1 = epo.evaluate_epoch(model, dt.test_loader, criterion, device)
    print(f"Test loss: {test_loss}")
    print(f"Test accyracy: {test_acc}")
    print(f"Test f1: {test_f1}")

    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_f1"].append(train_f1)
    history["test_f1"].append(test_f1)
    history["train_acc"].append(train_acc)
    history["test_acc"].append(test_acc)

    if test_f1 > best_f1:
        best_f1 = test_f1
        best_epoch = epoch
        torch.save(model.state_dict(), "models/best_model_fc.pth")
        print(f"Epoch: {epoch+1}, F1: {best_f1:.3f}")

print(f"Best model: Epoch {best_epoch+1}, F1: {best_f1:.3f}")

#вывод оценки test_f1 и train_f1
# print(f"Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f}")
# if test_f1 > best_f1:
#     best_f1 = test_f1
#     torch.save(model.state_dict(), "models/best_model_fc.pth")

#сохранение графика
plt.plot_history(history)