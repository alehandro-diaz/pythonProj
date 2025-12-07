import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, class_names=None):
    epochs = range(1, len(history['train_loss']) + 1)

    # print("Проверка данных:")
    # print(f"Train loss: {history['train_loss']}")
    # print(f"Test loss: {history['test_loss']}")
    # print(f"Train acc: {history['train_acc']}")
    # print(f"Test acc: {history['test_acc']}")
    # print(f"Train f1: {history['train_f1']}")
    # print(f"Test f1: {history['test_f1']}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    ax1, ax2, ax3, ax4 = axes
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['test_loss'], label='Test Loss')
    ax1.legend(); ax1.set_title('Loss')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
    
    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['test_acc'], label='Test Acc')
    ax2.legend(); ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy')
    
    ax3.plot(epochs, history['train_f1'], label='Train F1')
    ax3.plot(epochs, history['test_f1'], label='Test F1')
    ax3.legend(); ax3.set_title('F1 Score')
    ax3.set_xlabel('Epochs'); ax3.set_ylabel('F1 Score')

    if 'test_f1_per_class' in history and len(history['test_f1_per_class']) > 0:
        # Берем F1-scores из последней эпохи
        last_epoch_f1 = history['test_f1_per_class'][-1]
        
        # Создаем имена классов если не предоставлены
        if class_names is None:
            num_classes = len(last_epoch_f1)
            class_names = [f'Class {i}' for i in range(num_classes)]
        elif len(class_names) != len(last_epoch_f1):
            class_names = [f'Class {i}' for i in range(len(last_epoch_f1))]
        
        # Создаем гистограмму
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(last_epoch_f1)))
        bars = ax4.bar(range(len(last_epoch_f1)), last_epoch_f1, color=colors, edgecolor='black')
        
        # Добавляем значения
        for bar, value in zip(bars, last_epoch_f1):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax4.set_xlabel('Class')
        ax4.set_ylabel('F1 Score')
        ax4.set_xticks(range(len(class_names)))
        ax4.set_xticklabels(class_names, rotation=45, ha='right')
        ax4.set_ylim([0, 1.1])
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No per-class F1 data', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('F1 Score per Class')
    
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/plot15.jpg')