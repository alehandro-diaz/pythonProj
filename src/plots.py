import matplotlib.pyplot as plt

def plot_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    # print("Проверка данных:")
    # print(f"Train loss: {history['train_loss']}")
    # print(f"Test loss: {history['test_loss']}")
    # print(f"Train acc: {history['train_acc']}")
    # print(f"Test acc: {history['test_acc']}")
    # print(f"Train f1: {history['train_f1']}")
    # print(f"Test f1: {history['test_f1']}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
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
    
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/plot5.jpg')