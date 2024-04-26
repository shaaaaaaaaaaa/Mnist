import matplotlib.pyplot as plt
import os
import mplcursors  # 导入 mplcursors 库

# read data from txt
def read_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [float(line.strip()) for line in lines]
        return data

# plot curve with tooltips
def plot_curve(data, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(data, label=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    # 启用 tooltip
    mplcursors.cursor(hover=True, highlight=False).connect("add", lambda sel: sel.annotation.set_text(sel.annotation.get_text()))

    plt.show()

def plots():
    record_path = os.path.join(os.path.dirname(__file__), 'records')
    train_loss_path = os.path.join(record_path, 'result_of_train_loss.txt')
    train_accuracy_path = os.path.join(record_path, 'result_of_train_acc.txt')
    test_loss_path = os.path.join(record_path, 'result_of_test_loss.txt')
    test_accuracy_path = os.path.join(record_path, 'result_of_test_acc.txt')

    train_loss = read_data(train_loss_path)
    train_accuracy = read_data(train_accuracy_path)
    test_loss = read_data(test_loss_path)
    test_accuracy = read_data(test_accuracy_path)

    # 设置图形大小
    plt.figure(figsize=(10, 6))

    # 画出训练损失曲线
    plot_curve(train_loss, 'Train Loss Curve', 'Epoch', 'Loss')

    # 画出训练准确率曲线
    plot_curve(train_accuracy, 'Train Accuracy Curve', 'Epoch', 'Accuracy')

    # 画出测试损失曲线
    plot_curve(test_loss, 'Test Loss Curve', 'Epoch', 'Loss')

    # 画出测试准确率曲线
    plot_curve(test_accuracy, 'Test Accuracy Curve', 'Epoch', 'Accuracy')


if __name__ == "__main__":
    plots()
