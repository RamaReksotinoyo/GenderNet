import matplotlib.pyplot as plt
from typing import List


def plotting(train_accuracy: List,
             test_accuracy: List,
             train_cost: List,
             test_cost: List) -> None:
    """
    Plotting the accuracy and cost of the training and testing
    """

    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(train_cost, 'r-', label='Train')
    plt.plot(test_cost, 'b-', label='Test')
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.show()

    plt.subplot(122)
    plt.plot(train_accuracy, 'r-', label='Train')
    plt.plot(test_accuracy, 'b-', label='Test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()

