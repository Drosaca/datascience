from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


def train(x_train, x_label, y_test, y_label, classifiers):
    scores = []
    for cls in classifiers:
        beginning = datetime.now()
        print(cls, 'classifier:')
        print('\tTraining...')
        classifiers[cls].fit(x_train, x_label)
        print('\tTime to train: ', timedelta(seconds=(datetime.now() - beginning).seconds))

        beginning = datetime.now()
        print('\tTesting... (score)')
        score = classifiers[cls].score(y_test, y_label)
        scores.append(score)
        print('\tTime to test: ', timedelta(seconds=(datetime.now() - beginning).seconds))
        print('\tScore: ', score)

        graph = plot_confusion_matrix(classifiers[cls], y_test, y_label, cmap=plt.cm.Blues, normalize='true')
        graph.ax_.set_title(cls)
        plt.show()

    if len(classifiers) > 1:
        plt.bar(classifiers.keys(), scores, width=0.3, data=scores)
        plt.show()
