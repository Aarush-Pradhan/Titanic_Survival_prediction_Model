import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")

    tick= np.arange(2)
    plt.xticks(tick,['Not Survived','Survived'])
    plt.yticks(tick,['Not Survived','Survived'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout
    plt.show()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()
    


def plot_survival_by_gender(df):
    plt.figure()
    sns.countplot(x="Sex", hue="Survived", data=df)
    plt.title("Survival by Gender")
    plt.show()
    plt.savefig("outputs/survival_by_gender.png")
    plt.close()