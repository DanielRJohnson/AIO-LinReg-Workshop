#public libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from costs import rmse

def main():
    #create data
    data_X = np.fromiter( ( x for x in range(0,100) ), int).T[:, np.newaxis]
    data_y = np.fromiter([ y**2 for y in range(0,100) ], int)

    #split into train test
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.5, random_state=42)

    #make the model
    reg = LinearRegression().fit(X_train, y_train)
    prediction = reg.predict(X_test)
    for x, y, real in zip(X_test, prediction, y_test):
        print(x, y, real)
    
    print("R^2 ON TRAINING SET: ", reg.score(X_test, y_test))
    draw(X_test, y_test, prediction)

def draw(X, y, hx):
    fig, ax = plt.subplots()
    fig.suptitle('Linear Regression') #supertitle
    fig.tight_layout(pad=2.5, w_pad=1.5, h_pad=0) #fix margins
    ax.scatter(X[:], y)
    ax.plot(X[:], hx)
    ax.set(xlabel='X', ylabel='Y')
    ax.set_title('Prediction')
    ax.set_aspect(max(X[:])/max(y))
    plt.show()

if __name__ == "__main__": main()