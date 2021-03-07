'''
# Name: Daniel Johnson
# File: main.py
# Date: 1/5/2021
# Brief: This script trains a linreg model and plots its predictions
'''

#public libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#my files
from linreg import LinRegModel

def main():
    #create data
    data_X = np.fromiter( ( x for x in range(0,100) ), int).T[:, np.newaxis]
    data_y = np.fromiter([ y**2 for y in range(0,100) ], int)

    #split into train test
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.5)#, random_state=42)

    #make the model
    lr = LinRegModel(X_train, y_train, alpha = 3e-2)

    #train it
    J_Hist, theta_hist = lr.train(maxIters = int(1e3), convergenceThreshold = 1e-5)

    print("COST ON TRAINING SET: ", lr.cost(lr.y, lr.X @ lr.thetas))

    #draw it
    #draw(lr, X_test, y_test, J_Hist, theta_hist[-1], bool3D=False)

    #iterations over time
    for i in range(len(theta_hist)):
        if i % 10 == 0: draw(lr, X_test, y_test, J_Hist, theta_hist[i], bool3D=False)

'''
# @post: graphs are shown on the screen from the model, inputs, outputs, and cost history
'''
def draw(model, X: np.ndarray, y: np.ndarray, J_Hist: list, thetas: list, bool3D: bool = False):
    #set up subplots for the cost history and prediction graph
    fig = plt.figure()
    fig.suptitle('Linear Regression') #supertitle
    fig.tight_layout(pad=2.5, w_pad=1.5, h_pad=0) #fix margins

    costPlot = fig.add_subplot(121)
    drawCostHistory(J_Hist, costPlot)

    if bool3D:
        predPlot = fig.add_subplot(122, projection='3d')
        drawPrediction3D(model, X, y, predPlot)
    else:
        predPlot = fig.add_subplot(122)
        drawPrediction2D(model, X, y, thetas, predPlot)
    #show the cool graphs :)
    plt.show()

'''
# @post: cost history is plotted to the screen
'''
def drawCostHistory(J_Hist: list, plot) -> None:
    plot.plot(J_Hist)
    plot.set_ylabel('Cost')
    plot.set_xlabel('Iterations')
    plot.set_title('Cost vs. Iterations')
    plot.axis([0, len(J_Hist), 0, max(J_Hist)])
    plot.set_aspect(len(J_Hist)/max(J_Hist))

'''
# @post: predictions are plotted to the screen in 2D
'''
def drawPrediction2D(model, X: np.ndarray, y: np.ndarray, thetas: list, plot) -> None:
    #get inputs and predictions
    formatted_X = np.hstack( (np.ones( (model.m, 1)), model.scaleFeatures(X)) )
    hx = formatted_X @ thetas

    #plot them vs real y
    plot.scatter(X, y)
    plot.plot(X, hx)
    plot.set(xlabel='X', ylabel='Y')
    plot.set_title('Prediction')
    plot.set_aspect(max(X)/max(y))

'''
# @post: predictions are plotted to the screen in 3D
'''
def drawPrediction3D(model, X: np.ndarray, y: np.ndarray, plot) -> None:
    plot.scatter(X[:,0], X[:,1], y, s=0.5, c="blue")
    plot.scatter(X[:,0], X[:,1], model.predict(X), s=5, c="red")
    plot.set(xlabel='X', ylabel='Y', zlabel='Z')
    plot.set_title('Prediction\nBlue = Real, Red = Predicted')

if __name__ == "__main__": main()