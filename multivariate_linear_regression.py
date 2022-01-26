# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
# step -1
def initialize_parameters(lenw):
    w = np.random.randn(1,lenw)
    b = 0
    return w,b


# %%
# step -2
def forward_prop(X, w, b):
    z = np.dot(w,X) + b
    return z


# %%
# step -3
def cost_function(z, y):
    m = y.shape[1]
    J = (1/(2*m))*np.sum(np.square(z-y))
    return J


# %%
# step -4
def back_prop(X, y, z):
    m = y.shape[1]
    dz = (1/m)*(z-y)
    dw = np.dot(dz, X.T)
    db = np.sum(dz)
    return dw, db


# %%
# step -5
def gradient_descent_update(w,b,dw,db,learninig_rate):
    w = w - learninig_rate*dw
    b = b - learninig_rate*db
    return w,b


# %%
# step -6
def linear_regression_model(x_train, y_train, x_val, y_val, learning_rate, epochs):
    lenw = x_train.shape[0]
    w,b = initialize_parameters(lenw)

    costs_train = []
    m_train = y_train.shape[1]
    m_val = y_val.shape[1]

    for i in range(1,epochs+1):
        z_train = forward_prop(x_train, w, b)
        cost_train = cost_function(z_train, y_train)
        dw, db = back_prop(x_train, y_train, z_train)
        w,b = gradient_descent_update(w,b,dw, db, learning_rate)

        # store learning cost in a list for plotting purpose
        if i%10 == 0:
            costs_train.append(cost_train)

        # MAE_train
        MAE_train = (1/m_train)*np.sum(np.abs(z_train, y_train))

        # cost_val, MAE_val
        z_val = forward_prop(x_val, w, b)
        cost_val = cost_function(z_val, y_val)
        MAE_val = (1/m_val)*np.sum(np.abs(z_val, y_val))

        #print out cost_train, cost_val, MAE_train, MAE_val
        print("Epochs ", i, "/", epochs, " :")
        print("Training cost = ", cost_train, " | validation cost = ", cost_val)
        print("Training MAE = ", MAE_train, " | Valdation MAE = ", MAE_val)

    plt.plot(costs_train)
    plt.show()

# %% [markdown]
# # Testing on Boston Dataset

# %%
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


# %%
boston = load_boston()


# %%
boston['feature_names']


# %%
boston['data'].shape


# %%
bost = pd.DataFrame(boston['data'], columns=boston['feature_names'])


# %%
bost.head()


# %%
boston['target']


# %%
X = (bost-bost.mean())/(bost.max() - bost.min())


# %%
X.describe()


# %%
X.head()


# %%
y = boston['target']


# %%
y.shape


# %%
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=5)


# %%
x_train.shape


# %%
x_train = x_train.T
x_train.shape


# %%
y_train = np.array([y_train])


# %%
y_train.shape


# %%
x_val = x_val.T
y_val = np.array([y_val])


# %%
x_val.shape, y_val.shape


# %%
linear_regression_model(x_train, y_train, x_val, y_val, 0.4, 500)

# %% [markdown]
# # Compare our model with sklearn linear regression

# %%
from sklearn import linear_model


# %%
linear_regression = linear_model.LinearRegression()
model = linear_regression.fit(x_train.T, y_train.T)
predictions = linear_regression.predict(x_val.T)


# %%
predictions.shape


# %%
MAE_val_with_sklearn = (1/y_val.shape[1])*np.sum(np.abs(predictions-y_val.T))


# %%
MAE_val_with_sklearn


# %%


