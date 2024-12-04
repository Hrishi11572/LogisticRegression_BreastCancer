#Objective : Build a Binary Class Logistic Regression model 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

Bcancer = datasets.load_breast_cancer()
Bcancer_df = pd.DataFrame(Bcancer.data)

features = Bcancer.feature_names.copy()
Bcancer_df.columns = features
Bcancer_df['class'] = Bcancer.target
#print(Bcancer_df)

X = np.array(Bcancer_df.iloc[:,:-1])
y = np.array(Bcancer_df['class'])
X = np.insert(X, 0 , 1 , axis = 1)

#standard scaler 
scaler = StandardScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])

def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
#print(X.shape) = (569, 31) at this point ... xj0 = 1 and rest 30 features
def LogisticRegressor(X,y,num_class,max_iter, learn_rate):
    weights = np.random.randn(X.shape[1])
    lr = learn_rate 
    loss_history = []
    
    for i in range(max_iter):
        total_loss = 0 
        for j in range(X.shape[0]):
            output = sigmoid(np.dot(weights.T , X[j]))
            weights -= lr*(output - y[j])*X[j]
        # total loss calculation
        total_loss += -y[j] * np.log(output) - (1 - y[j]) * np.log(1 - output)
        loss_history.append(total_loss / X.shape[0])
        
    return weights , loss_history 
    
weights, loss_list = LogisticRegressor(X, y, 2 ,max_iter=1000, learn_rate=0.0198)
print(weights)

#### --------   V I S U A L I S A T I O N    O F    L O S S    F U N C T I O N    &&    D B    -------- ####
# Plotting the loss curve
plt.figure(figsize=(8, 6))
plt.plot(loss_list)
plt.title('Log Loss (Binary Cross-Entropy) Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.show()