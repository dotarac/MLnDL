import pandas as pd
from sklearn.model_selection import train_test_split
from models import LogisticRegression

dataset='../data/iris.csv'
df=pd.read_csv(dataset)
y=df['label'].values
df.drop('label', axis=1, inplace=True)
X=df.values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)
X_train,X_test,y_train,y_test=X_train.T,X_test.T,y_train.reshape(len(y_train),1).T,y_test.reshape(len(y_test),1).T



Lor=LogisticRegression(X_train,X_test,y_train,y_test)
W,b,costs=Lor.gradient_descent_train()
Lor.plot_costs()
y_pred=Lor.predict()
Lor.accuracy_metrics(y_pred)
				