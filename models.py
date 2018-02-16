# Logistic Regression in the form of a Neural Network

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
	'Perform Logistic Regression for binary classification'
	def __init__(self,X_train,X_test,y_train,y_test,n_epochs=1000,alpha=0.03):
		
		self.X_train=X_train
		self.X_test=X_test
		self.y_test=y_test
		self.y_train=y_train
		self.n_epochs=n_epochs
		self.alpha=alpha
		self.W=np.zeros((len(self.X_train),1))
		self.b=0
		self.costs=[]

	def sigmoid(self,z):
		'''sigmoid activation Function'''
		return(1/(1+np.exp(-z)))

	def tanh(self,z):	
		'''tanh activation function'''
		return((np.exp(z)-np.exp(-z))/float(np.exp(z)+np.exp(-z)))

	def forward_propogate(self):
		'''Forward Propogation step'''
		Z=np.dot(self.W.T,self.X_train)+self.b
		A=self.sigmoid(Z)
		return A

	def backward_propogate(self,A):
		'''Backward propogation step'''
		m=self.X_train.shape[1]
		J=np.squeeze(-np.sum(np.dot(self.y_train,np.log(A).T)+np.dot(1-self.y_train,np.log(1-A).T))/m)
		dW=np.dot(self.X_train,(A-self.y_train).T)/float(m)
		db=np.sum(A-self.y_train)/float(m)
		return J,dW,db

	def gradient_descent_train(self):
		'''Gradient Descent algorithm for updating weights'''
		for i in range(self.n_epochs):
			A=self.forward_propogate()
			J,dW,db=self.backward_propogate(A)
			self.W-=self.alpha*dW
			self.b-=self.alpha*db
			self.costs.append(J)		
		return self.W,self.b,self.costs

	def predict(self):
		'''predict new class labels'''
		Z=np.dot(self.W.T,self.X_test)+self.b
		A=self.sigmoid(Z).flatten()
		A=np.array([0 if i<0.5 else 1 for i in A])
		return A

	def accuracy_metrics(self,y_pred):
		'''performance measurements of algorithm'''
		stats={}
		diff=list(y_pred-self.y_test.flatten())
		sums=list(y_pred+self.y_test.flatten())
		stats["test_accuracy"]=diff.count(0)/float(len(diff))
		stats["miss_rate"]=1-stats["test_accuracy"]
		stats["confusion_matrix"]=[[sums.count(0),diff.count(-1)],[diff.count(1),sums.count(2)]]
		print(stats) 

	def plot_costs(self):
		'''plot cost-function curve'''
		if self.costs==[]:
			print("Run Predict first to avail costs\n")
		else:
			plt.plot(range(self.n_epochs),self.costs)
			plt.show()	





			





