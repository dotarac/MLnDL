# Logistic Regression in the form of a Neural Network

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
	'Logistic Regression for binary classification'
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

	
class ShallowNeuralNet:
	'2 layer Neural Network for binary classification'
	def __init__(self,X_train,X_test,y_train,y_test,h_neurons=5,n_epochs=1000,alpha=0.03):
		self.X_train=X_train
		self.X_test=X_test
		self.y_test=y_test
		self.y_train=y_train
		self.n_epochs=n_epochs
		self.alpha=alpha
		self.h_neurons=h_neurons
		self.W1=np.random.randn(h_neurons,len(self.X_train))*0.01
		self.b1=np.zeros((h_neurons,1))
		self.W2=np.random.randn(1,h_neurons)*0.01
		self.b2=np.zeros((1,1))
		self.costs=[]

	def sigmoid(self,z):
		'''sigmoid activation Function'''
		return(1/(1+np.exp(-z)))

	def tanh(self,z):	
		'''tanh activation function'''
		return((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))
		
	def forward_propogate(self):
		'''Forward propogating the activations'''
		Z1=np.dot(self.W1,self.X_train)+self.b1
		A1=self.tanh(Z1)
		Z2=np.dot(self.W2,A1)+self.b2
		A2=self.sigmoid(Z2)
		return A1,A2	

	def backward_propogate(self,A1,A2):
		'''Back propogating the gradients'''
		m=self.X_train.shape[1]
		dZ2=A2-self.y_train
		dW2=np.dot(dZ2,A1.T)/m
		db2=np.sum(dZ2,axis=1,keepdims=True)/m
		dZ1=np.dot(self.W2.T,dZ2)*(1-np.power(A1,2))
		dW1=np.dot(dZ1,self.X_train.T)/m
		db1=np.sum(dZ1,axis=1,keepdims=True)/m
		J=np.squeeze(-np.sum(np.multiply(np.log(A2),self.y_train)+np.multiply(np.log(1-A2),1-self.y_train))/m)
		return dW1,db1,dW2,db2,J

	def gradient_descent_train(self):
		for i in range(self.n_epochs):
			A1,A2=self.forward_propogate()
			dW1,db1,dW2,db2,J=self.backward_propogate(A1,A2)
			self.W1-=dW1*self.alpha			
			self.b1-=db1*self.alpha
			self.W2-=dW2*self.alpha
			self.b2-=db2*self.alpha
			self.costs.append(J)	
		return self.W1,self.b1,self.W2,self.b2,self.costs
		
	def predict(self):
		'''predict new class labels'''
		Z1=np.dot(self.W1,self.X_test)+self.b1
		A1=self.tanh(Z1)
		Z2=np.dot(self.W2,A1)+self.b2
		A2=self.sigmoid(Z2).flatten()
		A=np.array([0 if i<0.5 else 1 for i in A2])
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



class DeepNeuralNet:
	'Neural Network with n hidden layers for binary classification'
	def __init__(self,X_train,X_test,y_train,y_test,h_layers,n_epochs=1000,alpha=0.03):
		self.X_train=X_train
		self.X_test=X_test
		self.y_test=y_test
		self.y_train=y_train
		self.n_epochs=n_epochs
		self.alpha=alpha
		self.h_layers=h_layers # Hidden layer is a dict with Layer number as key & number of neurons as val.
		self.hl=list(self.h_layers.keys())
		# intializing weights and biases of all layers
		self.W=[np.random.randn(self.h_layers[self.hl[0]],len(self.X_train))*0.01]
		self.b=[np.zeros((self.h_layers[self.hl[0]],1))]
		for i in range(len(self.hl)-1):
			self.W.append(np.random.randn(self.h_layers[self.hl[i+1]],self.h_layers[self.hl[i]])*0.01)		
			self.b.append(np.zeros((self.h_layers[self.hl[i+1]],1)))
		self.W.append(np.random.randn(1,self.h_layers[self.hl[-1]])*0.01)
		self.b.append(np.zeros((1,1)))
		self.costs=[]

	def sigmoid(self,z):
		'''sigmoid activation Function'''
		return(1/(1+np.exp(-z)))

	def tanh(self,z):	
		'''tanh activation function'''
		return((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))
	
	def forward_propogate(self):
		activations=[]
		Z=np.dot(self.W[0],self.X_train)+self.b[0]
		A=self.tanh(Z)
		activations.append(A)
		for i in range(len(self.hl)-1):
			Z=np.dot(self.W[i+1],A)+self.b[i+1]
			A=self.tanh(Z)
			activations.append(A)
		Z=np.dot(self.W[-1],self.X_train)+self.b[-1]
		A=self.sigmoid(Z)
		activations.append(A)
		print(activations)
		return activations
