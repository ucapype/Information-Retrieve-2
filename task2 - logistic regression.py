import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle
from tqdm import trange,tqdm
import random

def crossFeature(query_embedding, passage_embedding):
	'''
	Function: cross the feature as the input
	'''

	feature_first_1 = np.multiply(query_embedding,passage_embedding)
	feature_first_2 = query_embedding+passage_embedding
	feature_second = np.multiply(feature_first_1,feature_first_2)
	return feature_second #obtain the second order feature


def getBatchData(
	select_keys,
	query_ID_list,
	passage_ID_list,
	query_dict,
	passage_dict,
	LUT,
	batch_size,
	it):
	'''
	Obtian the batch data, with input shape of x:[batch_size, 32], y:[batch_size]
	input:
	select_keys : list(LUT.keys())
	query_ID_list: np.array(list(query_list.keys()))
	passage_ID_list: np.array(list(passage_list.keys()))
	query_dict
	passage_dict
	batch_size
	it
	'''

	y = []
	x = []


	for i in range(it,it+batch_size):
		groupID = select_keys[i] #accelerating training (diretly chossing)

		if groupID[0] in query_ID_list and groupID[1] in passage_ID_list:
			if LUT[groupID] == 1.0:
				y.append(1)
			else:
				y.append(-1)
			x.append(crossFeature(
			query_dict[groupID[0]],
			passage_dict[groupID[1]]))
		else:
			y.append(-1)
			x.append(crossFeature(
			query_dict[groupID[0]],
			passage_dict[groupID[1]]))

	return np.array(x), np.array(y)[:,np.newaxis]


def Sigmoid(x):
	return 1/(1 + np.exp(-x))


class LogisticRegression():
	def __init__(self, inc, learning_rate):
		super(LogisticRegression, self).__init__()

		#init weight and bias
		self.w = np.random.randn(inc,1)
		self.b = np.random.randn(1,1)

		#init gradient
		self.gradient = 0

		#init loss value 
		self.loss = 0

		#init the learning rate
		self.learning_rate = learning_rate


	def loss_fn(self,x,y):
		linear = x.dot(self.w) + self.b #shape (128, 1)
		self.loss = -np.sum(np.log(Sigmoid(np.multiply(y,linear))))
		return self.loss

	def backward(self,x,y):
		x_extend = np.hstack((x,np.ones((x.shape[0],1)))) #shape:(128,33)
		w_extend = np.vstack((self.w,self.b)) #shape:(33,1)
		TnXn = np.multiply(y,x_extend)
		linear = x_extend.dot(w_extend)
		left = 1 - Sigmoid(np.multiply(y,linear))
		self.gradient = -np.sum(np.multiply(left,TnXn),axis=0)
	
	def step(self):
		gradient_w = self.gradient[:-1][:,np.newaxis]
		gradient_b = self.gradient[-1]
		self.w = self.w - self.learning_rate*gradient_w
		self.b = self.b - self.learning_rate*gradient_b

	def save(self):
		with open("LogisticModel.pth","wb") as f: pickle.dump((self.w, self.b),f)



def NDCG(predict_list, real_list):
	'''
	The NDCG algorithm
	predict_list: shape of [score] in order
	real_list: shape of [score] in order
	'''
	
	#cut them into 200 piece as the qid is not the same
	predict_list = np.array(predict_list)
	real_list = np.array(real_list)
	predict_list = np.split(predict_list,200)
	real_list = np.split(real_list,200)

	NDCG_index = []

	for p,r in zip(predict_list,real_list):

		#calculating DCG for each section
		DCG_predict = sum([p[i]/np.log2(i+2) for i in range(len(p))]) + 1 #avoid zero

		#calculating the ideal DCG
		r = sorted(r,reverse=True)
		IDCG = sum([r[i]/np.log2(i+2) for i in range(len(r))]) + 1 #avoid zero

		NDCG_index.append(DCG_predict/IDCG)

	return np.mean(NDCG_index)


if __name__ == '__main__':
	#1. read all embedding and other information in training set and the validation set
	with open("Sentense2VecTrainingSet.data","rb") as f: training_query,training_passage,TrainingLUT = pickle.load(f)
	with open("Sentense2VecValidation.data","rb") as f: validation_query,validation_passage,ValidationLUT = pickle.load(f)
	
	#2. type transform
	training_query = dict(training_query)
	training_passage = dict(training_passage)
	validation_query = dict(validation_query)
	validation_passage = dict(validation_passage)


	#3.trainingset packing (accelerating training)
	select_keys = list(TrainingLUT.keys())
	query_ID_list = np.array(list(training_query.keys()))
	passage_ID_list = np.array(list(training_passage.keys()))


	#4.setting the super parameters
	batch_size = 128
	learning_rate = 1e-3
	Num_Epoch = 1


	#5.create logstic regressor and iteration
	regressor = LogisticRegression(inc = 32, learning_rate = learning_rate)
	for it in trange(10000):
		x,y = getBatchData(select_keys,query_ID_list,passage_ID_list,training_query,training_passage,TrainingLUT,batch_size,it)
		loss = regressor.loss_fn(x,y)
		regressor.backward(x,y)
		regressor.step()
		regressor.save()

		if it%100 == 0:
			regressor.save()
			print("\n\n",it,regressor.loss,"\n\n")

