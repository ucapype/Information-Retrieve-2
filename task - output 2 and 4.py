import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle
from tqdm import trange,tqdm
import random
import torch
import torch.nn as nn

class CSPLayer(nn.Module):
	'''
	The CSP Bottle Neck-based MLP layer
	'''
	def __init__(self, channel, shunkrate = 4):
		super(CSPLayer, self).__init__()

		self.bottleneck = nn.Sequential(
			nn.Linear(channel,channel//shunkrate),
			nn.SiLU(),
			nn.Linear(channel//shunkrate,channel),
			)

	def forward(self,x):
		return x + self.bottleneck(x)

class AttentionLayer(nn.Module):
	'''
	The Attention layer based on SENet
	'''
	def __init__(self, channel, shunkrate=4):
		super(AttentionLayer, self).__init__()

		self.key = torch.LongTensor([0]).cuda()
		self.S = nn.Embedding(1,channel) #generate the weight
		self.E = nn.Sequential(
				nn.Linear(channel, channel//shunkrate,bias=False),
				nn.SiLU(),
				nn.Linear(channel//shunkrate, channel,bias=False),
				nn.Sigmoid(),
			)

	def forward(self,x):
		return x * self.E(self.S(self.key)).expand_as(x)

class MLPEncoderLayer(nn.Module):
	'''
	The 1d encoder layer based on MLP
	'''
	def __init__(self, channel, shunkrate=4):
		super(MLPEncoderLayer, self).__init__()

		self.attention = AttentionLayer(channel,shunkrate)
		self.layerNorm1 = nn.LayerNorm(channel)
		self.layerNorm2 = nn.LayerNorm(channel)
		self.feedforward = CSPLayer(channel,shunkrate)

	def forward(self,x):
		x = self.attention(x) + x
		x = self.layerNorm1(x)
		x = self.feedforward(x) + x
		x = self.layerNorm2(x)
		return x


class MLP(nn.Module):
	'''
	Using MLP to perform the problem (because there is no sequence!)
	input: query embedding and passage embedding
	'''
	def __init__(self, embedding_dim):
		super(MLP, self).__init__()

		#construct the MLP
		self.encoder = nn.Sequential(
			nn.Linear(2*embedding_dim+1, 128),
			MLPEncoderLayer(128,2),
			nn.Linear(128, 256),
			MLPEncoderLayer(256,4),			
			nn.Linear(256, 256),
			MLPEncoderLayer(256,8),
			nn.Linear(256, 256),
			MLPEncoderLayer(256,8),
			nn.Linear(256, 256),
			MLPEncoderLayer(256,4),
			nn.Linear(256, 128),
			MLPEncoderLayer(128,2),
			nn.Linear(128, 1),
			nn.Sigmoid(),
			)

		self.embedding = nn.Embedding(1,1) #only 1 embeddings SEP


	def forward(self,query,passage,symbol):
		'''
		query: [batchsize, embedding_dim]
		passage: [batchsize, embedding_dim]
		symbol: [1] with denotes the SEP
		'''
		
		#initially, it should obtian the CLS label
		SEP = self.embedding(symbol).repeat((1000,1))
		
		#then, merge the embedding symbol with the query and passage
		x = torch.cat((query,SEP,passage),axis=1)
		x = self.encoder(x)
		return x


def crossFeature(query_embedding, passage_embedding):
	'''
	Function: cross the feature as the input
	'''

	feature_first_1 = np.multiply(query_embedding,passage_embedding)
	feature_first_2 = query_embedding+passage_embedding
	feature_second = np.multiply(feature_first_1,feature_first_2)
	return feature_second #obtain the second order feature

def getAllData(
	select_keys,
	query_ID_list,
	passage_ID_list,
	query_dict,
	passage_dict,
	LUT,
	size=10000):
	'''
	Obtian all data with x:[len(select_keys), 32], y:[len(select_keys)]
	input:
	select_keys : list(LUT.keys())
	query_ID_list: np.array(list(query_list.keys()))
	passage_ID_list: np.array(list(passage_list.keys()))
	query_dict
	passage_dict
	'''

	y = []
	x = []


	for i in trange(size):
		groupID = select_keys[i] #accelerating training (diretly chossing)

		if groupID[0] in query_ID_list and groupID[1] in passage_ID_list:
			y.append(LUT[groupID])
			x.append(crossFeature(
			query_dict[groupID[0]],
			passage_dict[groupID[1]]))
		else:
			y.append(0)
			x.append(crossFeature(
			query_dict[groupID[0]],
			passage_dict[groupID[1]]))

	return np.array(x), np.array(y)[:,np.newaxis]

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


def getInputAndTraget(queryID, query, passageset, queryLUT):
	'''
	Function: provide the network input based on given query
	queryID: int
	query: shape:[32,]
	passageset: a set of passage with the element of 
	'''

	#intially, find where it can equal to 1 (as the positive)
	positive = queryLUT[np.logical_and(queryLUT[:,0] == queryID, queryLUT[:,-1] == 1.0)]
	
	#Then, find the corresponding passage according to the passage ID
	passagedict = dict(passageset)

	#initialized the embedding (as a whole, because it needs shuffle)
	input_embedding = []

	#add positive part to the embedding
	for _,passageID,score in positive:
		input_embedding.append([query[:,np.newaxis],passagedict[int(passageID)][:,np.newaxis],score])
	

	#Then, add 1000-len(input_embedding) other items and obtain there score
	negative = queryLUT[[random.randint(0,len(passageset)-1) for i in range(1000-len(input_embedding))]]


	#add then (negative part) into the embedding list
	for _,passageID,score in negative:
		input_embedding.append([query[:,np.newaxis],passagedict[int(passageID)][:,np.newaxis],score])


	#shuffle
	random.shuffle(input_embedding)
	input_embedding = np.array(input_embedding,dtype=object) #force to transform to np.ndarray
	

	#obtain the input embedding and score
	input_query = np.concatenate(input_embedding[:,0],axis=1).T.astype(np.float32)
	input_passage = np.concatenate(input_embedding[:,1],axis=1).T.astype(np.float32)
	target = input_embedding[:,2].astype(np.float32)
	
	return torch.FloatTensor(input_query).cuda(), torch.FloatTensor(input_passage).cuda(), torch.FloatTensor(target).unsqueeze(-1).cuda()



def outputTask2():
	#1. loading the test set
	with open("Sentense2VecTestSet.data","rb") as f: test_query,test_passage,testLUT = pickle.load(f)	
	test_query = dict(test_query)
	test_passage = dict(test_passage)

	#2. construct the standard input of the test set
	select_keys_test = list(testLUT.keys())
	query_ID_list_test = np.array(list(test_query.keys()))
	passage_ID_list_test = np.array(list(test_passage.keys()))

	#3. load the weight and bias of the logistic regression
	regressor = LogisticRegression(32,1e-3)
	with open("LogisticModel.pth","rb") as f: w,b = pickle.load(f)
	regressor.w = w
	regressor.b = b

	#4. test the model
	test_x,test_y = getAllData(select_keys_test,query_ID_list_test,passage_ID_list_test,test_query,test_passage,testLUT,size=189876)
	linear = test_x.dot(regressor.w) + regressor.b
	y_predict = Sigmoid(linear)

	#output the model (for arranging)
	#<qid1 A1 pid1 rank1 score1 algoname2>
	output_list = []
	for index,score in enumerate(y_predict):
		output_list.append([select_keys_test[index][0],'A1',select_keys_test[index][1],score,'LR'])

	output_list = sorted(output_list,key = lambda x: (x[0],x[3]),reverse=True)
	
	#re-arrange it to obtain the rank
	queryID = output_list[0][0]
	rank = 1
	with open("LR.txt","a+") as f:
		for index, each in enumerate(output_list):
			new_queryID = each[0]
			
			if new_queryID != queryID:
				queryID = new_queryID
				rank = 1
				f.write(str(each[0]) + " " + str(each[1]) + " " + str(each[2]) + " rank" + str(rank) + " " + str(each[3]) + " " + str(each[4]) + "\n")
			else:
				f.write(str(each[0]) + " " + str(each[1]) + " " + str(each[2]) + " rank" + str(rank) + " " + str(each[3]) + " " + str(each[4]) + "\n")
				rank += 1


def outputTask4():

	#1. loading the test set
	with open("Sentense2VecTestSet.data","rb") as f: test_query,test_passage,testLUT = pickle.load(f)	

	#2. construct the standard input of the test set
	queryTLUT = np.hstack((np.array(list(testLUT.keys())),np.array(list(testLUT.values()))[:,np.newaxis]))
	select_keys_test = list(testLUT.keys())

	#3. load the neural network model
	model = MLP(32).cuda()
	model.load_state_dict(torch.load("task4.pth")) 
	symbol = torch.LongTensor([0]).cuda()

	#4. test the model
	pbar = trange(0,len(test_query))

	output_list = []

	for it in pbar:
		query,passage,score = getInputAndTraget(test_query[it][0],test_query[it][1],test_passage,queryTLUT)
		predict_score = model(query,passage,symbol)
		
		for each in predict_score:
			output_list.append([select_keys_test[it][0],'A1',select_keys_test[it][1],each.item(),'NN'])

	output_list = sorted(output_list,key = lambda x: (x[0],x[3]),reverse=True)
	
	#re-arrange it to obtain the rank
	queryID = output_list[0][0]
	rank = 1
	with open("NN.txt","a+") as f:
		for index, each in enumerate(output_list):
			new_queryID = each[0]
			
			if new_queryID != queryID:
				queryID = new_queryID
				rank = 1
				f.write(str(each[0]) + " " + str(each[1]) + " " + str(each[2]) + " rank" + str(rank) + " " + str(each[3]) + " " + str(each[4]) + "\n")
			else:
				f.write(str(each[0]) + " " + str(each[1]) + " " + str(each[2]) + " rank" + str(rank) + " " + str(each[3]) + " " + str(each[4]) + "\n")
				rank += 1

if __name__ == '__main__':
	outputTask2()
	outputTask4()