import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle
from tqdm import trange,tqdm
import torch
import torch.nn as nn
import random

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


class NDCGLoss(nn.Module):
	def __init__(self):
		super(NDCGLoss, self).__init__()


	def forward(self, predict_score, score):
		#initially, it should normalized both score and the predict score (add = 1)
		score = score/torch.sum(score)
		predict_score = predict_score/torch.sum(score)


		#then, calcualte the ideal DCG
		DCGI = []
		for index,result in enumerate(score[score > 0.0]):
			DCGI += [result/np.log2(index + 2)]


		#After that, calculate predict DCG and compute the loss
		loss = 0.0 #initialization
		real_index = [score[score > 0.0].shape[0]+1,1] #real index
		for index,result in enumerate(predict_score):

			if index not in torch.where(score>0.0)[0]:
				loss += (result/(real_index[0] + 1))**2 #Using the MSELoss and so normalized
				real_index[0] += 1
			else:
				loss += (result/(real_index[1] + 1) - DCGI[real_index[1] - 1])**2 #Using the MSELoss but this time, it has the value
				real_index[1] += 1

		return torch.mean(loss)
		

if __name__ == '__main__':

	#1. read all embedding and other information in training set and the validation set
	with open("Sentense2VecTrainingSet.data","rb") as f: training_query,training_passage,TrainingLUT = pickle.load(f)
	with open("Sentense2VecValidation.data","rb") as f: validation_query,validation_passage,ValidationLUT = pickle.load(f)


	#2. type transformation
	queryTLUT = np.hstack((np.array(list(TrainingLUT.keys())),np.array(list(TrainingLUT.values()))[:,np.newaxis]))
	queryVLUT = np.hstack((np.array(list(ValidationLUT.keys())),np.array(list(ValidationLUT.values()))[:,np.newaxis]))


	#3. construct the network model and the parameters
	embedding_dim = 32
	learning_rate = 1e-3
	NumEpoch = 20
	batch_size = 1000
	model = MLP(embedding_dim).cuda()
	loss_fn = NDCGLoss() #Using the customer loss function to weight add
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	symbol = torch.LongTensor([0]).cuda()


	#4. Training the model
	pbar = trange(0,len(training_query))

	for epoch in range(NumEpoch):
		for it in pbar:
			#select the input data
			print(training_query[it][0])
			print(training_query[it][1])
			print(training_passage)
			exit()
			query,passage,score = getInputAndTraget(training_query[it][0],training_query[it][1],training_passage,queryTLUT)
			#output the result
			predict_score = model(query,passage,symbol)
			loss = loss_fn(predict_score, score)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			pbar.set_postfix({'loss':str(loss.item())})

		torch.save(model.state_dict(),"task4.pth")