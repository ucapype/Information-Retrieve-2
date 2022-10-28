import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle
from tqdm import trange,tqdm
import random
import xgboost as xgb

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


def model_train(step,xgb_train,xgb_test,eval_metric,y):
	trainingparams = {
		"objective":"rank:ndcg",
		"eta":abs(step[0]), # the learning rate (can be changed)
		"eval_metric":eval_metric, # must set to this
		"max_depth": abs(int(round(step[1]))), # Maximum depth of a tree (can be changed)
	}
	num_round = 500 #number of iterations (500 is enough)

	watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
	model = xgb.train(trainingparams, xgb_train, num_round, watchlist)
	y_predict = model.predict(xgb_train)
	NDCG_Train = NDCG(y_predict,y)
	MSE = (NDCG_Train - 1)**2 #Using MSELoss to help to near 1
	return MSE,NDCG_Train


def model_test(step,xgb_train,xgb_test,eval_metric):
	trainingparams = {
		"objective":"rank:ndcg",
		"eta":abs(step[0]), # the learning rate (can be changed)
		"eval_metric":eval_metric, # must set to this
		"max_depth": abs(int(round(step[1]))), # Maximum depth of a tree (can be changed)
	}
	num_round = 500 #number of iterations (500 is enough)

	watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
	model = xgb.train(trainingparams, xgb_train, num_round, watchlist)
	y_predict = model.predict(xgb_test)
	return y_predict


def train():
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

	x,y = getAllData(select_keys,query_ID_list,passage_ID_list,training_query,training_passage,TrainingLUT)
	xgb_train = xgb.DMatrix(x, y)


	#4. loading the test set
	with open("Sentense2VecTestSet.data","rb") as f: test_query,test_passage,testLUT = pickle.load(f)	
	test_query = dict(test_query)
	test_passage = dict(test_passage)

	#5. construct the standard input of the test set
	select_keys_test = list(testLUT.keys())
	query_ID_list_test = np.array(list(test_query.keys()))
	passage_ID_list_test = np.array(list(test_passage.keys()))	
	test_x,test_y = getAllData(select_keys_test,query_ID_list_test,passage_ID_list_test,test_query,test_passage,testLUT)
	xgb_test = xgb.DMatrix(test_x,test_y)

	Finetune_List = []

	for eval_metric in ["ndcg-",'ndcg','map','map-']:
		#Using gradient descent algorithm
		step_1 = np.array((1e-4,6))
		step_2 = np.array((2*1e-4,7)) 
		learning_rate = 1e-6 #rate for adjusting parameter

		#4. training for step.1-2
		for step in [step_1,step_2]:
			MSE,NDCG_Train = model_train(step,xgb_train,xgb_test,eval_metric,y)
			Finetune_List.append([MSE,NDCG_Train,step[0],step[1],eval_metric])


		for it in range(15):
			#5. calculating the gradient
			gradient = (Finetune_List[-1][0] - Finetune_List[-2][0])*np.array([0.01, 0.99])/(step_2 - step_1)

			#6. update the parameters
			step_1 = np.array((Finetune_List[-1][2],Finetune_List[-1][3]))
			step_2 = step_1 - learning_rate * gradient

			MSE,NDCG_Train = model_train(step_2,xgb_train,xgb_test,eval_metric,y)
			Finetune_List.append([MSE,NDCG_Train,step_2[0],step_2[1],eval_metric])

			print(it, MSE, step_2)

	Finetune_List = sorted(Finetune_List,key=lambda x:x[0]) #from MSE small to large
	print("Best Model", Finetune_List[0])
	return Finetune_List[0]



def test(eta, node, method):
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

	x,y = getAllData(select_keys,query_ID_list,passage_ID_list,training_query,training_passage,TrainingLUT)
	xgb_train = xgb.DMatrix(x, y)


	#4. loading the test set
	with open("Sentense2VecTestSet.data","rb") as f: test_query,test_passage,testLUT = pickle.load(f)	
	test_query = dict(test_query)
	test_passage = dict(test_passage)

	#5. construct the standard input of the test set
	select_keys_test = list(testLUT.keys())
	query_ID_list_test = np.array(list(test_query.keys()))
	passage_ID_list_test = np.array(list(test_passage.keys()))
	test_x,test_y = getAllData(select_keys_test,query_ID_list_test,passage_ID_list_test,test_query,test_passage,testLUT,size=189876) #189877-1
	xgb_test = xgb.DMatrix(test_x,test_y)

	#train and test the model
	y_predict = model_test((eta, node),xgb_train, xgb_test, method)
	
	#output the model (for arranging)
	#<qid1 A1 pid1 rank1 score1 algoname2>
	output_list = []
	for index,score in enumerate(y_predict):
		output_list.append([select_keys_test[index][0],'A1',select_keys_test[index][1],score,'LM'])

	output_list = sorted(output_list,key = lambda x: (x[0],x[3]),reverse=True)
	
	#re-arrange it to obtain the rank
	queryID = output_list[0][0]
	rank = 1
	with open("LM.txt","a+") as f:
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
	_, _, eta, node, method = train() #0.0012218448943188385,10.493757129927856,'ndcg-'
	test(eta, node, method)	
