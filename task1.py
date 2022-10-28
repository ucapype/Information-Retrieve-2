import numpy as np
import pandas as pd
import re
import pickle
from tqdm import trange,tqdm
import copy
import matplotlib.pyplot as plt

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
		p = sorted(p,reverse=True)
		DCG_predict = sum([p[i]/np.log2(i+2) for i in range(len(p))]) + 1 #avoid zero

		#calculating the ideal DCG
		r = sorted(r,reverse=True)
		IDCG = sum([r[i]/np.log2(i+2) for i in range(len(r))]) + 1 #avoid zero

		NDCG_index.append(DCG_predict/IDCG)

	return np.mean(NDCG_index)


def mAP(predict_list, real_list):
	'''
	The mean precision of the prediction list and the real list
	predict_list: shape of [score] in order
	real_list: shape of [score] in order
	'''

	#first, transfer to the binary
	predict_list = np.array(predict_list)
	real_list = np.array(real_list)

	#then, draw the PR plot
	PR = []

	for threshold in np.linspace(0,1,101):
		predict_list_ = copy.deepcopy(predict_list)
		predict_list_[predict_list_ >= threshold] = 1
		predict_list_[predict_list_ < threshold] = 0

		#then, calculating the index
		TP = len(predict_list_[np.logical_and(predict_list_ == 1, real_list == 1)])
		FN = len(predict_list_[np.logical_and(predict_list_ == 0, real_list == 1)])
		precision = TP/(len(predict_list_) + 1) #prevent zero value
		recall = TP/(TP + FN + 1) #prevent zero value
		PR.append([recall,precision])
	
	#draw the PR plot and obtain the mAP
	PR = np.array(PR).T
	plt.plot(PR[0],PR[1])
	plt.show()

	#calculating the mAP
	return PR[0].dot(PR[1])





if __name__ == '__main__':
	#1. read all text in training set
	corpus_array = pd.read_csv("train_data.tsv",sep='\t')[['qid', 'pid', 'relevancy']]

	#2. read the BM25-based test result
	BM25_predict_result = pd.read_csv("bm25.csv").values

	#3. normalized the BM25
	BM25_predict_result[:,-1] = (BM25_predict_result[:,-1] - BM25_predict_result[:,-1].min())/(BM25_predict_result[:,-1].max() - BM25_predict_result[:,-1].min())


	#4. generate the real_list
	real_list = []
	predict_list = []
	filling = np.mean(corpus_array.values.T[-1])

	for qid,pid,score in tqdm(BM25_predict_result):
		result = corpus_array.loc[(corpus_array['qid'] == int(qid)) & (corpus_array['pid'] == int(pid)), :].values.tolist()
		
		if result:
			#if find the result, it should use the real value
			real_list.append(result.values[-1])
		else:
			#Using the mean value to fill it
			real_list.append(filling)

	predict_list = BM25_predict_result[:,-1].tolist()

	#Using the metric to measure the performance
	avergae_precision = mAP(predict_list,real_list)
	NDCG_index = NDCG(predict_list,real_list)
	print(NDCG_index)

