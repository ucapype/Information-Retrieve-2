from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import re
import pickle
from tqdm import trange,tqdm
from nltk.corpus import stopwords
import logging
import copy
#nltk.download('stopwords')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def Word2VecTraining(embedding_dim):
	'''
	This function is used for training word embedding
	includes:
	1. read all text in training set
	2. obtain the query and passage only
	3. obtian the query-and-passage where score = 1 and merge them
	4. concate query, passage, and only the relavant query-and-passage
	5. delete the stopwords and the punctuation
	6. training the new gensim word2vec model
	7. create new dict to store
	8. output the result
	'''
	#1. read all text in training set
	corpus_array = pd.read_csv("train_data.tsv",sep='\t')[['queries', 'passage', 'relevancy']].values


	#2. obtain the query and passage only
	corpus = list(set(corpus_array[:,0].tolist())) + list(set(corpus_array[:,1].tolist()))


	#3-4. obtian the query-and-passage where score = 1 and merge them
	corpus_array = corpus_array[corpus_array[:,2] == 1.0]
	corpus_array[:,0] += " " + corpus_array[:,1]
	corpus += corpus_array[:,0].tolist()



	#5. delete the stopwords and the punctuation
	stop_words = set(stopwords.words('english'))
	pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
	punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./，。、《》？；：‘“{【】}|、！@#￥%……&*（）——+=-"

	for index,page in enumerate(tqdm(corpus)):
		page = re.sub(r'[{}]+'.format(punctuation), '', page)
		page = page.strip().lower()
		corpus[index] = pattern.sub('', page).split()



	#6. training the new gensim word2vec model
	model = Word2Vec(corpus, sg=1, vector_size=embedding_dim,  window=8,  min_count=0,  negative=3, sample=0.001, hs=1, workers=8)


	#7. create new dict to store the word and the each word vector
	dict_output = {}
	words = model.wv.index_to_key
	for index,word in enumerate(tqdm(words)):
		dict_output[word] = model.wv.vectors[index]


	#8. save the dict
	with open("Word2Vec.dict","wb") as f: pickle.dump(dict_output,f)


def Sentense2VecTraining(embedding_dim):
	'''
	This function is to obtain the sentence embedding for training set and the validation set
	includes:
	1. read the word embedding dict
	2. read all information in training set and the validation set
	3. construct the look-up table for each query and passage (only preserve the ids and the scores)
	4. only reserve the unique query and passage
	5. delete the stopwords and the punctuation
	6. calculate the average word embedding for all sentence (query and passage)
	7. save the training set embedding and validation set embedding respectively
	'''

	#1. read the word embedding dict
	with open("Word2Vec.dict","rb") as f: word_embedding = pickle.load(f)

	#2. read all information in training set and the validation set
	training_set = pd.read_csv("train_data.tsv",sep='\t')
	validation_set = pd.read_csv("validation_data.tsv",sep='\t')

	#3. construct the look-up table
	TrainingLUT = {(int(qid),int(pid)):relevancy for qid,pid,relevancy in training_set[['qid','pid','relevancy']].values}
	ValidationLUT = {(int(qid),int(pid)):relevancy for qid,pid,relevancy in validation_set[['qid','pid','relevancy']].values}


	#4. only reserve the unique query and passage
	training_query = copy.deepcopy(training_set).drop_duplicates(subset='queries').values[:,[0,-3]].tolist()
	training_passage = training_set.drop_duplicates(subset='passage').values[:,[1,-2]].tolist()
	validation_query = copy.deepcopy(validation_set).drop_duplicates(subset='queries').values[:,[0,-3]].tolist()
	validation_passage = validation_set.drop_duplicates(subset='passage').values[:,[1,-2]].tolist()


	#5. delete the stopwords and the punctuation
	stop_words = set(stopwords.words('english'))
	pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
	punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./，。、《》？；：‘“{【】}|、！@#￥%……&*（）——+=-"

	for corpus in [training_query,training_passage,validation_query,validation_passage]:
		for index, (_,content) in enumerate(tqdm(corpus)):
			content = re.sub(r'[{}]+'.format(punctuation), '', content)
			content = content.strip().lower()
			corpus[index][-1] = pattern.sub('', content).split()


	#6. calculate the average word embedding
	for corpus in [training_query,training_passage,validation_query,validation_passage]:
		for index,(_,content) in enumerate(tqdm(corpus)):

			#initialized
			content_embedding = np.zeros((embedding_dim,))
			len_effective_content = 0

			#generating
			for word in content:
				if word in word_embedding.keys(): 
					content_embedding += word_embedding[word]
					len_effective_content += 1

			#calculate the sentense embedding
			if len_effective_content > 0:
				content_embedding /= len_effective_content

			#add to corpus
			corpus[index][-1] = content_embedding

	#7. save all the things
	with open("Sentense2VecTrainingSet.data","wb") as f: pickle.dump((training_query,training_passage,TrainingLUT),f)
	with open("Sentense2VecValidation.data","wb") as f: pickle.dump((validation_query,validation_passage,ValidationLUT),f)


def Testset2Vec(embedding_dim = 32):
	'''
	Using word2vec tor transform testset into vector
	includes:
	1. read the word embedding dict
	2. read all information in test set
	3. only reserve the unique query and passage
	4. delete the stopwords and the punctuation
	5. calculate the average word embedding for all sentence (query and passage)
	6. save the training set embedding and validation set embedding respectively	
	'''

	#1. read the word embedding dict
	with open("Word2Vec.dict","rb") as f: word_embedding = pickle.load(f)

	#2. read all information in training set and the validation set
	test_set = pd.read_csv("candidate_passages_top1000.tsv",sep='\t',header=None)


	#3. only reserve the unique query and passage
	test_set.columns = ['qid','pid','queries','passage'] #add columns to the test set
	test_query = copy.deepcopy(test_set).drop_duplicates(subset='queries').values[:,[0,-2]].tolist()
	test_passage = test_set.drop_duplicates(subset='passage').values[:,[1,-1]].tolist()

	#3. construct the look-up table
	TestLUT = {(int(qid),int(pid)):0 for qid,pid in test_set[['qid','pid']].values}

	#4. delete the stopwords and the punctuation
	stop_words = set(stopwords.words('english'))
	pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
	punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./，。、《》？；：‘“{【】}|、！@#￥%……&*（）——+=-"

	for corpus in [test_query,test_passage]:
		for index, (_,content) in enumerate(tqdm(corpus)):
			content = re.sub(r'[{}]+'.format(punctuation), '', content)
			content = content.strip().lower()
			corpus[index][-1] = pattern.sub('', content).split()


	#6. calculate the average word embedding
	for corpus in [test_query,test_passage]:
		for index,(_,content) in enumerate(tqdm(corpus)):

			#initialized
			content_embedding = np.zeros((embedding_dim,))
			len_effective_content = 0

			#generating
			for word in content:
				if word in word_embedding.keys(): 
					content_embedding += word_embedding[word]
					len_effective_content += 1

			#calculate the sentense embedding
			if len_effective_content > 0:
				content_embedding /= len_effective_content

			#add to corpus
			corpus[index][-1] = content_embedding

	#7. save all the things
	with open("Sentense2VecTestSet.data","wb") as f: pickle.dump((test_query,test_passage,TestLUT),f)


if __name__ == '__main__':
	#Word2VecTraining(embedding_dim = 32)
	#Sentense2VecTraining(embedding_dim = 32)
	Testset2Vec(embedding_dim = 32)