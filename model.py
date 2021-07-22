from auxiliary import Tokenizer
from union_find import Union_Find

import pandas as pd
import numpy as np
import math
import itertools
import operator
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm



class TFPDF(Tokenizer):
        
    def __init__(self, news_data: pd.DataFrame):
        Tokenizer.__init__(self)
                
        self.news_data = news_data
        self.news_dataframe = self._get_news_data_with_tokenized(news_data)
        self.channel_number = self.news_dataframe.iloc[:,0:2].groupby('press_code').count().to_dict()['press_name']
        self.documents_number = len(self.channel_number)
     
        self.news_code_list = self.get_channel_words_number()
        self.word_list = self.get_transformed_dataframe(self.news_dataframe.title_tokenized)
        self.channel_norm_dictionary = {news_code: self._get_channel_norm(news_code) for news_code in self.news_code_list}
        
        self.word_weight_dictionary = self._get_word_weight_dictionary()
        
        
    def _get_news_data_with_tokenized(self, news_dataframe):        
        news_dataframe = news_dataframe.drop_duplicates(subset = ['title']).reset_index(drop = True)
        news_dataframe = news_dataframe.drop_duplicates(subset = ['url']).reset_index(drop = True)
        news_dataframe['title_tokenized'] = news_dataframe.title.apply(lambda x: self.get_noun(x))
        
        return news_dataframe
        
            
    def _get_word_list(self, dataframe):
        count_vectorizer = CountVectorizer(min_df = 0, max_df = 1.0)        
        fit_transformed_data = count_vectorizer.fit_transform(dataframe)
        fit_transformed_dataframe  = pd.DataFrame(fit_transformed_data.A, columns = count_vectorizer.get_feature_names())
                                                       
        return fit_transformed_dataframe
    
    
    # N_c // Total number of documents in channel c
    def get_number_of_document_in_channel(self):
        counts = self.news_dataframe.iloc[:, 0:2].groupby('press_code').count().rename(columns = {'press_name':'counts'})
        counts_dictionary = counts.to_dict()
        
        return counts_dictionary
    
    
    # the weight of a term rom a channel is linearly proportional to the term's within channel frequency, and exponentially proportional to the ratio of document containing the term in the channel    
    def get_weight(self):
        number_of_channels = self.channel_number.to_dict() # D
        
        return number_of_channels
        
        
    def get_channel_words_number(self):
        news_list = self.news_dataframe.groupby('press_code').count().index
        
        return news_list
        
           
    def get_transformed_dataframe(self, documents):
        
        count_vectorizer = CountVectorizer(min_df = 0, max_df = 1.0)        
        fit_transformed_data = count_vectorizer.fit_transform(documents)        
        fit_transformed_dataframe  = pd.DataFrame(fit_transformed_data.A, columns = count_vectorizer.get_feature_names())
        
        return fit_transformed_dataframe        
                                                       
                                                       
    def count_term_frequency(self, documents, press_code = False):
        
        fit_transformed_dataframe = self.get_transformed_dataframe(documents)
        frequency_dictionary = fit_transformed_dataframe.apply(sum, axis = 0).to_dict()
        vocabs = fit_transformed_dataframe.columns
        
        return frequency_dictionary, vocabs
    
        
    # W_j
    def _get_word_weight(self, word):    

        word_weight = 0 # w_j
        for c in self.news_code_list:
            word_weight += ( self.get_normalized_frequency_of_term(word, c) * self.get_exp(word, c) )
        
        return word_weight
            
            
    # F_jc
    def get_normalized_frequency_of_term(self, word, channel):
        
        return math.sqrt(self._get_frequency_of_term(word, channel) / self.channel_norm_dictionary[channel])
        
    
    # sqrt(F_kj)
    def _get_channel_norm(self, channel):
        
        channel_word_index = self.news_dataframe[self.news_dataframe.press_code == channel].index
        channel_word_list = self.word_list.iloc[channel_word_index].T
        
        return math.sqrt(channel_word_list[channel_word_list.apply(lambda x: sum(x), axis = 1) > 0].T.sum().apply(lambda x: x ** 2).sum())
    
    
    # exp(n_jc/N_c)
    def get_exp(self, word, channel):
        
        return math.exp(self._get_frequency_of_term(word, channel) / self.channel_number[channel])
    
    
    # n_jc
    def _get_frequency_of_term(self, word, channel):
        
        return self._find_word_in_channel(word, channel).sum()
    
    
    def _find_word_in_channel(self, word, channel):
        
        return self.word_list.iloc[(self.news_dataframe.press_code == channel).to_list()][word]
    
    
    def _get_word_weight_dictionary(self):
        
        print('calculate words weight')
        word_weight_dictionary = {word: self._get_word_weight(word) for word in tqdm(self.word_list.columns)}
        word_weight_dictionary = sorted(word_weight_dictionary.items(), reverse = True, key = operator.itemgetter(1))
        
        return word_weight_dictionary
    
    
    def get_unit_vector(self):

        return list(map(lambda x: x[0], self.word_weight_dictionary[:int(len(self.word_weight_dictionary) * 0.01)]))
    
    
    def _get_filtered_news_dataframe(self):
        
        # unit vector
        unit_vector = self.get_unit_vector()
        self.news_dataframe['unit_vector'] = self.news_dataframe.title_tokenized.apply(lambda x: list(filter(lambda x: x in unit_vector, x.split(' ')))).apply(lambda x: list(set(x)))    
        filtered_news_dataframe = self.news_dataframe[self.news_dataframe.unit_vector.apply(lambda x: len(x)) > 2].reset_index(drop = True)    
        filtered_news_dataframe['vector'] = filtered_news_dataframe.unit_vector.apply(lambda x: set(x)).apply(lambda x: ' '.join(x))
        
        return filtered_news_dataframe
        
    
    
    def get_cluster(self):
        
        filtered_news_dataframe = self._get_filtered_news_dataframe()
        
        # clustering
        unit_vector_matrix = self.get_transformed_dataframe(filtered_news_dataframe.vector)
        unit_vector_normalized = unit_vector_matrix.apply(lambda x: 1 / math.sqrt(sum(x)), axis = 1).to_numpy()
        reshaped_unit_vector_normalized = unit_vector_normalized.reshape(len(unit_vector_normalized), 1)
        normalized_vector_matrix = np.dot(reshaped_unit_vector_normalized, reshaped_unit_vector_normalized.T)
        cosine_similarity_matrix = np.dot(unit_vector_matrix, unit_vector_matrix.T) * normalized_vector_matrix    
        np.fill_diagonal(cosine_similarity_matrix, 0)    
        similar_index_lists = np.where(cosine_similarity_matrix > math.cos(0.6154)) # theta = 35.26    
        
        union_find = Union_Find(filtered_news_dataframe.shape[0])
        for i, j in zip(*similar_index_lists): 
            union_find.union(i, j)
        union_set = np.array(union_find.data) 
        union_set_int_index = np.where(union_set >= 0) 
        union_set_int_data = union_set[union_set_int_index] 
        union_set_unique_data = np.unique(union_set_int_data) 
        union_set_lists = [np.append(i, np.where(union_set == i)) for i in union_set_unique_data]
        union_set_lists = sorted(union_set_lists, key = lambda x: len(x), reverse = True)
        
        return union_set_lists
    
    
    def get_keyword(self, index_list, N = 3):
        
        filtered_news_dataframe = self._get_filtered_news_dataframe()
        
        word_list = itertools.chain(*filtered_news_dataframe.iloc[index_list].unit_vector)
        high_frequency_list = Counter(word_list).most_common(N)
        frequency_list = list(map(lambda x: x[0], high_frequency_list))
        
        return frequency_list