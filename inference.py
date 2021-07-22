# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:20:16 2021

@author: MJH
"""

from model import TFPDF
from auxiliary import extractor

import re
from bs4 import BeautifulSoup
from datetime import datetime




def main(news_data):
    
    tfpdf = TFPDF(news_data)
    
    filtered_news_dataframe = tfpdf._get_filtered_news_dataframe() 
    
    filtered_news_dataframe.title = filtered_news_dataframe.title.apply(lambda x: BeautifulSoup(x, 'lxml').text)
    filtered_news_dataframe.title = filtered_news_dataframe.title.apply(lambda x: re.sub('[\n|\t]', '', x))
    
    filtered_news_dataframe['date'] = filtered_news_dataframe.date_upload.apply(lambda x: extractor.getDate(x))
    filtered_news_dataframe.date = pd.to_datetime(filtered_news_dataframe.date, format = '%Y-%m-%d')
    filtered_news_dataframe['time'] = filtered_news_dataframe.date
    
    union_set_lists = tfpdf.get_cluster()
    
    main_list = []
    for union_set in union_set_lists[:10]:
        temp_dataframe = filtered_news_dataframe.iloc[union_set]
        sub_main_news = temp_dataframe.iloc[0]
        contents = BeautifulSoup(sub_main_news.contents, 'lxml')
        similar = [{'title': row.title, 'url':row.url, 'press_name': row.press_name, 'date': row.date} for row in temp_dataframe.iloc[1:].itertuples()]
        main_dictionary = {
            'date': sub_main_news.date,
            'keyword': tfpdf.get_keyword(union_set),
            'title': sub_main_news.title,
            'press_name': sub_main_news.press_name,
            'url': sub_main_news.url,
            'similar': similar
            }
        
        main_list.append(main_dictionary)
    
    return main_list
    

    
    
if __name__ == '__main__':
    news_data = pd.read_csv(input())
    main(news_data)
    