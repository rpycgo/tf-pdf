import re
from bs4 import BeautifulSoup
from kiwipiepy import Kiwi
from datetime import datetime




class Tokenizer:
    
    def __init__(self):
        self.kiwi = Kiwi()
        self.kiwi.prepare()
        

    def get_cleansed_string(self, string):
        
        string = BeautifulSoup(string, 'lxml').text
        string = re.sub('\t|\n', '' , string)
        
        return string
        
        
    def get_noun(self, string):
        
        string = self.get_cleansed_string(string)
        string = self.kiwi.analyze(string)[0][0]
        string = list(filter(lambda x: x[1].startswith('N'), string))
        string = list(map(lambda x: x[0], string))
        string = ' '.join(string)
        
        return string
    
    
    

class extractor:
    
    @staticmethod
    def getImageURLs(tag):
        
        tags = tag.select('img')
        tags = [i.get('src') for i in tags]
        
        return tags
    
    
    @staticmethod
    def getDate(date):        
        
        try:
            date = re.findall('(\d{4})\.(\d{2}).(\d{2})', date)[0]
            
            return '-'.join(date)
        
        except:
            return ''
     
    
    @staticmethod
    def getEnterDate(date):
        
        try:            
            enter_date = list(re.findall('(\d{4})\.(\d{2}).(\d{2})\.\s?([가-힣]{2})\s*(\d{1,2}):(\d{1,2})', date)[0])
            
            if enter_date[3] == '오전':
                enter_date.remove('오전')
                if enter_date[3] == 12:
                    enter_date[3] -= 12
                enter_date = list(map(lambda x: int(x), enter_date))
                enter_date = datetime(*enter_date)
                        
            elif enter_date[3] == '오후':
                enter_date.remove('오후')
                enter_date = list(map(lambda x: int(x), enter_date))
                if enter_date[3] != 12:
                    enter_date[3] += 12
                enter_date = datetime(*enter_date)
        except:
            enter_date = ''
        
        return enter_date        