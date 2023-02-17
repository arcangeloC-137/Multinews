import regex as re 
import pandas as pd 
import ast 
import string 
from thext_extended import Highlighter
from thext_extended import SentenceRankerPlus 
import nltk 

class HighlighterContext(): 
    
    def __init__(self): 
        self = self 

    def clean_text(self, text):
        clean = text.replace('\n',' ')
        clean = re.sub(' +', ' ', clean)
        return clean 

    def preprocess_data(self, df): 
        text_list = list()
        for index, row in df.iterrows(): 
            text = self.clean_text(row['papers']['full_text'])
            text_list.append(text)
        return text_list

    def tokenize_text(self, full_text_list): 
        res_dict = {}
        for idx, text in enumerate(full_text_list):
            tokenized_text = nltk.sent_tokenize(text)
            res_dict[idx] = {'full_text':tokenized_text}
        df_full = pd.DataFrame(res_dict).T
        return df_full 

    def generate_abstracts(self, predictions): 
        abstracts = []
        for k in range(len(predictions)):
            line=predictions[k].split()[1:-1]
            abstract=''
            for i in range(len(line)):
                abstract=abstract+line[i]+' '
                abstract=abstract.translate(str.maketrans('','',str("''")))
            abstracts.append(abstract)
        return abstracts 

    def HiglighterGeneration(self, path_df, path_context): 
        df = pd.read_json(path_df)
        text_list = self.preprocess_data(df)
        df_full = self.tokenize_text(text_list)
        df_context = pd.read_csv(path_context)
        predictions = df_context['prediction']
        abstracts = self.generate_abstracts(predictions)
        sentences_full_text = df_full['full_text']
        sentences = list()

        for s in sentences_full_text:
            sentence = ast.literal_eval(str(s))
            sentences.append(sentence)
            
        sr = SentenceRankerPlus()
        h = Highlighter(sr)
        index = 0
        for s in sentences:
          flag = False
          for x in s: 
            if (h.valid_sentence(x) == True): 
                flag = True 

          if (flag == False): 
            sentences.pop(index)
            abstracts.pop(index)
          index += 1

        return sentences, abstracts 
        


