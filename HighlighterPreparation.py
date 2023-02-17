import pandas as pd 
import ast
import string 

class HighlighterPreparation: 
    def __init__(self): 
        self = self 
    
    def open_files_PBSUM(self, df_path, new_abstract_path, full_text_path): 
        df_CS_test = pd.read_json(df_path)
        df_CS_test[:80]

        df_abstract = pd.read_csv(new_abstract_path)
        predictions = df_abstract['prediction']
        abstracts = []
        for k in range(len(predictions)): 
            line=predictions[k].split()[1:-1]
            abstract=''
            for i in range(len(line)):
                abstract=abstract+line[i]+' '
                abstract=abstract.translate(str.maketrans('','',str("''")))
            abstracts.append(abstract)
        
        df = pd.read_csv(full_text_path)
        sents = df['full_text']
        sentences = []
        for s in sents:
            sentence = ast.literal_eval(s)
            sentences.append(sentence)

        return sentences, abstract 



    def prepare_data_MultiNews(self, data_path): 
        df=pd.read_csv(data_path, header = None)
        test = df[1].apply(lambda x: ast.literal_eval(x))
        return test 


    def convert_highlights_in_dataframe(self, list_highlights): 
        df_high = pd.DataFrame(list_highlights)
        return df_high