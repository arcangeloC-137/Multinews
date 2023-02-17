import pandas as pd 
import rouge_score 
from rouge_score import rouge_scorer

class HighlighterEvaluation: 
    def __init__(self): 
        self = self 

    def open_files(self, path_original_highlights, path_highlights): 
        with open(path_original_highlights, encoding="utf-8") as f:
            list_original_highlights = f.readlines()
        
        df=pd.read_csv(path_highlights)
        df=df.fillna('')
        list_high=df.values.tolist()

        list_highlights=[]
        for lista in list_high:
            sentence=''
            for i in range(len(lista)):
                sentence=sentence+str(lista[i])
            list_highlights.append(sentence)

        return list_original_highlights, list_highlights 

    def evaluate_highlights(self, list_highlights, list_original_highlights): 
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = []
        for i in range(len(list_highlights)):
            scores.append(scorer.score(list_highlights[i], list_original_highlights[i]))

        return scores 

    def evaluate_highlights_led(self, list_highlights, list_original_highlights): 
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = []
        lhd=[]
        for l in list_highlights:
          sentence=''
          for i in range(len(l)):
            sentence=sentence+l[i]
          lhd.append(sentence)

        for i in range(len(lhd)):
            scores.append(scorer.score(lhd[i], list_original_highlights[i]))

        return scores 
