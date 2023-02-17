from numpy.matrixlib.defmatrix import N
import pandas as pd 
import numpy as np 
import ast 
import nltk
import csv 
import thext_extended
from thext_extended import DatasetPlus 

nltk.download('punkt')

class DatasetPreparation: 
    def __init__(self):
      self = self
        
    def read_dataset(self, path): 
      df = pd.read_csv(path)
      df = df.drop(columns = "Unnamed: 0")
      return df

    def split_df(self, df): 
        return df.join(df["articles"].str.split('story_separator_special_tag', expand = True, n=1))

    def create_dfCount(self, df): 
        df = df.fillna(" ")
        df_count = df.applymap(lambda x: len(nltk.sent_tokenize(x)) if isinstance(x, str) else x)
        df_count = df_count.drop('articles', axis=1)
        df_count = df_count.replace(0, np.NaN)
        df_count["mean"] = df_count.mean(axis = 'columns')
        return df_count

    def create_dfPercentage(self, df, percentage): 
        df_percentage = df.apply(lambda x: ((x*percentage)/100)+0.5)
        df_percentage = df_percentage.round()
        df_percentage = df_percentage.drop('mean', axis=1)
        df_percentage = df_percentage.fillna(0)
        return df_percentage

    def create_dfTokenized(self, df): 
        df_tokenized = df.applymap(lambda x: (nltk.sent_tokenize(x)) if isinstance(x, str) else x)
        df['tokenized_articles'] = df_tokenized['articles']
        df_tokenized = df_tokenized.drop('articles', axis = 1)
        return df_tokenized
        
    def assign_abstract(self, df, df_tokenized, df_percentage): 
        ab = list()
        for i in range(df_tokenized.shape[0]): 
            sentences = str()
            for j in range(df_tokenized.shape[1]): 
                number_of_sentences = df_percentage[j][i]
                count = 0
                for x in df_tokenized[j][i]:
                    if count <= number_of_sentences:
                        sentences = sentences + x
                        count += 1
                    else:
                        break 
            ab.append(sentences)
        df = df.assign(abstract = ab)
        df_def = pd.DataFrame().assign(articles=df['articles'], abstract=df['abstract'])
        return df_def 

    def preparation_test_for_DatasetPlus(self, path, percentage=20):
        df = self.read_dataset(path)
        df = self.split_df(df)
        df_count = self.create_dfCount(df)
        df_percentage = self.create_dfPercentage(df_count, percentage)
        df_tokenized = self.create_dfTokenized(df)
        df = self.assign_abstract(df, df_tokenized, df_percentage)
        list_text = list(df['articles'])
        list_abstract = list(df['abstract'])
        
        return list_text, list_abstract

    def preparation_train_for_DatasetPlus(self, path, path_label, n, percentage=20): 
        df = self.read_dataset(path)
        df = self.split_df(df)
        df_count = self.create_dfCount(df)
        df_percentage = self.create_dfPercentage(df_count, percentage)
        df_tokenized = self.create_dfTokenized(df)
        df = self.assign_abstract(df, df_tokenized, df_percentage)
        df2 =  pd.read_csv(path_label, sep="\n", header= None, names = ["summary"])
        df2 = df2[:n]
        df2['highlights'] = df2.apply(lambda row: nltk.sent_tokenize(row['summary']), axis=1)
        df_def = pd.DataFrame().assign(articles=df['articles'], abstract=df['abstract'], highlights = df2['highlights'])
        
        list_text = list(df_def['articles'])
        list_abstract = list(df_def['abstract'])
        list_highlights = list(df_def['highlights'])
        list_highlights = ["" if str(x)=='nan' else x for x in list_highlights]

        return list_text, list_abstract, list_highlights


    def readFileFromDatasetPlus(self, path): 
        df = pd.read_csv(path, header = None, names = ["index", "dictionary"])
        df = df["dictionary"].apply(lambda x: ast.literal_eval(x))
        return df

    def preparation_for_train(self, path_train, path_validation): 
        train_set = self.readFileFromDatasetPlus(path_train)
        validation_set = self.readFileFromDatasetPlus(path_validation)
        return train_set, validation_set 

    def save_output_of_DatasetPlus(self, path, dt): 
        # define a dictionary with key value pairs
        my_dict = dt.dataset
        # open file for writing, "w" is writing
        with open(path, "w", encoding='utf-8') as f:
            # create csv writer object
            w = csv.writer(f)
            # loop over dictionary keys and values
            for key, val in my_dict.items():
                # write every key and value to file
                w.writerow([key, val])