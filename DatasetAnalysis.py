import pandas as pd 

class DatasetAnalysis(): 

    def __init__(self): 
      self = self 

    def read_dataset(self, path): 
      return pd.read_csv(path, sep="\n", header=None, names=["articles"] )

    def sample(self, df, n, path):
        df = df[:n]
        df.to_csv(path)
        return df

    def count_articles(self, cluster): 
        articles = cluster.split(sep = "|||||")
        return len(articles)
        
    def get_statistics(self, df1, df2, df3):
        df = pd.concat([df1, df2, df3], ignore_index = True)
        df["num_articles"] = df["articles"].apply(lambda x: self.count_articles(x))
        avg = df["num_articles"].mean()
        max_ = max(df["num_articles"])
        min_ = min(df["num_articles"])
        return avg, max_, min_ 



