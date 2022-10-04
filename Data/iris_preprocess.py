import pandas as pd

df=pd.read_csv("iris.csv")
# df=df.iloc[1:]
df=df.replace("Setosa",1)
df=df.replace("Versicolor",2)
df=df.replace("Virginica",3)

final=df.to_csv("iris_preprocessed.csv")