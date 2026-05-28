import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import glob

arquivos = glob.glob("../dataset/*.csv")

dados=[]

rotulos=[]

for arquivo in arquivos:

    nome=arquivo.split("\\")[-1].replace(".csv","")

    df=pd.read_csv(arquivo,header=None)

    for _,linha in df.iterrows():

        dados.append(linha.values)

        rotulos.append(nome)

X_train,X_test,y_train,y_test=train_test_split(
    dados,
    rotulos,
    test_size=0.2
)

modelo=RandomForestClassifier()

modelo.fit(X_train,y_train)

print(
    "Precisão:",
    modelo.score(X_test,y_test)
)

joblib.dump(
    modelo,
    "../models/libras_model.pkl"
)