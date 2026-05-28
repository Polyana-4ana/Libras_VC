import pandas as pd
import glob
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Para procurar todos os CSVs

arquivos = glob.glob("../dataset/*.csv")

dados = []
rotulos = []

for arquivo in arquivos:
    nome_gesto = (
        arquivo
        .split("\\")[-1]
        .replace(".csv", "")
    )

    df = pd.read_csv(
        arquivo,
        header= None
    )

    for _, linha in df.iterrows():
        dados.append(
            linha.values
        )

        rotulos.append(
            nome_gesto
        )

X_train, X_test, y_train, y_test = train_test_split(
    dados,
    rotulos,
    test_size=0.2,
    random_state=42
    )

modelo = RandomForestClassifier()

#Treinando a IA

modelo.fit(
    X_train,
    y_train
)

precisao = modelo.score(
    X_test,
    y_test
)

print(
    f"Precisão: {precisao * 100:.2f}%"
)

#Salvando o modelo

joblib.dump(
    modelo,
    "../models/libras_model.pkl"
)

print(
    "modelo salvo com sucesso!"
)