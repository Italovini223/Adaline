import pandas as pd
import os

RESULTADOS_PATH = os.path.join(os.getcwd(), './datasets/resultados.xlsx')

def limpar(df_resultados):
    """Limpa todas as colunas de pesos e épocas para um novo ciclo de treinamento."""
    for index, row in df_resultados.iterrows():
        # Limpa pesos iniciais (incluindo o novo W4 e o limiar W0)
        df_resultados.at[index, 'W0-inicial'] = pd.NA
        df_resultados.at[index, 'W1-inicial'] = pd.NA
        df_resultados.at[index, 'W2-inicial'] = pd.NA
        df_resultados.at[index, 'W3-inicial'] = pd.NA
        df_resultados.at[index, 'W4-inicial'] = pd.NA

        # Limpa pesos finais (incluindo o novo W4 e o limiar W0)
        df_resultados.at[index, 'W0-final'] = pd.NA
        df_resultados.at[index, 'W1-final'] = pd.NA
        df_resultados.at[index, 'W2-final'] = pd.NA
        df_resultados.at[index, 'W3-final'] = pd.NA
        df_resultados.at[index, 'W4-final'] = pd.NA
        
        df_resultados.at[index, 'Numero-de-epocas'] = pd.NA
        
    df_resultados.to_excel(RESULTADOS_PATH, index=False)

def preencher_w_iniciais(df_resultados, treinamento, pesos, limiarDeAtivacao):
    """Registra os valores aleatórios iniciais antes do treinamento Adaline."""
    # O limiar theta é tratado como W0 conforme o modelo (pág 4 e 11)
    df_resultados.at[treinamento - 1, 'W0-inicial'] = limiarDeAtivacao
    df_resultados.at[treinamento - 1, 'W1-inicial'] = pesos[0]
    df_resultados.at[treinamento - 1, 'W2-inicial'] = pesos[1]
    df_resultados.at[treinamento - 1, 'W3-inicial'] = pesos[2]
    df_resultados.at[treinamento - 1, 'W4-inicial'] = pesos[3] # Adicionado X4
    df_resultados.to_excel(RESULTADOS_PATH, index=False)

def preencher_w_finais(df_resultados, treinamento, pesos_finais, epocas, limiarDeAtivacao):
    """Registra os valores ótimos encontrados pela Regra Delta ao fim do treinamento."""
    df_resultados.at[treinamento - 1, 'W0-final'] = limiarDeAtivacao
    df_resultados.at[treinamento - 1, 'W1-final'] = pesos_finais[0]
    df_resultados.at[treinamento - 1, 'W2-final'] = pesos_finais[1]
    df_resultados.at[treinamento - 1, 'W3-final'] = pesos_finais[2]
    df_resultados.at[treinamento - 1, 'W4-final'] = pesos_finais[3] # Adicionado X4
    df_resultados.at[treinamento - 1, 'Numero-de-epocas'] = epocas
    df_resultados.to_excel(RESULTADOS_PATH, index=False)