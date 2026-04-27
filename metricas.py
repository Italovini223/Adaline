import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from math import sqrt

PATH = os.path.join(os.getcwd(), './datasets/validacao.xlsx')
METRICAS_PATH = os.path.join(os.getcwd(), './datasets/metricas.xlsx')

df_validacao = pd.read_excel(PATH)
# Tenta ler o arquivo de métricas ou cria um novo se não existir
if os.path.exists(METRICAS_PATH):
    df_metricas = pd.read_excel(METRICAS_PATH)
else:
    df_metricas = pd.DataFrame()

def calcular():
    d_real = df_validacao['d']
    redes = ['Y_T1', 'Y_T2', 'Y_T3', 'Y_T4', 'Y_T5']

    for i, rede in enumerate(redes, start=1):
        if rede not in df_validacao.columns:
            continue
            
        y_predito = df_validacao[rede]
        
        # Matriz de Confusão [cite: 405, 408]
        matrizDeConfusao = confusion_matrix(d_real, y_predito, labels=[-1, 1])
        verdadeiroNegativo, falsoPositivo, falsoNegativo, verdadeiroPositivo = matrizDeConfusao.ravel()

        # Métricas de Classificação (Existentes)
        acuracia = (verdadeiroPositivo + verdadeiroNegativo) / len(d_real)
        sensibilidade = verdadeiroPositivo / (verdadeiroPositivo + falsoNegativo) if (verdadeiroPositivo + falsoNegativo) > 0 else 0
        especificiddade = verdadeiroNegativo / (verdadeiroNegativo + falsoPositivo) if (verdadeiroNegativo + falsoPositivo) > 0 else 0
        precisao = verdadeiroPositivo / (verdadeiroPositivo + falsoPositivo) if (verdadeiroPositivo + falsoPositivo) > 0 else 0
        
        # Adição Adaline: Erro Médio Quadrático de Validação [cite: 397]
        # Como o Adaline busca minimizar a distância entre d e u, 
        # medir o erro entre d e a saída final ajuda a ver a estabilidade.
        mse_validacao = mean_squared_error(d_real, y_predito)
        rmse_validacao = sqrt(mse_validacao)

        # Preenchimento do DataFrame
        df_metricas.at[i-1, 'Rede'] = f'T{i}'
        df_metricas.at[i-1, 'Acertos'] = verdadeiroPositivo + verdadeiroNegativo
        df_metricas.at[i-1, 'Erros'] = falsoPositivo + falsoNegativo
        df_metricas.at[i-1, 'Acurácia'] = acuracia
        df_metricas.at[i-1, 'Sensibilidade'] = sensibilidade
        df_metricas.at[i-1, 'Especificidade'] = especificiddade
        df_metricas.at[i-1, 'Precisao'] = precisao
        df_metricas.at[i-1, 'RMSE_Validação'] = rmse_validacao # Nova métrica Adaline

        # Salvamento parcial (dentro ou fora do loop conforme preferência)
        df_metricas.to_excel(METRICAS_PATH, index=False)

        # Plotagem da Matriz de Confusão
        fig, ax = plt.subplots(figsize=(6, 5))
        # Ajustei os labels para Classe A e B conforme o seu PDF 
        display = ConfusionMatrixDisplay(confusion_matrix=matizDeConfusao, display_labels=['Classe A (-1)', 'Classe B (+1)'])
        display.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        ax.set_title(f"Rede T{i} - Matriz de Confusão (Adaline)")
        plt.tight_layout()
        
        # Garante que a pasta existe antes de salvar
        save_path = './graphics/matrizes_de_confusao/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        plt.savefig(f'{save_path}rede_T{i}.png')
        plt.close() # Importante fechar para não consumir memória

    print(f"Cálculo de métricas Adaline concluído. Resultados salvos em {METRICAS_PATH}")

# Chamada da função para teste
if __name__ == "__main__":
    calcular()