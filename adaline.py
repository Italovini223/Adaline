import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error

import resultados
import classificar
import metricas

LOCAL_PATH = os.path.join(os.getcwd(), './datasets/treinamento.xlsx')
RESULTADOS_PATH = os.path.join(os.getcwd(), './datasets/resultados.xlsx')

df_treinamento = pd.read_excel(LOCAL_PATH)
df_resultados = pd.read_excel(RESULTADOS_PATH)

treinamento = 1
taxaDeAprendizagem = 0.0025
precisao = 1e-6 

resultados.limpar(df_resultados)

while treinamento <= 5:
    epocas = 0
    pesos = [random.uniform(0, 1) for _ in range(4)] 
    limiarDeAtivacao = random.uniform(0, 1)
    rmse_por_epoca = []
    
    mse_anterior = float('inf')

    resultados.preencher_w_iniciais(df_resultados, treinamento, pesos, limiarDeAtivacao)

    while epocas < 1000:
        epocas += 1
        soma_erro_quadratico = 0

        for index, row in df_treinamento.iterrows():
            x = [row['x1'], row['x2'], row['x3'], row['x4']] 
            d = row['d']
            
            # Cálculo da saída linear
            u = sum(w * xi for w, xi in zip(pesos, x)) - limiarDeAtivacao
            
            # Erro linear da Regra Delta
            erro_linear = d - u
            
            # Atualização dos pesos
            for i in range(len(pesos)):
                pesos[i] = pesos[i] + taxaDeAprendizagem * erro_linear * x[i]
            
            limiarDeAtivacao = limiarDeAtivacao + taxaDeAprendizagem * erro_linear * (-1)
            soma_erro_quadratico += (erro_linear ** 2)

        mse_atual = soma_erro_quadratico / len(df_treinamento)
        rmse_epoca = sqrt(mse_atual)
        rmse_por_epoca.append(rmse_epoca)

        if abs(mse_atual - mse_anterior) <= precisao:
            break
        
        mse_anterior = mse_atual

    pesos_finais = pesos.copy()
    for index, row in df_treinamento.iterrows():
        x = [row['x1'], row['x2'], row['x3'], row['x4']]
        u = sum(w * xi for w, xi in zip(pesos_finais, x)) - limiarDeAtivacao
        y = 1 if u >= 0 else -1
        df_treinamento.at[index, f'Y_{treinamento}'] = y
    
    df_treinamento.to_excel(LOCAL_PATH, index=False)
    print(f'Treinamento {treinamento} concluído em {epocas} épocas.')

    resultados.preencher_w_finais(df_resultados, treinamento, pesos_finais, epocas, limiarDeAtivacao)
    classificar.validar(pesos_finais, limiarDeAtivacao, treinamento)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rmse_por_epoca)+1), rmse_por_epoca)
    plt.title(f'Curva de Aprendizado Adaline - Treinamento {treinamento}')
    plt.xlabel('Épocas')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    grafico_dir = './graphics/Evolucao_do_erro/'
    if not os.path.exists(grafico_dir):
        os.makedirs(grafico_dir)
        
    plt.savefig(f'{grafico_dir}treinamento_{treinamento}.png')
    plt.close()
                
    treinamento += 1
        
metricas.calcular()