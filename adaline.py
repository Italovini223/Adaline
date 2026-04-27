import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error

import resultados
import classificar
import metricas

LOCAL_PATH = os.path.join(os.getcwd(), './datasets/treinamento.xlsx')
RESULTADOS_PATH = os.path.join(os.getcwd(), './datasets/resultados.xlsx')

df_treinamento = pd.read_excel(LOCAL_PATH)
df_resultados = pd.read_excel(RESULTADOS_PATH)

treinamento = 1
taxaDeAprendizagem = 0.01 # Definida cautelosamente para evitar instabilidade 
precisao = 1e-6 # Precisão requerida (epsilon) [cite: 384]

resultados.limpar(df_resultados)

while treinamento <= 5:
    epocas = 0
    # Inicialização de 4 pesos para x1, x2, x3, x4 [cite: 383]
    pesos = [random.uniform(-1, 1) for _ in range(4)] 
    limiarDeAtivacao = random.uniform(-1, 1)
    rmse_por_epoca = []
    
    mse_anterior = float('inf')

    resultados.preencher_w_iniciais(df_resultados, treinamento, pesos, limiarDeAtivacao)

    while epocas < 1000:
        epocas += 1
        soma_erro_quadratico = 0

        # Eanterior recebe o Eqm da época passada [cite: 387]
        # (Neste código, controlado pela variável mse_anterior)

        for index, row in df_treinamento.iterrows():
            x = [row['x1'], row['x2'], row['x3'], row['x4']] 
            d = row['d']
            
            # 1. Saída do combinador linear (u) [cite: 309, 313, 389]
            u = sum(w * xi for w, xi in zip(pesos, x)) - limiarDeAtivacao
            
            # 2. Cálculo do erro linear (d - u) [cite: 263, 389]
            erro_linear = d - u
            
            # 3. Regra Delta: Atualização discreta dos pesos [cite: 309, 389]
            for i in range(len(pesos)):
                pesos[i] = pesos[i] + taxaDeAprendizagem * erro_linear * x[i]
            
            # Atualização do limiar considerando a entrada fixa -1 
            limiarDeAtivacao = limiarDeAtivacao + taxaDeAprendizagem * erro_linear * (-1)
            
            soma_erro_quadratico += (erro_linear ** 2)

        # 4. Cálculo do Erro Médio Quadrático (Eqm) atual [cite: 392, 397]
        mse_atual = soma_erro_quadratico / len(df_treinamento)
        rmse_epoca = sqrt(mse_atual)
        rmse_por_epoca.append(rmse_epoca)

        # Critério de parada: Diferença absoluta entre Eqm atual e anterior 
        if abs(mse_atual - mse_anterior) <= precisao:
            break
        
        mse_anterior = mse_atual

    # Finalização e Classificação Final (Fase de Operação) [cite: 400]
    pesos_finais = pesos.copy()
    for index, row in df_treinamento.iterrows():
        x = [row['x1'], row['x2'], row['x3'], row['x4']]
        u = sum(w * xi for w, xi in zip(pesos_finais, x)) - limiarDeAtivacao
        # Função de ativação para gerar saída binária y [cite: 257, 405]
        y = 1 if u >= 0 else -1
        df_treinamento.at[index, f'Y_{treinamento}'] = y
    
    df_treinamento.to_excel(LOCAL_PATH, index=False)
    print(f'Treinamento {treinamento} concluído em {epocas} épocas.')

    # Gravação dos resultados finais atualizados para 4 pesos
    resultados.preencher_w_finais(df_resultados, treinamento, pesos_finais, epocas, limiarDeAtivacao)
    classificar.validar(pesos_finais, limiarDeAtivacao, treinamento)
    
    # Geração do gráfico de RMSE (Evolução do erro médio quadrático) 
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rmse_por_epoca)+1), rmse_por_epoca)
    plt.title(f'Curva de Aprendizado Adaline - Treinamento {treinamento}')
    plt.xlabel('Épocas')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.savefig(f'./graphics/Evolucao_do_erro/treinamento_{treinamento}.png')
    plt.close()
                
    treinamento += 1
        
metricas.calcular()