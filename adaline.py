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

# =====================================================================
# --- INÍCIO DO TRATAMENTO DE DADOS (CORREÇÃO DO ERRO) ---
# =====================================================================
# Lista das colunas numéricas que o modelo vai utilizar
colunas_numericas = ['x1', 'x2', 'x3', 'x4', 'd']

for col in colunas_numericas:
    # 1. Transforma tudo em texto (string)
    # 2. Troca vírgulas por pontos (para o Python entender os decimais)
    # 3. Força a conversão para número numérico. 
    # O "errors='coerce'" transforma qualquer letra intrusa (como aquele "S") em NaN (dado inválido/vazio)
    df_treinamento[col] = df_treinamento[col].astype(str).str.replace(',', '.').apply(pd.to_numeric, errors='coerce')

# Remove qualquer linha inteira que tenha ficado com NaN (exclui a linha com defeito)
df_treinamento = df_treinamento.dropna()
# =====================================================================
# --- FIM DO TRATAMENTO DE DADOS ---
# =====================================================================

treinamento = 1
taxaDeAprendizagem = 0.01 # Definida cautelosamente para evitar instabilidade 
precisao = 1e-6 # Precisão requerida (epsilon)

resultados.limpar(df_resultados)

while treinamento <= 5:
    epocas = 0
    # Inicialização de 4 pesos para x1, x2, x3, x4
    # Ajustei para (0, 1) conforme o PDF do seu projeto pede.
    pesos = [random.uniform(0, 1) for _ in range(4)] 
    limiarDeAtivacao = random.uniform(0, 1)
    rmse_por_epoca = []
    
    mse_anterior = float('inf')

    resultados.preencher_w_iniciais(df_resultados, treinamento, pesos, limiarDeAtivacao)

    while epocas < 1000:
        epocas += 1
        soma_erro_quadratico = 0

        # Eanterior recebe o Eqm da época passada
        for index, row in df_treinamento.iterrows():
            x = [row['x1'], row['x2'], row['x3'], row['x4']] 
            d = row['d']
            
            # 1. Saída do combinador linear (u)
            u = sum(w * xi for w, xi in zip(pesos, x)) - limiarDeAtivacao
            
            # 2. Cálculo do erro linear (d - u)
            erro_linear = d - u
            
            # 3. Regra Delta: Atualização discreta dos pesos
            for i in range(len(pesos)):
                pesos[i] = pesos[i] + taxaDeAprendizagem * erro_linear * x[i]
            
            # Atualização do limiar considerando a entrada fixa -1 
            limiarDeAtivacao = limiarDeAtivacao + taxaDeAprendizagem * erro_linear * (-1)
            
            soma_erro_quadratico += (erro_linear ** 2)

        # 4. Cálculo do Erro Médio Quadrático (Eqm) atual
        mse_atual = soma_erro_quadratico / len(df_treinamento)
        rmse_epoca = sqrt(mse_atual)
        rmse_por_epoca.append(rmse_epoca)

        # Critério de parada: Diferença absoluta entre Eqm atual e anterior 
        if abs(mse_atual - mse_anterior) <= precisao:
            break
        
        mse_anterior = mse_atual

    # Finalização e Classificação Final (Fase de Operação)
    pesos_finais = pesos.copy()
    for index, row in df_treinamento.iterrows():
        x = [row['x1'], row['x2'], row['x3'], row['x4']]
        u = sum(w * xi for w, xi in zip(pesos_finais, x)) - limiarDeAtivacao
        # Função de ativação para gerar saída binária y
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
    
    # Garantia de que a pasta existe antes de tentar salvar o gráfico
    grafico_dir = './graphics/Evolucao_do_erro/'
    if not os.path.exists(grafico_dir):
        os.makedirs(grafico_dir)
        
    plt.savefig(f'{grafico_dir}treinamento_{treinamento}.png')
    plt.close()
                
    treinamento += 1

# Por fim, chama o cálculo de métricas (que lê a aba de validação atualizada)
metricas.calcular()