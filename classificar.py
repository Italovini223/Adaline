import pandas as pd
import os

PATH = os.path.join(os.getcwd(), './datasets/validacao.xlsx')

def validar(pesos, limiarDeAtivacao, treino):

    df_validacao = pd.read_excel(PATH)
    

    colunas_numericas = ['x1', 'x2', 'x3', 'x4']
    
    for col in colunas_numericas:
        if col in df_validacao.columns:
            # Converte para string, troca vírgula por ponto e força para float
            df_validacao[col] = df_validacao[col].astype(str).str.replace(',', '.').apply(pd.to_numeric, errors='coerce')

    df_validacao[colunas_numericas] = df_validacao[colunas_numericas].fillna(0)


    # Inicializa a coluna de resultados para o treinamento atual
    df_validacao[f'Y_T{treino}'] = pd.NA 

    for index, row in df_validacao.iterrows():
        # Captura as entradas x1, x2, x3 e x4 conforme a planilha 
        x = [row['x1'], row['x2'], row['x3'], row['x4']]
        
        # 1. Calcula a saída linear (U) 
        # U = Σ(wi * xi) - θ 
        U = sum(w * xi for w, xi in zip(pesos, x)) - limiarDeAtivacao
        
        # 2. Função de Ativação (Degrau Bipolar / Função Sinal) [cite: 405]
        # Conforme o material da UEMG, se U >= 0, a amostra pertence a uma classe [cite: 406, 409]
        if U >= 0:
            y = 1
        else:
            y = -1
        
        # Salva o resultado da classificação na planilha de validação
        df_validacao.at[index, f'Y_T{treino}'] = y
  
    # Salva as alterações no arquivo Excel de validação
    df_validacao.to_excel(PATH, index=False)