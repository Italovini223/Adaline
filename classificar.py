import pandas as pd
import os

PATH = os.path.join(os.getcwd(), './datasets/validacao.xlsx')

def validar(pesos, limiarDeAtivacao, treino):
    """Fase de operação do Adaline para classificar Válvula A ou B."""
    
    # Carrega a planilha a cada chamada para garantir dados frescos
    df_validacao = pd.read_excel(PATH)
    
    # =====================================================================
    # --- INÍCIO DO TRATAMENTO DE DADOS (PROTEÇÃO CONTRA ERROS NO EXCEL) ---
    # =====================================================================
    # Como na validação usamos apenas as entradas para calcular a saída,
    # tratamos as colunas x1, x2, x3 e x4. A coluna 'd' é tratada nas métricas.
    colunas_numericas = ['x1', 'x2', 'x3', 'x4']
    
    for col in colunas_numericas:
        if col in df_validacao.columns:
            # Converte para string, troca vírgula por ponto e força para float
            df_validacao[col] = df_validacao[col].astype(str).str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
    
    # Preenche valores que ficaram como NaN com 0 (ou pode usar .dropna() se preferir excluir a linha)
    # Na validação, excluir linha pode bagunçar a ordem, então preencher com 0 é mais seguro para testes acadêmicos
    df_validacao[colunas_numericas] = df_validacao[colunas_numericas].fillna(0)
    # =====================================================================
    # --- FIM DO TRATAMENTO DE DADOS ---
    # =====================================================================

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