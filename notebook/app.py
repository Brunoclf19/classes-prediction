#!/usr/bin/env python
# coding: utf-8

# In[48]:


# Abrindo a conexão com o banco de dados

import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import gradio as gr
import matplotlib.pyplot as plt


# In[49]:


base = r"C:\Users\Giovanna\OneDrive\Documentos\classes-prediction\base\database.db"
conn = sqlite3.connect(base)


# In[50]:


# consulta dos dados no banco de dados
consulta_atividade = """
 SELECT
 *
 FROM
 flight_activity fa
 LEFT JOIN flight_loyalty_history flh
 ON (fa.loyalty_number=flh.loyalty_number)
"""
df_atividade = pd.read_sql_query(consulta_atividade,conn)

# Consulta de tabela
df_atividade


# In[51]:


# Fechando conexão com o banco de dados
conn.close()


# -------

# Parte 2
# 
# 1.4 Coletando os dados para análise:
# 
# Para executar o próximo passo do plano de solução do problema de negócio, precisamos coletar os dados do banco de dados

# In[52]:


df1 = df_atividade.copy()


# 4.0. Preparando os dados para treinamento do algoritmo

# In[53]:


# selecionando somente as colunas numéricas para o modelo
colunas = ['year', 'month', 'flights_booked',
'flights_with_companions', 'total_flights',
'distance', 'points_accumulated', 'salary',
'clv', 'loyalty_card']

df_colunas_selecionadas = df_atividade.loc[:, colunas]

# removendo linhas com alguma coluna vazia
df_treinamento = df_colunas_selecionadas.dropna()

# verificando o numero de linhas vazias
df_treinamento.isna().sum()


# In[54]:


# selecionando somente as colunas numéricas para o modelo
colunas_scr = ['year','month','flights_booked','flights_with_companions','total_flights','points_accumulated','salary','clv','loyalty_card']

df_colunas_selecionadas_scr = df_atividade.loc[:, colunas_scr]

# removendo linhas com alguma coluna vazia
df_treinamento_scr = df_colunas_selecionadas_scr.dropna()

# verificando o numero de linhas vazias
df_treinamento_scr.isna().sum()


# In[55]:


# Verificando quantidade de linhas nos dados 
df_treinamento.shape[0]


# In[56]:


# Verificando quantidade de colunas nos dados 
df_treinamento.head()


# Análise exploratória dos dados

# In[57]:


df_treinamento.describe()


# In[58]:


#Plot da densidade de distribuição do fenômeno que queremos prever
sns.displot(df_treinamento['loyalty_card'],kde=False)


# Maior volume de clientes com cartão Star.

# In[59]:


# Visualização da Distribuição das Variáveis
df_treinamento.hist(bins=20, figsize=(15,10))
plt.show()


# Dado a forma das distribuições das variáveis, vemos que a maior parte das pessoas das pessoas fazem poucos voos, com curtas distâncias e de acordo com o seu comportamento, há poucos pontos acumulados e um baixo CLV. Ou seja, há um desempenho padrão nos dados apresentados.

# In[60]:


# Correlação entre Variáveis
correlacao = df_treinamento.iloc[:,0:9].corr()
print(correlacao)


# In[61]:


# Visualização da Correlação
plt.figure(figsize=(12, 8))
sns.heatmap(correlacao, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação')
plt.show()


# Treinando algoritmo - machine learning

# In[62]:


from sklearn import tree as tr
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# ### Árvore de decisão

# In[63]:


X = df_treinamento.drop( columns=['loyalty_card'] )
y = df_treinamento.loc[:, 'loyalty_card']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[64]:


# Definindo o algoritmo Decision Tree
modelo = tr.DecisionTreeClassifier(max_depth=10)


# In[65]:


# Treinar o modelo no conjunto de treinamento
modelo_treinado = modelo.fit(X_train, y_train)


# In[66]:


# Fazer previsões no conjunto de teste
previsoes = modelo_treinado.predict(X_test)


# In[67]:


# Calcular a acurácia
acuracia = accuracy_score(y_test, previsoes)
print("Acurácia do modelo:", acuracia)


# ### Random Forest

# In[68]:


X = df_treinamento_scr.drop( columns=['loyalty_card'] )
y = df_treinamento_scr.loc[:, 'loyalty_card']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[69]:


# Definir o modelo Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)


# In[70]:


# Treinar o modelo Random Forest no conjunto de treinamento
modelo_rf_treinado = modelo_rf.fit(X_train, y_train)


# In[71]:


# Fazer previsões no conjunto de teste
previsoes_rf = modelo_rf_treinado.predict(X_test)


# In[72]:


# Calcular a acurácia do modelo Random Forest
acuracia_rf = accuracy_score(y_test, previsoes_rf)
print("Acurácia do modelo Random Forest:", acuracia_rf)


# Dado o cenário que estávamos

# In[73]:


# Visualização gráfica do modelo treinado
tr.plot_tree(modelo,
             class_names=['Aurora', 'Nova', 'Star'],
             filled=True);


# In[76]:


# Demonstração do resultado da previsão do modelo
X_novo = df_treinamento.drop(columns=['loyalty_card']).sample()
previsao = modelo.predict_proba(X_novo)
print('Probabilidades de cartão para o cliente - Aurora: {:.2f}, Nova: {:.2f}, Star: {:.2f}'.format(100*previsao[0][0], 100*previsao[0][1], 100*previsao[0][2]))


# In[77]:


# Treinando o algoritmo Decision Tree
modelo_final_rf = modelo_rf.fit(X, y)


# In[78]:


# Demonstração do resultado da previsão do modelo
X_novo = X.sample()
previsao = modelo_final_rf.predict_proba(X_novo)
print('Probabilidades de cartão para o cliente - Aurora: {:.2f}, Nova: {:.2f}, Star: {:.2f}'.format(100*previsao[0][0], 100*previsao[0][1], 100*previsao[0][2]))


# Painel publicado - Propensão de cliente

# In[79]:


# Testes para Slicer
X.year.min()


# In[80]:


# Testes para Slicer
X.year.max()


# In[82]:


# Definindo minha previsão

# Função de recebimento dos dados 
def predict(* args):

    # Guardando em um array
    X = np.array([args]).reshape(1,-1)

    
    previsao = modelo_final_rf.predict_proba(X)
    return{"Aurora": previsao[0][0], "Nova": previsao[0][1], "Star": previsao[0][2]}




# Criando a vizualização em painél nomeado "demo"
with gr.Blocks() as demo:
    # Título do painél
    gr.Markdown('''# Propensão de Compra''')

    with gr.Row():
        with gr.Column():
            gr.Markdown('''# Dados de voo do cliente''')
            year                        = gr.Slider(label="Ano de voo", minimum=2017, maximum=2018, step=1, randomize=True)
            month                       = gr.Slider(label="Mês de voo", minimum=1, maximum=12, step=1, randomize=True)
            #flights_booked              = gr.Slider(label="Voos Reservados", minimum=0, maximum=21, step=1, randomize=True)
            flights_with_companions     = gr.Slider(label="Voos Acompanhados", minimum=0, maximum=11, step=1, randomize=True)
            total_flights               = gr.Slider(label="Total de voos", minimum=0, maximum=32, step=1, randomize=True)
            distance                    = gr.Slider(label="Diastância média percorrida", minimum=0, maximum=6293, step=1, randomize=True)
            points_accumulated          = gr.Slider(label="Pontos Acumulados", minimum=0.00, maximum=676.50, step=0.1, randomize=True)
            salary                      = gr.Slider(label="Salário Anual", minimum=58486.00, maximum=407228.00, step=0.1, randomize=True)
            clv                         = gr.Slider(label="CLV", minimum=2119.89, maximum=83325.38, step=0.1, randomize=True)

            with gr.Row():
                gr.Markdown('''# Botão de previsão''')

                # Botão
                predict_btn = gr.Button(value='Previsão')

        with gr.Column():
            gr.Markdown('''# Cartão sugerido''')

            # Resposta da previsão
            resposta = gr.Label()
        

    # Botão de previsão
    predict_btn.click(
        # Modelo a ser executado
        fn=predict,
        # Entrada dos dados para previsão do modelo
        inputs=[
            year,
            month,
            #flights_booked,
            flights_with_companions,
            total_flights,
            distance,
            points_accumulated,
            salary,
            clv ],
        # Retorno do modelo testado nos dados
        outputs=[resposta],
    )

demo.launch(debug=True, share=True)


# In[ ]:




