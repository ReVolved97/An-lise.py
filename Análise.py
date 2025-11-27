# ---> 1. IMPORTACAO DAS BIBLIOTECAS 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---> 2. CONFIGURACOES INICIAIS DE VISUALIZACAO 
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# ---> 3. CARREGAMENTO DOS DADOS 
try:
    df_pedidos = pd.read_csv('olist_orders_dataset.csv')
    df_clientes = pd.read_csv('olist_customers_dataset.csv')
    df_itens_pedido = pd.read_csv('olist_order_items_dataset.csv')
    df_produtos = pd.read_csv('olist_products_dataset.csv')
    df_pagamentos = pd.read_csv('olist_order_payments_dataset.csv')
    df_avaliacoes = pd.read_csv('olist_order_reviews_dataset.csv')
    print("OK! Todas as bases de dados foram carregadas com sucesso!")
except FileNotFoundError as e:
    print(f"ERRO: Erro ao carregar um arquivo: {e}. Certifique-se de que os arquivos estao no diretorio correto.")
    exit() 
# ---> 4. MESCLAGEM E PREPARACAO DOS DADOS 
print("\n--- INSPECAO RAPIDA: Tabela de pedidos ---") 
print(df_pedidos.head(3)) 
print("\n--- Informacoes sobre as colunas (Tipos e Nulos) ---")
df_pedidos.info(verbose=False, memory_usage='deep') 

# Juncao das tabelas
df_analise = pd.merge(df_pedidos, df_clientes, on='customer_id', how='inner')
df_analise = pd.merge(df_analise, df_itens_pedido, on='order_id', how='inner')
df_analise = pd.merge(
    df_analise,df_produtos[['product_id', 'product_category_name']], on='product_id', how='left')
df_analise = pd.merge(
    df_analise, 
    df_avaliacoes[['order_id', 'review_score']], 
    on='order_id', 
    how='left')
print("\n--- TABELA DE ANALISE MESCLADA (amostra) ---") 
print(df_analise.sample(5)) 
print(f"\nTABELA FINAL tem {df_analise.shape[0]} linhas e {df_analise.shape[1]} colunas.")

# ---> 5. ENGENHARIA DE FEATURES (COLUNAS DE DATA) 
df_analise['data_compra'] = pd.to_datetime(df_analise['order_purchase_timestamp']) 
df_analise['ano_compra'] = df_analise['data_compra'].dt.year
df_analise['mes_ano_compra'] = df_analise['data_compra'].dt.to_period('M')
print("\n--- OK! Colunas de Tempo Adicionadas! ---")
print(df_analise[['order_id', 'data_compra', 'ano_compra', 'review_score', 'product_category_name']].head())

# ---> 6. ANALISE: TOP 15 CATEGORias MAIS BEM AVALIADAS 
print("\n--- ANALISE: Media de Avaliacoes por Categoria ---")
media_avaliacoes = df_analise.groupby('product_category_name')['review_score'].mean().reset_index()
media_avaliacoes = media_avaliacoes.rename(columns={'review_score': 'media_avaliacao'})
media_avaliacoes = media_avaliacoes.dropna(subset=['product_category_name'])

# Formatando os nomes das categorias
media_avaliacoes['product_category_name'] = media_avaliacoes['product_category_name'].str.replace('_', ' ')
media_avaliacoes['product_category_name'] = media_avaliacoes['product_category_name'].str.title()
media_avaliacoes = media_avaliacoes.sort_values(by='media_avaliacao', ascending=False)
top_15_avaliadas = media_avaliacoes.head(15)
print("\n--- Top 15 Categorias Mais Bem Avaliadas (Dataframe) ---")
print(top_15_avaliadas)

# Grafico 1: Avaliacoes
plt.figure(figsize=(12, 8))
sns.barplot(
    data=top_15_avaliadas, 
    x='media_avaliacao', 
    y='product_category_name',
    palette='Greens_r')
plt.title('Top 15 Categorias Mais Bem Avaliadas', fontsize=16)
plt.xlabel('Media da Pontuacao (1 a 5)', fontsize=12)
plt.ylabel('Categoria do Produto (em Portugues)', fontsize=12)
plt.xlim(0, 5) 
plt.tight_layout()

# ---> 7. ANALISE: EVOLUCAO DAS VENDAS (SERIE TEMPORAL) 
print("\n--- ANALISE: Evolucao das Vendas por Mes ---")
vendas_por_mes = df_analise.groupby('mes_ano_compra')['price'].sum().reset_index()
vendas_por_mes = vendas_por_mes.rename(columns={'price': 'total_vendas'})
vendas_por_mes['mes_ano_compra'] = vendas_por_mes['mes_ano_compra'].astype(str)
print(vendas_por_mes.tail()) 

# Grafico 2: Vendas por Mes
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=vendas_por_mes, 
    x='mes_ano_compra', 
    y='total_vendas',
    marker='o')
plt.title('Evolucao Total das Vendas (R$) por Mes', fontsize=16)
plt.xlabel('Mes/Ano', fontsize=12)
plt.ylabel('Total Vendido (R$)', fontsize=12)
plt.xticks(rotation=45) 
plt.tight_layout()

# ---> 8. ANALISE: TOP 10 ESTADOS POR VALOR DE VENDA 
print("\n--- ANALISE: Top 10 Estados por Valor de Venda ---")
vendas_por_estado = df_analise.groupby('customer_state')['price'].sum().reset_index()
vendas_por_estado = vendas_por_estado.rename(columns={'price': 'total_vendas_R$'})
vendas_por_estado = vendas_por_estado.sort_values(by='total_vendas_R$', ascending=False)
top_10_estados = vendas_por_estado.head(10)

print(top_10_estados)

# Grafico 3: Vendas por Estado
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_10_estados, 
    x='total_vendas_R$', 
    y='customer_state',
    palette='Blues_r')
plt.title('Top 10 Estados por Valor Total de Vendas (R$)', fontsize=16)
plt.xlabel('Total Vendido (R$)', fontsize=12)
plt.ylabel('Estado (UF)', fontsize=12)
plt.tight_layout()

# ---> 9. ANALISE: VALOR TOTAL POR TIPO DE PAGAMENTO 
print("\n--- ANALISE: Valor Total por Tipo de Pagamento ---")

pagamentos_total = df_pagamentos.groupby('payment_type')['payment_value'].sum().reset_index()
pagamentos_total = pagamentos_total.rename(columns={'payment_value': 'total_pago_R$'})
pagamentos_total = pagamentos_total.sort_values(by='total_pago_R$', ascending=False)
print(pagamentos_total)

# Grafico 4: Tipos de Pagamento
plt.figure(figsize=(10, 6))
sns.barplot(
    data=pagamentos_total,
    x='total_pago_R$',
    y='payment_type',
    palette='Oranges_r')
plt.title('Valor Total (R$) por Tipo de Pagamento', fontsize=16)
plt.xlabel('Total Pago (R$)', fontsize=12)
plt.ylabel('Tipo de Pagamento', fontsize=12)
plt.tight_layout()

# ---> 10. ANALISE: PRECO DO PRODUTO VS. VALOR DO FRETE 
print("\n--- ANALISE: Preco vs. Frete (Amostra de 2000 pontos) ---")
df_amostra = df_analise.sample(2000, random_state=42)

# Grafico 5: Preco vs. Frete (Dispersao)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_amostra,
    x='price',
    y='freight_value')
plt.title('Preco do Produto vs. Valor do Frete (Amostra)', fontsize=16)
plt.xlabel('Preco do Produto (R$)', fontsize=12)
plt.ylabel('Valor do Frete (R$)', fontsize=12)
plt.xlim(0, 1000)
plt.ylim(0, 200)
plt.tight_layout()

# ---> 11. EXIBICAO DE TODOS OS GRAFICOS 
plt.show()