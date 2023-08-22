'''

objetivos:

JUNTAR OS DATASETs DAS FEATURES médias dias de atraso (30,60 e 90 dias), média dos números de boletos (30,60 e 90 dias) e
média dos numeros de transações (30,60 e 90 dias).

ATENCÃO: SE OS SCRIPTs das etapas MOD 3 - 3_3 (MEDIAS dias e SOMAs de numero de boleto e transações) NÃO TIVEREM SIDO RODADAS 3 VEZES CADA UMA,
VC NÃO TERÁ TODAS AS FEATURES CRIADAS PARA JUNTAR OS DATASETs neste SCRIPT !!!!!!!!

TAMBÉM, VAMOS CLASSIFICAR O CLIENTE EM BOM E MAU pagador (0 ou 1), criando a coluna do TARGET (entre as linhas 57 e 73 deste script)

'''


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)


df30 = pd.read_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/CLIENTES_MEDIA30DIAS.parquet', engine='auto')
print('\n tamanho do dataset com feature 30 dias:', df30.shape)


df30_cli = df30.loc[(df30['CONTA_CREDITO'] == '1007062')]
#print(df30_cli)



df60 = pd.read_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/CLIENTES_MEDIA60DIAS.parquet', engine='auto')
print('\n tamanho do dataset com feature 60 dias:', df60.shape)

df90 = pd.read_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/CLIENTES_MEDIA90DIAS.parquet', engine='auto')
print('\n tamanho do dataset com feature 90 dias:', df90.shape)

df1 = pd.read_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/MEDIA30DIAS_3ago.parquet', engine ='auto')
print('\n tamanho do dataset PRINCIPAL:', df1.shape)

#Juntar o dataset de todos cadastros e duplicadas
df2 = df30.merge(df60, how='inner', on=['CONTA_CREDITO'])
print('\n tamanho do dataset MERGE df30 e 60:', df2.shape)

df2 = df2.merge(df90, how='inner', on=['CONTA_CREDITO'])
print('\n tamanho do dataset MERGE df30, 60 e 90:', df2.shape)

df2 = df2.merge(df1, how='inner', on=['CONTA_CREDITO'])
print('\n tamanho do dataset MERGE df30, 60, 90 e PRINCIPAL:', df2.shape)
print(df2.info())



df2['MEDIA_30_DIA_ATR'] = df2['MEDIA_30_DIA_ATR'].round(decimals=0)
df_cliente = df2.loc[(df2['CONTA_CREDITO'] == '1000651')]
#print(df_cliente)

#COLOCANDO O TARGET COMO 0 OU 1
condições = [
    (df2['MEDIA_30_DIA_ATR'] < 1),
    (df2['MEDIA_30_DIA_ATR'] >= 1 ),
  ]

rating = ['0', '1']

#CRIAR A COLUNA DE RATING
df2['TARGET'] = np.select(condições,rating)

df2['TARGET'] = np.select([
    (df2['MEDIA_30_DIA_ATR'] < 1),
    (df2['MEDIA_30_DIA_ATR'] >= 1),
    ], ['0', '1']
)



df30_bol = pd.read_parquet('C:/Users/eliciane.silva/downloads/CLIENTES_SOMA_BOL_30DIAS_3ago.parquet', engine='auto')
print('\n tamanho do dataset com feature 30 dias:', df30_bol.shape)

df60_bol = pd.read_parquet('C:/Users/eliciane.silva/downloads/CLIENTES_SOMA_BOL_60DIAS_3ago.parquet', engine='auto')
print('\n tamanho do dataset com feature 60 dias:', df60_bol.shape)

df90_bol = pd.read_parquet('C:/Users/eliciane.silva/downloads/CLIENTES_SOMA_BOL_90DIAS_3ago.parquet', engine='auto')
print('\n tamanho do dataset com feature 90 dias:', df90_bol.shape)



#Juntar o dataset de todos cadastros e duplicadas
df2 = df2.merge(df30_bol, how='inner', on=['CONTA_CREDITO'])
print('\n tamanho do dataset MERGE df30 e 60:', df2.shape)

df2 = df2.merge(df60_bol, how='inner', on=['CONTA_CREDITO'])
print('\n tamanho do dataset MERGE df30, 60 e 90:', df2.shape)

df2 = df2.merge(df90_bol, how='inner', on=['CONTA_CREDITO'])
print('\n tamanho do dataset MERGE df30, 60, 90 e PRINCIPAL:', df2.shape)
print(df2.info())



df30_trans = pd.read_parquet('C:/Users/eliciane.silva/downloads/CLIENTES_SOMA_TRANS_30DIAS_3ago.parquet', engine='auto')
print('\n tamanho do dataset com feature 30 dias:', df30_trans.shape)

df60_trans = pd.read_parquet('C:/Users/eliciane.silva/downloads/CLIENTES_SOMA_TRANS_60DIAS_3ago.parquet', engine='auto')
print('\n tamanho do dataset com feature 60 dias:', df60_trans.shape)

df90_trans = pd.read_parquet('C:/Users/eliciane.silva/downloads/CLIENTES_SOMA_TRANS_90DIAS_3ago.parquet', engine='auto')
print('\n tamanho do dataset com feature 90 dias:', df90_trans.shape)



#Juntar o dataset de todos cadastros e duplicadas
df2 = df2.merge(df30_trans, how='inner', on=['CONTA_CREDITO'])
print('\n tamanho do dataset MERGE df30 e 60:', df2.shape)

df2 = df2.merge(df60_trans, how='inner', on=['CONTA_CREDITO'])
print('\n tamanho do dataset MERGE df30, 60 e 90:', df2.shape)

df2 = df2.merge(df90_trans, how='inner', on=['CONTA_CREDITO'])
print('\n tamanho do dataset MERGE df30, 60, 90 e PRINCIPAL:', df2.shape)
print(df2.info())

df2.to_parquet('C:/Users/eliciane.silva/downloads/DatasetMOD3_3ago.parquet', engine = 'pyarrow', compression = 'gzip')

