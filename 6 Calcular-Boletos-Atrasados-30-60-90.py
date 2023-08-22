'''

Aqui vamos usar o mesmo raciocício do scritpt anterior, de calculo de médias dos ultimos de dias.
Porém, ao invés de calcular a média de dias de atraso, vamos calcular a soma de boletos atrasados.

objetivo: calcular a média de de boletos atrasados nos ultimos 30 dias de transação de cada cliente, idependente se ele é um cliente atual ou não atual.
Ou seja, é feito o calculo da média dos ultimos trinta dias que o cliente teve transção de pagamentos na OF

ATENÇÃO: TBM COM ESTE SCRIPT VAMOS CRIAR AS FEATURES DE MÉDIA DOS ULTIMOS 60 E DOS ULTIMOS 90 DIAS DE ATRASO. ISTO SERÁ FEITO RODANDO O SCRIPT 3 VEZES E EM CADA RODADA
ATUALIZAR O SCRIPT NAS LINHAS (PRÓXIMAS) 57, 84, onde se referem ao periodo desejado e, também alterar o script na linha 108(para exportar o arquivo correto)



'''
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from datetime import date
from io import StringIO
import sys
from datetime import datetime
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)



df = pd.read_parquet('C:/Users/eliciane.silva/downloads/MEDIA30DIAS_3ago.parquet', engine = 'auto')
#print(df.shape)
print(df.info())
#print(df)

#PESQUISAR UM CLIENTE
df_cli = df[(df['CONTA_CREDITO'] == '1007062')]
df_cli = df_cli[['CONTA_CREDITO', 'VENCIMENTO_LIQUIDO', 'D_ATR', 'BOL_ATR_DIA', 'N_TRANS', 'DATA_ULT_30D', 'DATA_MAX_VENC']]
#print(df_cli.to_string())


df = df.sort_values(by=['CONTA_CREDITO'])

# AGRUPAR O DATAFRAME POR CLIENTE
grouped_df = df.groupby('CONTA_CREDITO')




# PARA CADA GRUPO CALCULAR A DATA MAXIMA E MINIMA
# PARA CADA GRUPO CALCULAR A DATA MAXIMA
max_due_dates = grouped_df['VENCIMENTO_LIQUIDO'].max()
#print(max_due_dates)

# CALCULAR A PRIMEIRA DATA COMO SENDO 30 DIAS ANTES DA DATA MÁXIMA
first_dates = max_due_dates - pd.Timedelta(days=60) #aqui to time delta deve ser o numero de dias (periodo) que se deseja fazer o cáculo dos boletos atrasados
#print(first_dates)


# FILTAR O DATAFRAME ORIGINAL PARA INCLUIR SOMENTE AS LINHAS PARA CADA CLIENTE ENTRE A PRIMEIRA DATA E A DATA MÁXIMA
# filter the original dataframe to include only the rows for each client_id between the first date and the maximum due date
filtered_dfs = []
for client_id, first_date in first_dates.items():
    max_due_date = max_due_dates[client_id]
    filtered_df = df[(df['CONTA_CREDITO'] == client_id) & (df['VENCIMENTO_LIQUIDO'] >= first_date) & (df['VENCIMENTO_LIQUIDO'] <= max_due_date)]
    filtered_dfs.append(filtered_df)

#print(len(filtered_dfs))
#print(filtered_dfs)
#filtered_dfs = pd.DataFrame(filtered_dfs)

#print(filtered_dfs)


# calculate the mean of late_day column for each client_id
sum_late_bol = []
for filtered_df in filtered_dfs:
    sum_late_bol.append(filtered_df['BOL_ATR_DIA'].sum())

#print(mean_late_days)

df3 = pd.DataFrame(sum_late_bol)
df3= df3.rename(columns={0: 'SOMA_60_BOL_ATR'})
print(df3)

df4 = df['CONTA_CREDITO'].unique()
print(df4)

df5 = pd.DataFrame(df4)
df5= df5.rename(columns={0: 'CONTA_CREDITO'})
print(df5)

df6 = pd.concat([df5, df3],axis=1)
# create a new column in the original dataframe with these mean values using the groupby() and transform() methods
#df['MEDIA_ATR_30D'] = df.groupby('CONTA_CREDITO')['D_ATR'].transform('mean')

#print(df)
#print(df)

#print(df6)

df_cli = df6[(df6['CONTA_CREDITO'] == '1007062')]
print(df_cli.to_string())



df6.to_parquet('C:/Users/eliciane.silva/downloads/CLIENTES_SOMA_BOL_60DIAS_3ago.parquet', engine = 'pyarrow', compression = 'gzip')



