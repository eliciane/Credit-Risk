'''

Objetivo: Cálculo das médias de dias de atraso 30, 60 e 90 dias de atraso


'''



import pandas as pd

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)



# Entrar com a base de dados exportada da etapa anterior '4 DesenvolverRFM.py'
df = pd.read_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/RFM.parquet', engine = 'auto')
#print(df.shape)
print(df.info())
#print(df)

df = df.sort_values(by=['CONTA_CREDITO'])


# AGRUPAR O DATAFRAME POR CLIENTE
grouped_df = df.groupby('CONTA_CREDITO')

# PARA CADA GRUPO CALCULAR A DATA MAXIMA E MINIMA
# PARA CADA GRUPO CALCULAR A DATA MAXIMA
max_due_dates = grouped_df['VENCIMENTO_LIQUIDO'].max()
#print(max_due_dates)

# CALCULAR A PRIMEIRA DATA COMO SENDO 30 DIAS ANTES DA DATA MÁXIMA
first_dates = max_due_dates - pd.Timedelta(days=60)
#print(first_dates)


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
mean_late_days = []
for filtered_df in filtered_dfs:
    mean_late_days.append(filtered_df['D_ATR'].mean())

#print(mean_late_days)

df3 = pd.DataFrame(mean_late_days)
df3= df3.rename(columns={0: 'MEDIA_60_DIA_ATR'})
print(df3)

df4 = df['CONTA_CREDITO'].unique()
print(df4)

df5 = pd.DataFrame(df4)
df5= df5.rename(columns={0: 'CONTA_CREDITO'})
print(df5)

df6 = pd.concat([df5, df3],axis=1)



df6.to_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/CLIENTES_MEDIA60DIAS.parquet', engine = 'pyarrow', compression = 'gzip')




