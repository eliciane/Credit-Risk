'''

CONFORME FEITO NO BI, AQUI VAMOS DESENVOLVER O RFM

'''
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

# Entrar com a base de dados exportada da etapa anterior '3 Feature-Recencia-Frequencia-De-Atraso.py'
df_RV = pd.read_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/tREC_FREQ_ATR.parquet.parquet', engine='auto')
print(df_RV.info())

# pegando a data mais recente de compra
data_doc_maxima = df_RV['DATA_DOCUMENTO'].max()
print('\n ver a data mais recente de compra no dataset \n', data_doc_maxima)

#COLAR INDICE EM DATA DOCUMENTO PARA SEPARAR OS ANOS E CALCULAR A MÉDIA DE ATRASO POR ANO
df_RV = df_RV.set_index('DATA_DOCUMENTO')
#print(df3[['CONTA']])

#SEPARAR O DATASET A PARTIR DE DATA_DOCUMENTOS DE 2018
df_RV = df_RV.loc['2018-01-01':'2023-04-28']
print(df_RV.shape)

df_RV = df_RV.reset_index()


#Criar o total faturado por conta, valor momentário total, e frequencia de compras
freq_mon = (df_RV
            .groupby(['CONTA_CREDITO'])
            .agg(FREQUENCIA=('REFERENCIA', lambda x: x.count()),
                 VALOR_MONETARIO_TOTAL=('MONTANTE_MOEDA_INTERNA', lambda x: x.sum()))
            .reset_index())

print(freq_mon.head())

#Juntar a coluna de total faturado no dataset de todos cadastros e duplicadas
df_RV = df_RV.merge(freq_mon, how='inner', on=['CONTA_CREDITO'])
print(df_RV.info())

pd.set_option('display.max_columns', None)

#transformar a coluna DATA_DOCUMENTO em data time afim de CALCULAR A RECENCIA
df_RV['DATA_DOCUMENTO']=pd.to_datetime(df_RV['DATA_DOCUMENTO'])
pd.to_datetime(['01/01/2018', '02/01/2018', '03/01/2018'], format='%d/%m/%Y')

#verificar se converteu para data
print(type(df_RV.DATA_DOCUMENTO[0]))

# pegando a data mais recente de compra
data_doc_maxima = df_RV['DATA_DOCUMENTO'].max()
print('\n ver a data mais recente de compra no dataset \n', data_doc_maxima)

recencia_conta =(df_RV.groupby(['CONTA_CREDITO'])
                  .agg(RECENCIA=('DATA_DOCUMENTO', lambda  x:x.max()))
                  .reset_index())

# CALCULAR OS DIAS DE NÃO ATRASO OU RECENCIA DE ATRASO
recencia_conta['RECENCIA'] = (data_doc_maxima - recencia_conta['RECENCIA']).dt.days

print(recencia_conta)

recencia_conta.reset_index()

#Juntar a coluna de RECENCIA
df_RV = df_RV.merge(recencia_conta, how='inner', on=['CONTA_CREDITO']).reset_index()
print(df_RV.info())

#Criar uma coluna de Ticket Médio
df_RV['TICKET_MÉDIO'] = df_RV['VALOR_MONETARIO_TOTAL'] / df_RV['FREQUENCIA']

print(df_RV.info())

##### CALCULAR O TEMPO DO RELACIONAMENTO #######

data_doc_maxima = df_RV['DATA_DOCUMENTO'].max()
print('\n ver a data mais recente de compra no dataset \n', data_doc_maxima)



#AGRUPAR O TEMPO DE RELACIONAMENTO POR CONTA CRÉDITO
tempo_rel = (df_RV.groupby(['CONTA_CREDITO'])
             .agg(DATA_CAD_CC=('DATA_CADASTRO', lambda x: x.min())))

tempo_rel = tempo_rel.reset_index()
#print(tempo_rel_cliente)

tempo_rel['TP_RELAC']= data_doc_maxima - tempo_rel['DATA_CAD_CC']
tempo_rel['TP_RELAC']= tempo_rel['TP_RELAC']/np.timedelta64(1,'Y')


#Juntar com o dataframe original
df2 = df_RV.merge(tempo_rel, how='inner', on=['CONTA_CREDITO']).reset_index(drop=True)

#print(df2.info())


##### CALCULAR O TEMPO DE FUNDAÇÃO #######
#AGRUPAR O TEMPO DE FUNDAÇÃO POR CONTA CRÉDITO
tempo_fund = (df_RV.groupby(['CONTA_CREDITO'])
             .agg(DATA_FUND_CC=('DATA_FUNDACAO', lambda x: x.min())))


tempo_fund = tempo_fund.reset_index()
#print(tempo_rel_cliente)


tempo_fund['TP_FUND']= data_doc_maxima - tempo_fund['DATA_FUND_CC']
tempo_fund['TP_FUND']= tempo_fund['TP_FUND']/np.timedelta64(1,'Y')


#Juntar com o dataframe original
df3 = df2.merge(tempo_fund, how='inner', on=['CONTA_CREDITO']).reset_index(drop=True)
print(df3.info())


df3.to_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/RFM', engine = 'pyarrow', compression = 'gzip')

