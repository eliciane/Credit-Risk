'''

Nesta etapa vamos calcular a Recência e Frequencia de dias de Atraso

'''

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

# Entrar com a base de dados exportada da etapa anterior '2 Feature-Calculo-Dias-De-Atraso.py'
df_dz_BASE = pd.read_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/Duplicatas_Dias_Atraso.parquet', engine='auto')
print(df_dz_BASE.shape)
print(df_dz_BASE.info())


# Tratar a base
# alterar o tipo da variável
df_dz_BASE['TOT_FAT_ULT_12'] = df_dz_BASE['TOT_FAT_ULT_12'].astype('float32')
print(df_dz_BASE['CONTA'].memory_usage(index=False, deep=True))
df_dz_BASE['CONTA'] = df_dz_BASE ['CONTA'].astype('category')
df_dz_BASE['CONTA_CREDITO'] = df_dz_BASE['CONTA_CREDITO'].astype('category')
df_dz_BASE['REGIAO'] = df_dz_BASE['REGIAO'].astype('category')
df_dz_BASE['REFERENCIA'] = df_dz_BASE['REFERENCIA'].astype('category')
df_dz_BASE['EQUIPE_VENDAS'] = df_dz_BASE['EQUIPE_VENDAS'].astype('category')
df_dz_BASE['EQUIPE_DE_VENDAS'] = df_dz_BASE['EQUIPE_DE_VENDAS'].astype('category')
df_dz_BASE['ESCRITORIO_VENDAS'] = df_dz_BASE['ESCRITORIO_VENDAS'].astype('category')
df_dz_BASE['ESCRITORIO_DE_VENDAS'] = df_dz_BASE['ESCRITORIO_DE_VENDAS'].astype('category')
df_dz_BASE['DOC_COMPENSACAO'] = df_dz_BASE['DOC_COMPENSACAO'].astype('category')

# verificar se existem dados nulos
print(df_dz_BASE.isnull().sum())

#criar um dataset com os NANs e verificar a conta crédito que está nan
df_nan_values = df_dz_BASE[df_dz_BASE.isna().any(axis=1)]
df_nan_values = df_nan_values[df_nan_values['CONTA_CREDITO'].isna()]

print(df_nan_values[['CONTA', 'CONTA_CREDITO', 'CHV.SETOR INDUSTRIAL']])

# Substuir o numero da conta crédito pelo numero da conta do cliente (filial)
#df_dz_BASE['CONTA_CREDITO'] = np.where(df_dz_BASE['CONTA']== '1018269', '4007249', df_dz_BASE['CONTA_CREDITO'] )
print(df_dz_BASE[['CONTA', 'CONTA_CREDITO']].tail(50))

#ENCONTRAR A DATA MAXIMA DO DOCUMENTO PARA SABER A DATA MAIS RECENTE DO DATASET
df_data_max = df_dz_BASE['DATA_DOCUMENTO'].max()

# Criar uma coluna com os dias totais do dataset
df_dz_BASE['dias_dataset'] = (df_data_max - df_dz_BASE['DATA_DOCUMENTO'].min())
df_dz_BASE['dias_dataset'] = df_dz_BASE['dias_dataset'].dt.days
print(df_dz_BASE.info())

print(df_dz_BASE[['dias_dataset']])
print('\n ver a data mais recente de compra no dataset \n', df_data_max)

#criar uma coluna de valores positivos e SUBSTITUIR VALORES NEGATIVOS POR ZERO PARA CALCULAR A MÉDIA e RECENCIA DE ATRASO
df_dz_BASE['pos_val_DZ'] = df_dz_BASE['DIAS_ATRASO_COMP_DZ']
df_dz_BASE['pos_val_DZ'].loc[df_dz_BASE['pos_val_DZ']<0] = 0
print(df_dz_BASE.info())

#CRIAR UMA COLUNA PARA CALCULAR NUMERO DE DUPLICATAS ATRASADAS POR TRASAÇÃO
df_dz_BASE['FREQ_ATRASO_DZ_DIA'] = np.where(df_dz_BASE['pos_val_DZ']>0, 1, 0)

#número de transações ou boletos gerados
df_dz_BASE['N_TRANS_DZ'] = np.where(df_dz_BASE['TIPO_DOCUMENTO'] == 'DZ', 1, 0)



####### FAZER UM FILTRO PARA PEGAR SOMENTE OS DIAS DE ATRASO, que são os dias positivos ###########
df_filtro = df_dz_BASE[(df_dz_BASE['pos_val_DZ'] >= 1)]
print(df_filtro.info())

#CLASSIFICAR POR DATA DE VENCIMENTO
df_filtro = df_filtro.sort_values(by=['VENCIMENTO_LIQUIDO'])



##############CALCULAR OS DIAS (RECENCIA DE ATRASO) DE ATRASO POR CONTA CRÉDITO #######
# AGRUPAR A DATA DO ULTIMO ATRASO POR CONTA CRÉDITO
#data_natraso_max = df_filtro.groupby(['CONTA_CREDITO'], as_index = False)['VENCIMENTO_LIQUIDO'].max()
data_natraso_max = (df_filtro
                    .groupby(['CONTA_CREDITO'])
                    .agg(DIAS_NAO_ATRASO_DZ=('VENCIMENTO_LIQUIDO', lambda  x:x.max()))
                    .reset_index())

print(data_natraso_max)
print(data_natraso_max.shape)


# CALCULAR OS DIAS DE NÃO ATRASO OU RECENCIA DE ATRASO
data_natraso_max ['DIAS_NAO_ATRASO_DZ'] = (df_data_max - data_natraso_max['DIAS_NAO_ATRASO_DZ']).dt.days
print(data_natraso_max.head())

print(df_dz_BASE.info())


########FAZER UM MERGE COM TODA A BASE#############
df3 = pd.merge(df_dz_BASE, data_natraso_max, on=['CONTA_CREDITO'], how='outer')
print(df3.info())


#PESQUISAR SOMENTE UM CLIENTE
df3_cli = df3.loc[(df3['CONTA_CREDITO'] == '1000183')]
print(df3[['CONTA', 'CONTA_CREDITO', 'DIAS_ATRASO_COMP_DZ', 'pos_val_DZ', 'DIAS_NAO_ATRASO_DZ', 'VENCIMENTO_LIQUIDO']].head(50))

print(df3.isnull().sum())


#criar um dataset com os NANs e veriFIicar a conta crédito que está nan, QUE SÃO AQUELES QUE NAO TEM DIA DE ATRASO
df_nan_values2 = df3[df3.isna().any(axis=1)]
df_nan_values2 = df_nan_values2[df_nan_values2['DIAS_NAO_ATRASO_DZ'].isna()]
print(df_nan_values2[['CONTA', 'NOME_CLIENTE', 'CONTA_CREDITO', 'DIAS_ATRASO_COMP_DZ',  'pos_val_DZ', 'DIAS_NAO_ATRASO_DZ']])


#COMPLETAR OS DIAS DE NÃO ATRASOS DOS NANs COM O MAXIMO DE DIAS DO DATASET, POIS ESTES NÃO TIVERAM NENHUM ATRASO
df3['DIAS_NAO_ATRASO_DZ'] = df3['DIAS_NAO_ATRASO_DZ'].fillna(df3['dias_dataset'])
print(df3[['CONTA', 'CONTA_CREDITO', 'DIAS_ATRASO_COMP_DZ',  'pos_val_DZ', 'DIAS_NAO_ATRASO_DZ']])
print(df3.shape)



############## NOVAMENTE FILTRAR O DATASET COMPLETO, ATUALIZADO NO MERGE ACIMA ################

df3_filtro = df3[(df3['pos_val_DZ'] >= 1)]
print(df3_filtro.info())

#ENCONTAR A FREQUENCIA DE ATRASO POR CONTA CRÉDITO
freq_atr = (df3_filtro
                    .groupby(['CONTA_CREDITO'])
                    .agg(TT_BOL_ATR_DZ_CC=('pos_val_DZ', lambda  x:x.count()))
                    .reset_index())

print(freq_atr)
print(freq_atr.info())


#fazer um merge para colocar as novas colunas no dataset completo
df5 = pd.merge(df3, freq_atr, on=['CONTA_CREDITO'], how='outer')
print(df5[['CONTA', 'CONTA_CREDITO', 'DIAS_ATRASO_COMP_DZ', 'pos_val_DZ', 'DIAS_NAO_ATRASO_DZ', 'TT_BOL_ATR_DZ_CC']])
print(df5.info())
print(df5.shape)

#COMPLETAR AS CONTAS QUE FICARAM COM NANs NA FREQUENCIA DE ATRASO COM ZERO, POIS ESTES NÃO TIVERAM NENHUM ATRASO
df5['TT_BOL_ATR_DZ_CC'] = df5['TT_BOL_ATR_DZ_CC'].fillna(0)

print(df5[['CONTA', 'CONTA_CREDITO', 'DIAS_ATRASO_COMP_DZ', 'pos_val_DZ', 'DIAS_NAO_ATRASO_DZ', 'TT_BOL_ATR_DZ_CC']])

# Alterar o tipo da variável
df5['DIAS_NAO_ATRASO_DZ'] = df5['DIAS_NAO_ATRASO_DZ'].astype('uint16')
df5['TT_BOL_ATR_DZ_CC'] = df5['TT_BOL_ATR_DZ_CC'].astype('uint16')
df5['pos_val_DZ'] = df5['pos_val_DZ'].astype('uint16')
df5['DIAS_ATRASO_COMP_DZ'] = df5['DIAS_ATRASO_COMP_DZ'].astype('int16')

print(df5.info())

df5.to_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/tREC_FREQ_ATR.parquet', engine = 'pyarrow', compression = 'gzip')



