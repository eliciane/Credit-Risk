'''

Aqui iremos calcular os dias pagos em atraso por cliente

'''


#importar bibliotecas
import gzip
import pandas as pd
import numpy as np
import warnings
from pandas.compat import pyarrow
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)


# Entrar com a base de dados Tratada conforme realizado na Etapa 1 'Tratar Base'
df = pd.read_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/BasesDuplicatasTratadas.parquet', engine='auto')
print(df.info())

# pegando a data mais recente de compra
data_doc_maxima = df['DATA_DOCUMENTO'].max()
print('\n ver a data mais recente de compra no dataset \n', data_doc_maxima)

#Substituir os campos em brandos em Nan para que possam aparecer
missing_values=['nan']
df = df.replace(missing_values, np.NaN)
#excluir DOC DE COMPENSAÇÃO NULOS, pois precisamos encontrar os créditos e débitos dos mesemos numero de doc.compensação
df = df.dropna(subset = ['DOC_COMPENSACAO'])

#Filtrar por tipo de documento, documentos faturados
df_RV = df.loc[(df['TIPO_DOCUMENTO'] == 'RV')] # Documento de Faturamento
print('\n tamanho do dataset de RVs: \n', df_RV.shape)

#Filtrar por tipo de documento, pagamento feito em depósito bancário
df_DZ = df.loc[(df['TIPO_DOCUMENTO'] == 'DZ')] # Liquidação manual (depósito)
print('\n tamanho do dataset de DZs: \n',df_DZ.shape)

#############  ENCONTRAR AS DUPLICATAS DE FATURAMENTO E PAGAMENTO (PARES DÉBITO/CRÉDITO)   ##################

#concatenar somente os df com os tipos de documentos Faturamento (RV) e Pagamento (ZP) para calculo dos dias de atraso
df_DZ_RV = pd.concat([df_DZ, df_RV], axis=0)
print('\n tamanho do dataframe juntando DZ e RV: \n', df_DZ_RV.shape)
#print(df_DZ_RV.info())

#classificar por tipo de documento para aparecer os RVs primeiro (ordem alfabética)
df_DZ_RV = df_DZ_RV.sort_values(by=['TIPO_DOCUMENTO'], ignore_index=True)

#mudar os valores da coluna montante documento para valor absoluto
df_DZ_RV = df_DZ_RV.copy()
df_DZ_RV['MONTANTE_MOEDA_INTERNA'] = df_DZ_RV['MONTANTE_MOEDA_INTERNA'].abs()

#CRIAR um dataframe e somar o montante moeda interna por DOC COMPENSACAO e TIPO DOCUMENTO para poder manter somente os pares de RVs e DZs
df_DZ_RV_somaMMI = df_DZ_RV.groupby(['DOC_COMPENSACAO', 'TIPO_DOCUMENTO'])[['MONTANTE_MOEDA_INTERNA']].sum()

df_DZ_RV_somaMMI = df_DZ_RV_somaMMI.reset_index()
print('\n tamanho do dataset simplicado, com a soma de MMI por conta: \n', df_DZ_RV_somaMMI.shape)

# Fazer uma consulta aletória
df_DZ_RV_doc_7 = df_DZ_RV_somaMMI.loc[(df_DZ_RV_somaMMI['DOC_COMPENSACAO'] == '1400011732')]
print(df_DZ_RV_doc_7[['MONTANTE_MOEDA_INTERNA', 'TIPO_DOCUMENTO', 'DOC_COMPENSACAO']].tail(50))

#Juntar a coluna de montante moeda interna com soma total no dataset_original
df_DZ_RV_2 = df_DZ_RV.merge(df_DZ_RV_somaMMI, on=['DOC_COMPENSACAO', 'TIPO_DOCUMENTO'])
print(df_DZ_RV_2)
print(df_DZ_RV_2.info())
print('\n tamanho do dataset simplicado, com a soma de MMI por conta: \n', df_DZ_RV_2.shape)

#classificar por tipo de documento para aparecer os RVs primeiro (ordem alfabética)
df_DZ_RV_2 = df_DZ_RV_2.sort_values(by=['DOC_COMPENSACAO', 'TIPO_DOCUMENTO'], ascending=False, ignore_index=True)

#MARTER SOMENTE OS DUPLICADOS EM DOCUMENTOS DE COMPENSACAO E TIPOS DE DOCUMENTOS
df_DZ_RV_3 = df_DZ_RV_2[df_DZ_RV_2.duplicated(subset=['TIPO_DOCUMENTO'])]

print('\n tamanho do dataset simplicado, com a soma de MMI: \n', df_DZ_RV_3.shape)

#MARTER SOMENTE OS DUPLICADOS EM DOCUMENTOS DE COMPENSACAO E TIPOS DE DOCUMENTOS
df_DZ_RV_4 = df_DZ_RV_3[~df_DZ_RV_3.duplicated(subset=['DOC_COMPENSACAO', 'TIPO_DOCUMENTO', 'VENCIMENTO_LIQUIDO'])]
print('\n tamanho do dataset simplicado, com a soma de MMI por conta: \n', df_DZ_RV_4.shape)

#EXCLUIR OS RVs que NÃO são pares com DZs e que estão duplicados pelo DC_COMPENSAÇÃO E MONTANTE MOEDA INTERNA
df_DZ_RV_2_RVs = df_DZ_RV_4.loc[(df_DZ_RV_4['TIPO_DOCUMENTO'] == 'RV')]
df_DZ_RV_2_RVs = df_DZ_RV_2_RVs.sort_values(by=['VENCIMENTO_LIQUIDO'], ignore_index=True)
df_DZ_RV_2_RVs = df_DZ_RV_2_RVs.drop_duplicates(subset=['DOC_COMPENSACAO'], keep='first')
df_DZ_RV_2_RVs_doc = df_DZ_RV_2_RVs.loc[(df_DZ_RV_2_RVs['DOC_COMPENSACAO'] == '1400018100')]
print('\n tamanho do dataset de RVs: \n', df_DZ_RV_2_RVs.shape)

df_DZ_RV_2_DZs = df_DZ_RV_4.loc[(df_DZ_RV_4['TIPO_DOCUMENTO'] == 'DZ')]

df_DZ_RV_2_DZs['VENCIMENTO_LIQUIDO'] = df_DZ_RV_2_DZs['VENCIMENTO_LIQUIDO'].replace('2023-11-17', '2022-11-17')

df_DZ_RV_2_DZs = df_DZ_RV_2_DZs.sort_values(by=['VENCIMENTO_LIQUIDO'], ignore_index=True)
df_DZ_RV_2_DZs = df_DZ_RV_2_DZs.drop_duplicates(subset=['DOC_COMPENSACAO'], keep='last')
print('\n tamanho do dataset de DZs: \n', df_DZ_RV_2_DZs.shape)

df_DZ_RV_5 = pd.concat([df_DZ_RV_2_RVs, df_DZ_RV_2_DZs], axis=0)
print('\n tamanho do dataset juntando dos RVs e DZs novamente:\n', df_DZ_RV_5.shape)

df_DZ_RV_keep_dup = df_DZ_RV_5.loc[df_DZ_RV_5.duplicated(subset=['DOC_COMPENSACAO'], keep=False)]
df_DZ_RV_6 = df_DZ_RV_keep_dup.copy()
print('\n  tamanho do dataset com os pares DZ e RV com o mesmo numero de doc compensação e valor SOMA MMI:\n',df_DZ_RV_keep_dup.shape)

df_DZ_RV_6 = df_DZ_RV_6.sort_values(by=['DOC_COMPENSACAO', 'TIPO_DOCUMENTO'], ascending=False, ignore_index=True)

df_DZ_RV_2_DZsS = df_DZ_RV_6.loc[(df_DZ_RV_6['TIPO_DOCUMENTO'] == 'DZ')]
print(df_DZ_RV_2_DZsS.shape)
df_DZ_RV_2_RVsS = df_DZ_RV_6.loc[(df_DZ_RV_6['TIPO_DOCUMENTO'] == 'RV')]
print(df_DZ_RV_2_RVsS.shape)


#######  FAZER O CALCULO DOS DIAS DE ATRASO ######################

#fazer o calculo dos dias de atraso por numero do documento de compensação
df_DZ_RV_6['DIAS_ATRASO_COMP_DZ'] = df_DZ_RV_6["VENCIMENTO_LIQUIDO"].diff().apply(lambda x: x/np.timedelta64(1, 'D')).fillna(0).astype('int64')
df_DZ_RV_6.loc[df_DZ_RV_6['TIPO_DOCUMENTO'] == 'RV', 'DIAS_ATRASO_COMP_DZ'] = np.nan
df_DZ_RV_6['DIAS_ATRASO_COMP_DZ'] = df_DZ_RV_6['DIAS_ATRASO_COMP_DZ'].fillna(0)

#passar as colunas CONTA E DIAS_ATRASO_COMP PARA INTEIRO
df_DZ_RV_6['CONTA'] = df_DZ_RV_6['CONTA'].astype(str)
df_DZ_RV_6['DIAS_ATRASO_COMP_DZ'] = df_DZ_RV_6['DIAS_ATRASO_COMP_DZ'].astype(int)

df_DZ_RV_6.to_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/Duplicatas_Dias_Atraso.parquet', engine = 'pyarrow', compression = 'gzip')

