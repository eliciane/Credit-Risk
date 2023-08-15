
'''
Este código em limpar e tratar dados de duplicatas de clientes
'''


# Importar Bibliotecas
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings("ignore")

# imprimir maximo colunas, linhas e 4 digitos para casas decimais para float
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.expand_frame_repr', False)

### Puxar BASE DE DUPLICATAS
#df1 = pd.read_excel("C:/Users/eliciane.silva/OneDrive - Ourofino Saude Animal/Score crédito/AnimaisProdução/FBL5N_JUN_2023_EM_ABERTO.xlsx")
df1 = pd.read_parquet("C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/Bases_Duplicatas.parquet", engine = 'auto')
print(df1.info())
print(df1.shape)

# excluir duplicados
df1 = df1.drop_duplicates()

print('\n tamanho do dataset depois que excluiu os duplicados:', df1.shape)


#ler as colunas para excluir aquelas que não iremos usar
print(df1.columns)

# Excluir as colunas que não forem usadas
df1 = df1.drop(['Loc.negócios', 'Símb.prtds.em aberto/comp', 'Atribuição', 'Nº documento', 'Dias 1', 'Símbolo de vencimento líquido', 'Atraso após vencimento líquido',
       'Cód.Razão Especial', 'Texto cabeçalho documento', 'FrmPgto', 'Bloqueio pgto.', 'Motivo Bloqueio', 'Banco da empresa', 'Chave referência 3', 'Texto', 'Montante pagamento',
       'Data de lançamento', 'Nome do usuário', 'Data de pagamento', 'Data de compensação', 'Exercício', 'Solicitação de L/C'], axis=1)

print(df1.info())


# verificar dados nulos e excluir linhas em branco no campo CONTA
print(df1.isnull().sum())

#Substituir alguns campos que são nulos em Nan o para que possam aparecer
missing_values=['nan']
df1 = df1.replace(missing_values, np.NaN)

missing_values=['None']
df1 = df1.replace(missing_values, np.NaN)

#excluindo dados nulos onde aparecem o numero de contas nulos
df1 = df1.dropna(subset=['Conta'])
print('\n imprimir quantidade de dados nulos:', df1.isnull().sum())

print(df1.columns)

# RENOMEAR COLUNAS
df1 = df1.rename(columns={'Empresa': 'EMPRESA', 'Conta':'CONTA', 'Nome Cliente': 'NOME_CLIENTE', 'Referência': 'REFERENCIA',
                    'Item': 'ITEM', 'Tipo de documento': 'TIPO_DOCUMENTO ', 'Data do documento': 'DATA_DOCUMENTO',
                    'Vencimento líquido': 'VENCIMENTO_LIQUIDO', 'Montante em moeda interna': 'MONTANTE_MOEDA_INTERNA', 'Moeda do documento': 'MOEDA_DOCUMENTO',
                    'Equipe de vendas': 'EQUIPE_VENDAS', 'Equipe de vendas.1':'EQUIPE_DE_VENDAS', 'Escritório de vendas':'ESCRITORIO_VENDAS',
                    'Escritório de vendas.1': 'ESCRITORIO_DE_VENDAS', 'Doc.compensação': 'DOC_COMPENSACAO'})


print(df1.head(10))
print(df1.tail(10))
print(df1.info())

# EXISTE UMA DATA QUE FOI DIGITA ERRADA MANUALMENTE NO SAP e ENTÃO, VAMOS CORRIGIR
df1 = df1.replace('2202-07-15', '2022-07-15')

#verificar o período inicial das datas de documento
df1_data = df1.sort_values(by='DATA_DOCUMENTO')
print(df1_data[['DATA_DOCUMENTO']].head(5))
print(df1_data[['DATA_DOCUMENTO']].tail(5))

#MUDAR AS COLUNAS ESPECIFICADAS A SEGUIR PARA STRING E EXLUIR O .0 DA STRAING
df1[['EMPRESA', 'CONTA', 'ITEM', 'DOC_COMPENSACAO']] = df1[['EMPRESA', 'CONTA', 'ITEM', 'DOC_COMPENSACAO']].astype(str)
df1[['EMPRESA', 'CONTA', 'ITEM', 'DOC_COMPENSACAO']] = df1[['EMPRESA', 'CONTA', 'ITEM', 'DOC_COMPENSACAO']].replace(r'\.0$', '', regex=True)

print(df1)

print(df1.info())


# EXPORTAR BASE
df1.to_parquet("C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/BasesDuplicatasTratadas.parquet", engine = 'pyarrow', compression = 'gzip')


