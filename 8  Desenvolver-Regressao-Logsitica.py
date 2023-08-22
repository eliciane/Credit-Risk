'''

MODELO 3 - VERSÃO 2

ESTE CÓDIGO DIZ RESPEITO AO MODELO DE REGRESSÃO LOGÍSTICA .
A BASE DE DADOS CONTEM INFORMAÇÃO DE DUPLICATAS PAGAS DE JANEIRO DE 2018 A ABRIL 2023.
O TARGET FOI DEFINIDO PELA MÉDIA MENSAL DE ATRASOS >= 1 DIA ATRASO


A BASE DO SCORE COM OS CLIENTES E OS REFERIDOS SCOREs É EXPORTADA NA LINHA 305


'''




### Importar bibliotecas

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics



import warnings
warnings.simplefilter('ignore', DeprecationWarning)

plt.rcParams['figure.figsize'] = [5, 5]

# imprimir maximo colunas, linhas e 4 digitos para casas decimais para float
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.expand_frame_repr', False)



# Entrar com a base de dados exportada da etapa anterior '7 Determinar-Cliente-Bom-Mau_Pagador.py'
df = pd.read_parquet('C:/Users/eliciane.silva/PycharmProjects/Credit-Risk/MODELO3_REG_LOG.parquet', engine = 'auto')

print(df.info())



#IMPRIMIR APENAS UM CLIENTE PARA VER COMO SAI O TARGET
df_cliente = df.loc[(df['CONTA_CREDITO'] == '1004168')]
print(df_cliente)




print(df.columns)

print(len(df.columns))

df = df.drop(['DATA_MAX_VENC', 'DATA_ULT_30D', 'LIMITE_CREDITO', 'NOME_CLIENTE',], axis=1)

# MUDAR A ORDEM DAS COLUNAS
df = df[['CONTA_CREDITO', 'EQUIPE_VENDAS_new','MACROREGIAO', 'CHV.SETOR INDUSTRIAL', 'TARGET', 'PER_BOL_ATR_NTRANS_CC',
'VALOR_MONETARIO_TOTAL', 'TOT_FAT_ULT_12',  'TICKET_MÉDIO', 'PER_BOL_ATR_NTRANS_30', 'PER_BOL_ATR_NTRANS_60', 'PER_BOL_ATR_NTRANS_90',
'MEDIA_30_DIA_ATR', 'MEDIA_60_DIA_ATR',   'MEDIA_90_DIA_ATR',  'N_TRANS_CC', 'BOL_ATR_CC',  'DIAS_SEM_ATRASO_CC', 'FREQUENCIA', 'RECENCIA', 'TP_RELAC',
'TP_FUND',  'D_ATR_CC', 'SOMA_30_BOL_ATR', 'SOMA_60_BOL_ATR',  'SOMA_90_BOL_ATR', 'SOMA_30_TRANS', 'SOMA_60_TRANS', 'SOMA_90_TRANS']]

print(len(df.columns))

# TRANSFORMAR O TIPO DE COLUNAS
df[['MEDIA_30_DIA_ATR', 'MEDIA_60_DIA_ATR', 'MEDIA_90_DIA_ATR',  'N_TRANS_CC', 'BOL_ATR_CC',  'DIAS_SEM_ATRASO_CC', 'FREQUENCIA', 'RECENCIA', 'TP_RELAC',
'TP_FUND',  'D_ATR_CC', 'TARGET', 'SOMA_30_BOL_ATR', 'SOMA_60_BOL_ATR',  'SOMA_90_BOL_ATR', 'SOMA_30_TRANS', 'SOMA_60_TRANS', 'SOMA_90_TRANS']] = df[['MEDIA_30_DIA_ATR', 'MEDIA_60_DIA_ATR',   'MEDIA_90_DIA_ATR',  'N_TRANS_CC', 'BOL_ATR_CC',  'DIAS_SEM_ATRASO_CC', 'FREQUENCIA', 'RECENCIA', 'TP_RELAC',
'TP_FUND',  'D_ATR_CC', 'TARGET', 'SOMA_30_BOL_ATR', 'SOMA_60_BOL_ATR',  'SOMA_90_BOL_ATR', 'SOMA_30_TRANS', 'SOMA_60_TRANS', 'SOMA_90_TRANS']].astype('int')

df['TARGET'] = df['TARGET'].astype(int)

print(df.info())

#verificar se existem duplicados
duplicados = df.duplicated()
duplicados = duplicados.sum()
print('a base de dados tem %s dados duplicados.' %(duplicados))

#imprimir as colunas
print(df.columns)


#### criar listas de variáveis preditoras e variável resposta para usar posteriormente
var_preditoras = ['CONTA_CREDITO','EQUIPE_VENDAS_new','MACROREGIAO', 'CHV.SETOR INDUSTRIAL', 'PER_BOL_ATR_NTRANS_CC',
'VALOR_MONETARIO_TOTAL', 'TOT_FAT_ULT_12',  'TICKET_MÉDIO', 'PER_BOL_ATR_NTRANS_30', 'PER_BOL_ATR_NTRANS_60', 'PER_BOL_ATR_NTRANS_90',
'MEDIA_30_DIA_ATR', 'MEDIA_60_DIA_ATR',   'MEDIA_90_DIA_ATR',  'N_TRANS_CC', 'BOL_ATR_CC',  'DIAS_SEM_ATRASO_CC', 'FREQUENCIA', 'RECENCIA', 'TP_RELAC',
'TP_FUND',  'D_ATR_CC', 'SOMA_30_BOL_ATR', 'SOMA_60_BOL_ATR',  'SOMA_90_BOL_ATR', 'SOMA_30_TRANS', 'SOMA_60_TRANS', 'SOMA_90_TRANS']

var_resposta = ['TARGET']

df= df[var_preditoras + var_resposta]
df = df.drop_duplicates()
print(df.info())
print(df)


#TRANSFORMAR AS VARIÁVEIS CATEGÓRICAS DE DUMMIES
var_cat = ['CHV.SETOR INDUSTRIAL', 'EQUIPE_VENDAS_new', 'MACROREGIAO']

for attr in var_cat:
    df = df.merge(pd.get_dummies(df[attr], prefix=attr), left_index=True, right_index=True)
    df.drop(attr, axis=1, inplace=True)


#### DEFINIR A VARIÁVEL RESPOSTA e PREDITORAS. Transformar/padronizar o dataset. Separar a base em treino e teste #######
# Putting feature variable to X
X = df.drop(['TARGET','CONTA_CREDITO'],axis=1)

# Definir variáveis resposta (Y)
y = df[['TARGET']]
y_CC = df[['TARGET', 'CONTA_CREDITO']]  #guardar a conta crédito para usar posteriormente quando obtiver as probalidades
print(y.head())

### FAZER ANÁLISE FATORIAL EXPLORATÓRIA PARA TODAS AS VARIÁVEIS de X
pca = PCA(n_components=7)
X = pca.fit_transform(X)
print(X)


##### SIMULAÇÃO 1: RODAR O MODELO SEM TREINO E TESTE E TENTAR POSTERIORMENTE FAZER STEP WISE
#ADICIONAR A CONSTANTE
x_const = sm.add_constant(X)
# INSTANCIAR O MODELO USANDO STATS
lm = sm.Logit(y, x_const).fit(method='nm', maxiter=2000)
results = lm.summary()
print(results)

### INSTANCIAR A REGRESSÃO LOGISTICA SOMENTE COM default settings e depois medir a acurácia, usando o sklearn
modelo = LogisticRegression() #quantidade de iterações que o modelo irá realizar
modelo.fit(X, y)

#medir a acurácia do modelo
print('\n a acurácia do modelo é:', modelo.score(X, y)) # O modelo ficou super ajustado

# TENTAR OUTRO ALGORÍTMO PARA STEPWISE ### o step wise não desempenha, pois não sabemos quais fatores foram reduzidos
#backselect = step_reg.backward_regression(X, y, 0.05,verbose=False)
#print('\n variaveis selecionada pelo STEPWISE:', backselect)

### FAZENDO A PREDIÇÃO, USANDO PREDICT_PROB PARA SAIR AS PROBABILIADES AGORA .predict_proba do sklearn
prob_previsao = modelo.predict_proba(X)[:,1]
print(prob_previsao) #precisa imprimir todas as linhas

preds = modelo.predict(X)
print(preds)


#Imprimir os resultados da predição: Depois que um modelo é definido, você pode verificar seu desempenho com .predict_proba(),
# que retorna a matriz de probabilidades, sendo a saída prevista seja para o bom pagador (0) e para o mau pagador (1)
prob_previsao = modelo.predict_proba(X)[:,1]
print(prob_previsao) # imprimi somente as probabilidades dos maus pagadores (1)

prob_matrix = modelo.predict_proba(X)
#print(prob_matrix)
# Imprimir no formato lado a lado e com três casas decimais e com as probabilidades dos bons (0) e maus pagadores (1)
for probabilities in prob_matrix:
    print("{:.3f}  {:.3f}".format(probabilities[0], probabilities[1]))

num_rows = prob_matrix.shape[0]
print("O número de linhas de probabilidades é:", num_rows)


#medir a acurácia do modelo
print('\n a acurácia do modelo treino é:', modelo.score(X, y))


# Imprimir a matriz de confusão
tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
print('True negatives: ', tn, '\nFalse positives: ', fp, '\nFalse negatives: ', fn, '\nTrue positives: ', tp)


print(classification_report(y, preds))


# EXPORTAR AS PREDIÇÕES
# Converter o array das previsões de probabilidades do mau pagador em dataframe
y_pred_df = pd.DataFrame(prob_matrix)
print(y_pred_df)
# Converter somente o mau pagador para uma coluna
y_pred_1 = y_pred_df.iloc[:,[1]]
print(y_pred_1.shape)
print(y_pred_1.head())

# Tazer o dataframe origninal com as colunas CONTA_CREDITO e TARGET
y_test_cc = y_CC.reset_index(drop=True)
#print(y_test_cc)
pd.set_option('display.precision', 4)#IMPRIMIR 4 digitos

# CONCATENAR/ADICIONAR OS DATAFRAME ORIGINAL DE CONTA_CREDITO, TARGET COM O DATAFRAME DE PROBILIDADES
y_pred_final = pd.concat([y_test_cc,y_pred_1],axis=1)


# RENOMEAR A COLUNA DE PROBABILIDES PARA TARGET_PROB
y_pred_final= y_pred_final.rename(columns={ 1 : 'TARGET_Prob'})

# REARANJAR AS COLUNAS
y_pred_final = y_pred_final.reindex(['CONTA_CREDITO','TARGET','TARGET_Prob'], axis=1)

# VER COMO FICOU O DAFRAME FINAL
print(y_pred_final.head())



# CRIAR UMA COLUNA DE PREDIÇÃO COM VALOR 1 SE TARGET_PROB > 0.5 caso contrário, atribuir VALOR 0
y_pred_final['PREDICAO'] = y_pred_final.TARGET_Prob.map( lambda x: 1 if x > 0.5 else 0)

# IMPRIMIR NOVAMENTE
print(y_pred_final.head())

# CLASSIFICAR A CONTA CRÉDITO DO MENOR PARA MAIOR
y_pred_final = y_pred_final.sort_values('CONTA_CREDITO')


#exportar o DATAFRAME
#y_pred_final.to_excel('C:/Users/eliciane.silva/downloads/Prob_Modelo3.xlsx', index=False)

# Confusion matrix
confusion = metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final.PREDICAO ) # fazer a matriz de confusão pelo pacote sklearn
print(confusion)

sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.6g')
print(confusion)
plt.plot(confusion)
plt.show()

#Let's check the overall accuracy.
acc = metrics.accuracy_score( y_pred_final.TARGET, y_pred_final.PREDICAO)
print(acc)
TP = confusion[1,1] # true positive
print(TP)
TN = confusion[0,0] # true negatives
print(TN)
FP = confusion[0,1] # false positives
print(FP)
FN = confusion[1,0] # false negatives
print(FN)
# Let's see the sensitivity of our logistic regression model
Sensitivity = TP / float(TP+FN)
# Let us calculate specificity
Specificity = TN / float(TN+FP)
print('Accuracy : ', acc)
print('Sensitivity ',Sensitivity, ' Specificity ', Specificity)



def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, thresholds
print(draw_roc(y_pred_final.TARGET, y_pred_final.PREDICAO))


#Encontrar o ponto ótimo de cutoff
#O ponto ótimo do cutoff da probabilidade é ponto onde obtemos um balanço entre sensistividade e especificidade

# Criar uma coluna com diferentes pontos de cutoffs de probabilidade
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_pred_final[i]= y_pred_final.TARGET_Prob.map( lambda x: 1 if x > i else 0)
print(y_pred_final.head())

#y_pred_final.to_excel('C:/Users/eliciane.silva/downloads/PredicaoFinal.xlsx', index=False)

# Calcular a acurácia, sensibilidade e especificidade para diferentes pontos de cutoffs de probabilidade
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    sensi = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    speci = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)

#cutoff_df.to_excel('C:/Users/eliciane.silva/downloads/Prob_Modelo2_14meses_varios_cutt_off_otimo.xlsx', index=False)

# Plotar o gráfico com a acurácia, sensitividade e especificidade para as várias probabilidades.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# Mediante a curva ROC, 0.6 é o ponto ótimo do cutoff probabilidade d
y_pred_final['final_predicted'] = y_pred_final.TARGET_Prob.map( lambda x: 1 if x > 0.6 else 0)
print(y_pred_final.head())

y_pred_final.to_excel('C:/Users/eliciane.silva/downloads/PREDIÇÃO_MODELO3_cutt_off_otimo_04AGO.xlsx', index=False) ############### AQUI QUE SAI O RESULTADO DO MODELO ACEITO ##########

#Checar novamente a acurácia geral
print(metrics.accuracy_score( y_pred_final.TARGET, y_pred_final.final_predicted))

print(metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final.final_predicted ))

#Fazer a matriz de confusão com a nova classificação apresentada na coluna final_predicted por meio do ponto ótimo
confusion = metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final.final_predicted  )
acc = metrics.accuracy_score( y_pred_final.TARGET, y_pred_final.final_predicted)
TP = confusion[1,1] # true positive
print('\n Verdadeiros Positivos 1-1:', TP )
TN = confusion[0,0] # true negatives
print('\n Verdadeiros Negativos 0-0:', TN )
FP = confusion[0,1] # false positives
print('\n Falsos Positivos 0-1:', FP )
FN = confusion[1,0] # false negatives
print('\n Falsos Negativos 1-0:', FN )
# Let's see the sensitivity of our logistic regression model
Sensitivity = TP / float(TP+FN)
# Let us calculate specificity
Specificity = TN / float(TN+FP)
print('Accuracy : ', acc)
print('Sensitivity ',Sensitivity, ' Specificity ', Specificity)



# Imprimir a MATRIX na forma de gráfico
confusion = metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final.final_predicted)
print(confusion)
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.6g')
#print(confusion)
plt.plot(confusion)
plt.show()

print(draw_roc(y_pred_final.TARGET, y_pred_final.final_predicted))






##### IMPUTAR DADOS E DIVIDIR EM TREINO E TESTE

### PARA IMPUTAR OS DADOS O DATASET DEVE SER SEPARADO EM TREINO E TESTE
#### DEFINIR A VARIÁVEL RESPOSTA e PREDITORAS. Separar a base em treino e teste #######
# Putting feature variable to X
X = df.drop(['TARGET','CONTA_CREDITO'],axis=1)

# Putting response variable to y
y = df[['TARGET', 'CONTA_CREDITO']] #guardar a conta crédito para usar posteriormente quando obtiver as probalidades
print(y.head())


### FAZER ANÁLISE FATORIAL EXPLORATÓRIA PARA TODAS AS VARIÁVEIS de X
pca = PCA(n_components=19)
X = pca.fit_transform(X)
print(X)

# Splitting the data into train and test
SEED = 77
X_train, X_test, y_train_cc, y_test_cc = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=SEED)


#Nos dados de y_test e y_train manter somente o TARGET para o rodar o modelo, excluindo a conta crédito
print(y_test_cc) # aqui a conta crédito foi mantida para posteriormente, juntar com o dataset de predições
print(y_test_cc.shape)
y_test = y_test_cc['TARGET']
y_train = y_train_cc['TARGET']

print('quantidade de mau pagadores na base toda:\n \n', df['TARGET'].value_counts()[1])
print('quantidade de bons pagadores na base toda:\n \n', df['TARGET'].value_counts()[0])

print('\n \n quantidade de maus pagadores na base de treino:',y_train_cc['TARGET'].value_counts()[1])
print('\n \n quantidade de bons pagadores na base de treino:', y_train_cc['TARGET'].value_counts()[0])

print('\n \n quantidade de maus pagadores na base de teste:', y_test_cc['TARGET'].value_counts()[1])
print('\n \n quantidade de bons pagadores na base de teste:', y_test_cc['TARGET'].value_counts()[0])



### IMPUTAR DADOS
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

print('\n \n quantidade de maus pagadores na base de treino:',y_train_cc['TARGET'].value_counts()[1])
print('\n \n quantidade de bons pagadores na base de treino:', y_train_cc['TARGET'].value_counts()[0])

print('\n \n quantidade de maus pagadores na base de treino:',y_resampled.value_counts()[1])
print('\n \n quantidade de bons pagadores na base de treino:', y_resampled.value_counts()[0])


#ADICIONAR A CONSTANTE
x_const = sm.add_constant(X_resampled)
# INSTANCIAR O MODELO USANDO STATS
lm = sm.Logit(y_resampled, x_const).fit(method='nm', maxiter=2000)
results = lm.summary()
print(results)


### INSTANCIAR A REGRESSÃO LOGISTICA SOMENTE COM default settings e depois medir a acurácia, usando o sklearn
modelo_smote = LogisticRegression() #quantidade de iterações que o modelo irá realizar
modelo_smote = modelo_smote.fit(X_resampled, y_resampled)


#medir a acurácia do modelo
print('\n a acurácia do modelo com SMOTE nos dados de treinoé:', modelo_smote.score(X_resampled, y_resampled)) # O modelo ficou super ajustado


# Fazer a predição somente com o conjunto de teste e não usar os dados imputados a fim de obter uma avalição honesta sobre os dados reais
preds_smote = modelo_smote.predict(X_test)
print(preds_smote)

#Imprimir os resultados da predição: Depois que um modelo é definido, você pode verificar seu desempenho com .predict_proba(),
# que retorna a matriz de probabilidades, sendo a saída prevista seja para o bom pagador (0) e para o mau pagador (1)
prob_previsao_smote = modelo_smote.predict_proba(X_test)[:,1]
print(prob_previsao_smote) # imprimi somente as probabilidades dos maus pagadores (1)

prob_matrix_smote = modelo_smote.predict_proba(X_test)
#print(prob_matrix)
# Imprimir no formato lado a lado e com três casas decimais e com as probabilidades dos bons (0) e maus pagadores (1)
for probabilities in prob_matrix_smote:
    print("{:.3f}  {:.3f}".format(probabilities[0], probabilities[1]))

num_rows = prob_matrix_smote.shape[0]
print("O número de linhas de probabilidades é:", num_rows)


# EXPORTAR AS PREDIÇÕES
# Converter o array das previsões de probabilidades do mau pagador em dataframe
y_pred_df = pd.DataFrame(prob_matrix_smote)
print(y_pred_df)
# Converter somente o mau pagador para uma coluna
y_pred_1 = y_pred_df.iloc[:,[1]]
print(y_pred_1.shape)
print(y_pred_1.head())

# Tazer o dataframe origninal com as colunas CONTA_CREDITO e TARGET
y_test_cc = y_test_cc.reset_index(drop=True)
print(y_test_cc)
print(y_test_cc.shape)
pd.set_option('display.precision', 4)#IMPRIMIR 4 digitos


# CONCATENAR/ADICIONAR OS DATAFRAME ORIGINAL DE CONTA_CREDITO, TARGET COM O DATAFRAME DE PROBILIDADES
y_pred_final = pd.concat([y_test_cc,y_pred_1],axis=1)


# RENOMEAR A COLUNA DE PROBABILIDES PARA TARGET_PROB
y_pred_final= y_pred_final.rename(columns={ 1 : 'TARGET_Prob'})

# REARANJAR AS COLUNAS
y_pred_final = y_pred_final.reindex(['CONTA_CREDITO','TARGET','TARGET_Prob'], axis=1)

# VER COMO FICOU O DAFRAME FINAL
y_pred_final.head()

# CRIAR UMA COLUNA DE PREDIÇÃO COM VALOR 1 SE TARGET_PROB > 0.5 caso contrário, atribuir VALOR 0
y_pred_final['PREDICAO'] = y_pred_final.TARGET_Prob.map( lambda x: 1 if x > 0.5 else 0)

# IMPRIMIR NOVAMENTE
print(y_pred_final.head())

# CLASSIFICAR A CONTA CRÉDITO DO MENOR PARA MAIOR
y_pred_final = y_pred_final.sort_values('CONTA_CREDITO')


#exportar o DATAFRAME
#y_pred_final.to_excel('C:/Users/eliciane.silva/downloads/Prob_Modelo3_SMOTE_teste.xlsx', index=False)



# IMPRIMIR MATRIZ DE CONFUSÃO
confusion = metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final.PREDICAO ) # fazer a matriz de confusão pelo pacote sklearn
print(confusion) # A ACURÁCIA E MATRIZ DE CONFUSÃO COM OS DADOS DE TESTE DERIVADOS DO MODELO SMOTE NÃO DESEMENHOU BEM

sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.6g')
print(confusion)
#plt.plot(confusion)
#plt.show()

#Let's check the overall accuracy.
acc = metrics.accuracy_score( y_pred_final.TARGET, y_pred_final.PREDICAO)
TP = confusion[1,1] # true positive
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our logistic regression model
Sensitivity = TP / float(TP+FN)
# Let us calculate specificity
Specificity = TN / float(TN+FP)
print('Accuracy : ', acc)
print('Sensitivity ',Sensitivity, ' Specificity ', Specificity)



def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 4))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, thresholds
print(draw_roc(y_pred_final.TARGET, y_pred_final.PREDICAO))







#Encontrar o ponto ótimo de cutoff
#O ponto ótimo do cutoff da probabilidade é ponto onde obtemos um balanço entre sensistividade e especificidade

# Criar uma coluna com diferentes pontos de cutoffs de probabilidade
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_pred_final[i]= y_pred_final.TARGET_Prob.map( lambda x: 1 if x > i else 0)
print(y_pred_final.head())

#y_pred_final.to_excel('C:/Users/eliciane.silva/downloads/PredicaoFinal.xlsx', index=False)

# Calcular a acurácia, sensibilidade e especificidade para diferentes pontos de cutoffs de probabilidade
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    sensi = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    speci = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)

#cutoff_df.to_excel('C:/Users/eliciane.silva/downloads/Prob_Modelo2_14meses_varios_cutt_off_otimo.xlsx', index=False)

# Plotar o gráfico com a acurácia, sensitividade e especificidade para as várias probabilidades.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show() # ao imprimir o ponto de corte ótimo, o valor da sensibilidade está muito baixo e não é possível encontrar um ponto de corte ótimo



# Mediante a curva ROC, 0.7 é o ponto ótimo do cutoff probabilidade d
y_pred_final['final_predicted'] = y_pred_final.TARGET_Prob.map( lambda x: 1 if x > 0.7 else 0)
print(y_pred_final.head())

y_pred_final.to_excel('C:/Users/eliciane.silva/downloads/PREDICOES_cutt_off_otimo_SMOTE.xlsx', index=False)

#Checar novamente a acurácia geral
print(metrics.accuracy_score( y_pred_final.TARGET, y_pred_final.final_predicted))

print(metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final.final_predicted ))

#Fazer a matriz de confusão com a nova classificação apresentada na coluna final_predicted por meio do ponto ótimo
confusion = metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final.final_predicted  )
acc = metrics.accuracy_score( y_pred_final.TARGET, y_pred_final.final_predicted)
TP = confusion[1,1] # true positive
print('\n Verdadeiros Positivos 1-1:', TP )
TN = confusion[0,0] # true negatives
print('\n Verdadeiros Negativos 0-0:', TN )
FP = confusion[0,1] # false positives
print('\n Falsos Positivos 0-1:', FP )
FN = confusion[1,0] # false negatives
print('\n Falsos Negativos 1-0:', FN )
# Let's see the sensitivity of our logistic regression model
Sensitivity = TP / float(TP+FN)
# Let us calculate specificity
Specificity = TN / float(TN+FP)
print('Accuracy : ', acc)
print('Sensitivity ',Sensitivity, ' Specificity ', Specificity)



# Imprimir a MATRIX na forma de gráfico
confusion = metrics.confusion_matrix( y_pred_final.TARGET, y_pred_final.final_predicted)
print(confusion)
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='.6g')
#print(confusion)
plt.plot(confusion)
plt.show()

print(draw_roc(y_pred_final.TARGET, y_pred_final.final_predicted))



