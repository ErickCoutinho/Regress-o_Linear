import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


data = 'Life_Expectancy_Data.csv'
df = pd.read_csv(data)
X = df[['Adult Mortality', 'in.fant deaths', ' thinness  1-19 years',' HIV/AIDS', 'GDP', 'Alcohol']]
y = df['Life expectancy ']

# Adicione uma constante (intercepto) ao modelo
# 1. Multicolinearidade
correlation_matrix = X.corr()
print("Matriz de Correlação:")
print(correlation_matrix)
plt.figure(figsize=(12, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis Independentes')
plt.show()

# Calcula o VIF para cada variável independente
vif = pd.DataFrame()
vif["Variável"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)

####### 2. Análise de Resíduos ##################################################
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
residuals = model.resid
# Gráfico de probabilidade normal dos resíduos
sm.qqplot(residuals, line='s')
plt.title("Gráfico de Probabilidade Normal dos Resíduos")
plt.show()
#Distribuição
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Histograma dos Resíduos')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()
print(residuals.head(50))

# ############## 3. Testes para Verificar as Suposições do Modelo #########################################
# Teste de Shapiro-Wilk para normalidade dos resíduos
shapiro_test = stats.shapiro(residuals)
print("Teste de Shapiro-Wilk:")
print("Estatística de Teste:", shapiro_test[0])
print("Valor-p:", shapiro_test[1])


# Teste de White para Homocedasticidade
white_test = sm.stats.diagnostic.het_white(model.resid, X)
print("Teste de White para Homocedasticidade:")
print("Estatística de Teste:", white_test[0])
print("Valor-p:", white_test[1])

# Teste de Durbin-Watson para autocorrelação
durbin_watson_test = sm.stats.stattools.durbin_watson(model.resid)
print("Teste de Durbin-Watson:")
print("Estatística de Teste:", durbin_watson_test)

