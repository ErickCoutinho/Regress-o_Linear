import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
import statsmodels.api as sm
import statsmodels.stats.api as sms
import scipy.stats as stats
import numpy as np


data = 'Life_Expectancy_Data.csv'
df = pd.read_csv(data)

X = df[['Adult Mortality', 'in.fant deaths', ' thinness  1-19 years',' HIV/AIDS', 'GDP', 'Alcohol']]
y = df['Life expectancy ']
#Expectativa de vida: observe a bola de cristal da saúde da população, revelando quanto tempo as pessoas podem esperar viver.
# Mortalidade de adultos: Avalie as probabilidades de sobrevivência entre 15 e 60 anos de idade por 1.000 habitantes.
# Mortes Infantis: Aprofunde-se na saúde infantil com o número de mortes infantis por 1.000 nascidos vivos.
#Magreza de 1 a 19 anos: Explore a prevalência de magreza entre crianças e adolescentes de 1 a 19 anos.
#Magreza de 5 a 9 anos: Amplie a magreza entre crianças de 5 a 9 anos.

#VERIFICANDO SE Y É NORMAL
plt.hist(y, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histograma da Expectativa de Vida')
plt.xlabel('Expectativa de Vida')
plt.ylabel('Densidade')
plt.show()
# Shapiro-Wilk
stat, p = shapiro(df['Life expectancy '])
if p > 0.05:
    print('A distribuição parece normal (não rejeitar a hipótese nula)')
else:
    print('A distribuição não parece normal (rejeitar a hipótese nula)')


"""#APLICAR TRANSFOMAÇÃO LOGARITMA PARA NORMALIZAR VARIAVEL RESPOSTA
transformed_life_expectancy, lambda_best_fit = stats.yeojohnson(df['Life expectancy '])
log_var_resposta = transformed_life_expectancy

# Shapiro-Wilk após transformação
stat, p = shapiro(log_var_resposta)
if p > 0.05:
    print('A distribuição parece normal (não rejeitar a hipótese nula)')
else:
    print('A distribuição não parece normal (rejeitar a hipótese nula)')

plt.hist(log_var_resposta, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histograma da Expectativa de Vida')
plt.xlabel('Expectativa de Vida')
plt.ylabel('Densidade')
plt.show()"""


#NORMALIZAÇAÕ DAS VARIAVEIS INDEPENDENTES
"""scaler = StandardScaler()
normalized_data = scaler.fit_transform(X)
normalized_df = pd.DataFrame(data=normalized_data, columns=X.columns)"""
#######################################################################

#treinamento e ajuste do modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R²:", r2)
print("MSE:", mse)
print("MAE:", mae)
coefficients = model.coef_
intercept = model.intercept_

print("Coeficientes:", coefficients)
print("Intercept:", intercept)





plt.scatter(y_test, y_pred, c='blue', label='Dados Reais vs. Previstos')

# Adicionar a linha de regressão
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Regressão Linear')
plt.xlabel('Expectativa de Vida Real')
plt.ylabel('Expectativa de Vida Prevista')
plt.title('Regressão Linear Múltipla')
plt.legend(loc='best')

plt.show()

#Este código irá gerar um gráfico de dispersão onde os pontos azuis representam os valores reais da expectativa de vida em relação aos valores previstos pelo modelo de regressão linear. A linha preta tracejada representa a linha de regressão.
