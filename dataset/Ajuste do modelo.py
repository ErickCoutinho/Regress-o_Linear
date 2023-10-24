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

X = df[['Adult Mortality', 'in.fant deaths', ' thinness  1-19 years']]
y = df['Life expectancy ']

#Expectativa de vida: observe a bola de cristal da saúde da população, revelando quanto tempo as pessoas podem esperar viver.
# Mortalidade de adultos: Avalie as probabilidades de sobrevivência entre 15 e 60 anos de idade por 1.000 habitantes.
# Mortes Infantis: Aprofunde-se na saúde infantil com o número de mortes infantis por 1.000 nascidos vivos.
#Magreza de 1 a 19 anos: Explore a prevalência de magreza entre crianças e adolescentes de 1 a 19 anos.
#Magreza de 5 a 9 anos: Amplie a magreza entre crianças de 5 a 9 anos.

#VERIFICANDO SE Y É NORMAL
plt.hist(df['Life expectancy '], bins=30, density=True, alpha=0.6, color='g')
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

#EXPLICAÇÃO SEM NORMALIZAR
"""Os valores fornecidos são métricas comuns usadas para avaliar a qualidade de um modelo de regressão linear. Aqui está a interpretação de cada um deles:
R² (R-squared): O R² é uma medida da proporção da variabilidade na variável dependente (nesse caso, a expectativa de vida) que é explicada pelas variáveis independentes (Adult Mortality, in.fant deaths, thinness 1-19 years, thinness 5-9 years) incluídas no modelo. Um valor de R² de 0,5208 significa que aproximadamente 52,08% da variabilidade na expectativa de vida pode ser explicada pelas variáveis independentes no modelo. Quanto mais próximo de 1, melhor o modelo.
MSE (Mean Squared Error): O MSE é uma medida da média dos quadrados dos erros entre os valores reais e os valores previstos pelo modelo. Nesse caso, o valor do MSE é 34,0365. Quanto menor o MSE, melhor o ajuste do modelo aos dados. Portanto, um valor mais baixo indica que o modelo é melhor em fazer previsões precisas.
MAE (Mean Absolute Error): O MAE é uma medida da média dos valores absolutos dos erros entre os valores reais e os valores previstos pelo modelo. Nesse caso, o valor do MAE é 4,1411. O MAE fornece uma ideia da magnitude dos erros de previsão. Quanto menor o MAE, menor é a magnitude dos erros e, portanto, melhor o modelo em fazer previsões precisas.
Coeficientes: Os coeficientes fornecidos estão associados às variáveis independentes. No seu modelo, os coeficientes são os seguintes:
Adult Mortality: -0,0432
in.fant deaths: -0,0008
thinness 1-19 years: -0,4488
thinness 5-9 years: -0,1245
Esses coeficientes representam como cada variável independente afeta a expectativa de vida. Por exemplo, um aumento em "Adult Mortality" de uma unidade está associado a uma diminuição de aproximadamente 0,0432 na expectativa de vida, mantendo as outras variáveis constantes.
Intercept: O intercepto é o valor da expectativa de vida quando todas as variáveis independentes são iguais a zero. Neste caso, o intercepto é de aproximadamente 79,286 anos.
Em resumo, seu modelo de regressão linear explica cerca de 52,08% da variabilidade na expectativa de vida usando as variáveis independentes fornecidas. Ele tem um erro médio quadrático de 34,0365 anos e um erro médio absoluto de 4,1411 anos. Os coeficientes indicam como cada variável afeta a expectativa de vida, e o intercepto representa a expectativa de vida quando todas as variáveis são zero."""




plt.scatter(y_test, y_pred, c='blue', label='Dados Reais vs. Previstos')

# Adicionar a linha de regressão
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Regressão Linear')
plt.xlabel('Expectativa de Vida Real')
plt.ylabel('Expectativa de Vida Prevista')
plt.title('Regressão Linear Múltipla')
plt.legend(loc='best')

plt.show()

#Este código irá gerar um gráfico de dispersão onde os pontos azuis representam os valores reais da expectativa de vida em relação aos valores previstos pelo modelo de regressão linear. A linha preta tracejada representa a linha de regressão.

"""Para realizar uma regressão linear múltipla, é necessário que tanto as variáveis independentes (X) quanto a variável dependente (Y) atendam a certos pressupostos. Aqui estão os principais pressupostos para a regressão linear múltipla:
Para as variáveis independentes (X):
Independência: As variáveis independentes devem ser independentes umas das outras. Isso significa que não deve haver multicolinearidade significativa, o que ocorre quando duas ou mais variáveis independentes estão altamente correlacionadas.
Linearidade: A relação entre as variáveis independentes e a variável dependente deve ser linear. Isso significa que as mudanças nas variáveis independentes devem se traduzir em mudanças proporcionais na variável dependente.
Homoscedasticidade: A variância dos erros deve ser constante em todos os níveis das variáveis independentes. Isso significa que a dispersão dos resíduos (diferenças entre os valores observados e os valores previstos) deve ser a mesma em todos os níveis das variáveis independentes.
Para a variável dependente (Y):
4. Normalidade: A distribuição dos resíduos deve ser aproximadamente normal. Isso implica que a distribuição dos erros deve seguir uma distribuição normal, com média zero.
Portanto, tanto X (variáveis independentes) quanto Y (variável dependente) precisam atender a esses pressupostos para que a regressão linear múltipla seja apropriada. Se esses pressupostos não forem atendidos, os resultados da análise de regressão podem não ser confiáveis e precisos, e podem ser necessárias transformações nos dados ou considerar outros modelos estatísticos."""