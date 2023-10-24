import numpy as np
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


data = 'Life_Expectancy_Data.csv'
df = pd.read_csv(data)
X = df[['Adult Mortality', 'in.fant deaths', ' thinness  1-19 years']]
y = df['Life expectancy ']

# Adicione uma constante (intercepto) ao modelo
# 1. Multicolinearidade
correlation_matrix = X.corr()
print("Matriz de Correlação:")
print(correlation_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis Independentes')
plt.show()


"""parece que há alguma multicolinearidade moderada entre as variáveis independentes "Adult Mortality," "in.fant deaths," e "thinness 1-19 years" em seu modelo de regressão. Vamos destacar alguns pontos-chave:
Correlações entre Variáveis Independentes: A matriz de correlação que você forneceu mostra que as correlações entre essas variáveis independentes não são muito altas. As correlações variam de 0.04245 a 0.463415. Geralmente, multicolinearidade é mais preocupante quando as correlações são próximas de 1.
Valores de VIF: Os valores do Fator de Inflação da Variância (VIF) que você calculou também sugerem que a multicolinearidade não é um problema significativo. Um VIF abaixo de 5 é geralmente considerado aceitável. Neste caso, todos os valores de VIF estão abaixo desse limite.
Portanto, com base nessas informações, parece que a multicolinearidade entre essas variáveis independentes em seu modelo de regressão é moderada, mas não tão problemática a ponto de exigir ação imediata. No entanto, sempre é aconselhável estar ciente da multicolinearidade e monitorá-la durante a análise, especialmente se você adicionar mais variáveis ao modelo."""

# Calcula o VIF para cada variável independente
vif = pd.DataFrame()
vif["Variável"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif)

####### 2. Análise de Resíduos ##################################################
model = sm.OLS(y, X).fit()
X = sm.add_constant(X)
# Resíduos do modelo
residuals = model.resid
# Gráfico de probabilidade normal dos resíduos
sm.qqplot(residuals, line='s')
plt.title("\nGráfico de Probabilidade Normal dos Resíduos")
plt.show()
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.title('Histograma dos Resíduos')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()


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



""""Teste de Shapiro-Wilk:
Estatística de Teste: 0.9343
Valor-p: 2.3762e-26

O Teste de Shapiro-Wilk é usado para verificar se os resíduos seguem uma distribuição normal. No seu caso, o valor-p é muito baixo (próximo de zero), o que sugere que os resíduos não seguem uma distribuição normal. Isso indica que o pressuposto de normalidade dos resíduos não é satisfeito. Em termos simples, os resíduos não estão distribuídos de forma simétrica em torno de zero.
Teste de White para Homocedasticidade:
Estatística de Teste: 1547.9434
Valor-p: 0.0
O Teste de White é utilizado para verificar a homocedasticidade dos resíduos. O valor-p muito baixo (próximo de zero) sugere a presença de heterocedasticidade nos resíduos. Isso significa que a variabilidade dos erros não é constante ao longo dos valores previstos. Em outras palavras, a dispersão dos resíduos varia à medida que os valores previstos aumentam ou diminuem.

Teste de Durbin-Watson:
Estatística de Teste: 0.2798
O Teste de Durbin-Watson verifica a autocorrelação nos resíduos. Um valor próximo de 2 indica que não há autocorrelação positiva ou negativa nos resíduos, o que é desejável. No entanto, no seu caso, a estatística de teste é muito baixa (próxima a 0), sugerindo a presença de autocorrelação positiva nos resíduos. Isso significa que os resíduos em um determinado período estão correlacionados com os resíduos em um período anterior, o que pode indicar que o modelo não está capturando adequadamente a estrutura de dependência temporal nos dados.
Em resumo, esses resultados sugerem que há desafios em relação aos pressupostos do modelo de regressão linear. Os resíduos não seguem uma distribuição normal, há heterocedasticidade presente e também autocorrelação positiva nos resíduos. Isso pode afetar a precisão das estimativas do modelo e requer considerações adicionais, como transformações nos dados ou a escolha de outros modelos estatísticos adequados."""