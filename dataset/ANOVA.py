import pandas as pd
from scipy.stats import f_oneway

data = 'Life_Expectancy_Data.csv'
df = pd.read_csv(data)

#ANOVA USANDO SCIPY
groups = df['Status'].unique()
data = []
for group in groups:
    data.append(df[df['Status'] == group]['Life expectancy '])
f_statistic, p_value = f_oneway(*data)
print("Estatística F:", f_statistic)
print("Valor-p:", p_value)

# Interpretação dos resultados
alpha = 0.05
if p_value < alpha:
    print("Rejeitamos a hipótese nula. Pelo menos um grupo é significativamente diferente.")
else:
    print("Não rejeitamos a hipótese nula. Não há evidência de diferença significativa entre os grupos.")

"""O resultado do teste ANOVA indica que há uma diferença significativa entre pelo menos um dos grupos do "Status do País" em relação à "Expectativa de Vida". Como o valor-p (p-value) é muito baixo (3.92e-80), podemos rejeitar a hipótese nula.
Isso significa que o "Status do País" tem um efeito significativo na "Expectativa de Vida" e que pelo menos um dos grupos de status é significativamente diferente em termos de expectativa de vida dos outros grupos. Você pode prosseguir com análises adicionais, como testes de comparação de pares (por exemplo, teste de Tukey) para determinar quais grupos específicos são diferentes entre si. Isso pode ajudar a identificar quais categorias de status têm um impacto mais significativo na expectativa de vida."""




"""O valor-p (p-value) é uma medida estatística que indica a probabilidade de obter os resultados observados (ou resultados mais extremos) em um teste estatístico, assumindo que a hipótese nula seja verdadeira. Quanto menor o valor-p, mais forte é a evidência contra a hipótese nula. Em geral, um valor-p menor que 0.05 (ou 5%) é frequentemente considerado um limiar comum para rejeitar a hipótese nula.
No seu resultado anterior, o valor-p foi 3.92e-80, o que significa que é um número muito próximo de zero. Nesse caso, isso indica que a probabilidade de observar a diferença na "Expectativa de Vida" entre os grupos de "Status do País" devido ao acaso é extremamente baixa. Portanto, podemos dizer que a diferença é estatisticamente significativa, e isso é uma evidência robusta contra a hipótese nula.
Em resumo, um valor-p muito baixo, como 3.92e-80, é considerado muito significativo e indica uma forte evidência estatística de que existe uma diferença real entre os grupos analisados."""