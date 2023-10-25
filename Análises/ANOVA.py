import pandas as pd
from scipy.stats import f_oneway, levene, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = 'Life_Expectancy_Data.csv'
df = pd.read_csv(data)

#ANOVA USANDO SCIPY
unique_years = df['Year'].unique()
life_expectancy_by_year = []
for year in unique_years:
    group = df[df['Year'] == year]['Life expectancy ']
    if len(group) >= 3:
        life_expectancy_by_year.append(group)

# Realize a ANOVA apenas nos grupos válidos
if len(life_expectancy_by_year) > 1:
    f_statistic, p_value = f_oneway(*life_expectancy_by_year)
    # Examine o valor de p para determinar se há diferenças significativas entre os grupos
    if p_value < 0.05:
        print("Há diferenças significativas entre os grupos (p < 0.05)")
    else:
        print("Não há diferenças significativas entre os grupos (p >= 0.05)")
else:
    print("Não há grupos com pelo menos 3 observações para realizar a ANOVA.")
print("Estatística F:", f_statistic)
print("Valor P:", p_value)


# Realizar o teste de Shapiro-Wilk para cada grupo
for year, group in zip(unique_years, life_expectancy_by_year):
    stat, p = shapiro(group)
    print(f'Ano {year}: Statistic={stat:.3f}, p-value={p:.3f}')


# Realizar o teste de Levene
stat, p = levene(*life_expectancy_by_year)
print(f'\nTeste de Levene: Statistic={stat:.3f}, p-value={p:.3f}')









