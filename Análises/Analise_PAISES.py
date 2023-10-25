import pandas as pd
import matplotlib.pyplot as plt


data = 'Life_Expectancy_Data.csv'
df = pd.read_csv(data)

Mean_LifeEx = df.groupby('Country')['Life expectancy '].mean()
sorted_mean_life = Mean_LifeEx.sort_values(ascending=False)
top_5 = sorted_mean_life.head(5)
latest_5 = sorted_mean_life.tail(5)

#GRAFICO MAIORES E MENORES EXPECTATIVAS DE VIDA
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)  # Subplot 1
top_5.plot(kind='bar', color='gold')
plt.title('Países com as maiores expectativas de vida')
plt.xlabel('País')
plt.ylabel('Expectativa de Vida')
plt.xticks(rotation=0)
for i, v in enumerate(top_5):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.subplot(1, 2, 2)  # Subplot 2
latest_5.plot(kind='bar', color='black')
plt.title('Países com as menores expectativas de vida')
plt.xlabel('País')
plt.ylabel('Expectativa de Vida')
plt.xticks(rotation=0)
plt.tight_layout()  # Para evitar sobreposição
for i, v in enumerate(latest_5):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.show()

################################################################################################
#EXPECTATIVA DE VIDA DO BRASIL AO LONGO DO TEMPO
brazil_data = df[df['Country'] == 'Brazil']
plt.figure(figsize=(10, 6))
plt.plot(brazil_data['Year'], brazil_data['Life expectancy '], label='Brazil', marker='o', color = 'black')
plt.xlabel('Ano')
plt.ylabel('Expectativa de Vida')
plt.title('Tendência de Expectativa de Vida no Brasil')
plt.legend()
plt.grid(True)
plt.show()
#Argentina
usa_data = df[df['Country'] == 'Argentina']
plt.figure(figsize=(10, 6))
plt.plot(usa_data['Year'], usa_data['Life expectancy '], label='Argentina', marker='o', color='black')
plt.xlabel('Ano')
plt.ylabel('Expectativa de Vida')
plt.title('Tendência de Expectativa de Vida na Argentina')
plt.legend()
plt.grid(True)
plt.show()
# Expectativa de vida: observe a bola de cristal da saúde da população, revelando quanto tempo as pessoas podem esperar viver.

############################################################################################################################
# ANALISES ESTATISTICAS SOBRE NIVEL DE DESENVOLVIMENTO
status_groups = df.groupby('Status')
# Calculando estatísticas resumidas
summary_stats = status_groups[['Life expectancy ', 'in.fant deaths', 'Schooling']].mean()
plt.figure(figsize=(14, 8))
# Gráfico de barras para a expectativa de vida
plt.subplot(131)  # 1 linha, 3 colunas, posição 1
summary_stats['Life expectancy '].plot(kind='bar', color = 'tomato')
plt.title('Expectativa de Vida por Status de Desenvolvimento', fontsize=10)
plt.xlabel('Status')
plt.ylabel('Expectativa de Vida')
# Gráfico de barras para a mortalidade infantil
plt.subplot(132)  # 1 linha, 3 colunas, posição 2
summary_stats['in.fant deaths'].plot(kind='bar', color='black')
plt.title('Mortalidade Infantil por Status de Desenvolvimento', fontsize=10)
plt.xlabel('Status')
plt.ylabel('Mortalidade Infantil')
# Gráfico de barras para a escolaridade
plt.subplot(133)  # 1 linha, 3 colunas, posição 3
summary_stats['Schooling'].plot(kind='bar', color='teal')
plt.title('Escolaridade por Status de Desenvolvimento', fontsize=10)
plt.xlabel('Status')
plt.ylabel('Escolaridade')
plt.tight_layout()
plt.show()

#Mortes Infantis: Aprofunde-se na saúde infantil com o número de mortes infantis por 1.000 nascidos vivos.
#Escolaridade: Meça a dádiva de conhecimento com dados sobre a média de anos de escolaridade.
#Status: Entenda o status de desenvolvimento, seja “Desenvolvido” ou “Em Desenvolvimento”, que molda o curso da saúde.


#######################################################################################################################
#ANALISE SOBRE HIV

paises_europa = ['Albania', 'Austria', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom']
paises_africa = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe']
df_europa = df[df['Country'].isin(paises_europa)]
df_africa = df[df['Country'].isin(paises_africa)]
#Separação
HIV_Country_EU = df_europa.groupby('Country')[' HIV/AIDS'].mean()
HIV_Country_AF = df_africa.groupby('Country')[' HIV/AIDS'].mean()
HIV_Country_EU = HIV_Country_EU.sort_values(ascending=False)
HIV_Country_AF = HIV_Country_AF.sort_values(ascending=False)
HIV_Country_EU = HIV_Country_EU.head()
HIV_Country_AF = HIV_Country_AF.head()
print(HIV_Country_AF.head(), HIV_Country_EU.head())
#GRAFICO
fig, ax = plt.subplots(figsize=(10, 6))
bars_EU = ax.bar(HIV_Country_EU.index, HIV_Country_EU, label='Europa', color='teal')
bars_AF = ax.bar(HIV_Country_AF.index, HIV_Country_AF, label='África', color='black')
ax.set_xlabel('País')
ax.set_ylabel('Média de HIV/AIDS')
ax.set_title('Média de HIV/AIDS nos Países da Europa e da África')
ax.legend()
plt.xticks(rotation=45, ha='right')
for bar in bars_EU + bars_AF:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                textcoords="offset points", ha='center', va='bottom')
plt.tight_layout()
plt.show()
# VIH/SIDA: Testemunhe a prevalência do VIH/SIDA como percentagem da população.
#########################################################################################################

























