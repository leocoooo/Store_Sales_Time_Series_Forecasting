import pandas as pd

# Chargement des datasets 
main_df = pd.read_csv('data/main_df.csv', parse_dates=['date'])
stores = pd.read_csv('data/stores.csv')
oil = pd.read_csv('data/oil.csv', parse_dates=['date'])
holidays = pd.read_csv('data/holidays_events.csv', parse_dates=['date'])

# Gestion du pétrole : Remplir les dates manquantes (week-ends) et les NaN
all_dates = pd.date_range(start=main_df['date'].min(), end=main_df['date'].max())
oil = oil.set_index('date').reindex(all_dates).fillna(method='ffill').fillna(method='bfill').reset_index()
oil.columns = ['date', 'dcoilwtico']

# Nettoyage des Jours Fériés (gestion de la colonne 'transferred')
# On ne garde que les jours qui ont réellement été célébrés
holidays = holidays[holidays['transferred'] == False]
holidays = holidays.drop_duplicates(subset=['date']).set_index('date')
holidays = holidays[['type', 'locale', 'locale_name', 'description']]
holidays.columns = ['holiday_type', 'holiday_locale', 'holiday_locale_name', 'holiday_description']

# Fusion avec les magasins
df = main_df.merge(stores, on='store_nbr', how='left')

# Fusion avec le pétrole
df = df.merge(oil, on='date', how='left')

# Fusion avec les jours fériés
df = df.merge(holidays, left_on='date', right_index=True, how='left')

# Feature des salaires (Pay day 15 et dernier jour du mois)
df['is_payday'] = ((df['date'].dt.day == 15) | (df['date'].dt.is_month_end)).astype(int)

# Feature Séisme (16 Avril 2016)
# On crée une feature binaire ou une décroissance sur 4 semaines après le séisme
earthquake_date = pd.to_datetime('2016-04-16')
df['is_earthquake_impact'] = ((df['date'] >= earthquake_date) & 
                              (df['date'] <= earthquake_date + pd.Timedelta(weeks=4))).astype(int)

# Remplir les valeurs nulles pour les jours non fériés
df['holiday_type'] = df['holiday_type'].fillna('Work Day')

print(f"Taille dataset: {df.shape}")
print(df.head())