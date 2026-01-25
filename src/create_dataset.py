import pandas as pd
import numpy as np



import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

class DatasetProcessor:
    def __init__(self, stores, oil, holidays):
        self.stores = stores
        self.oil = oil
        self.holidays = holidays
        self.family_stats = None
        self.q1 = None
        self.q3 = None
        # On stocke l'historique du train pour les futurs lags du test
        self.train_history = None 
        
    def _prepare_oil(self, start_date, end_date):
        print(f"Préparation des données du pétrole du {start_date.date()} au {end_date.date()}...")
        all_dates = pd.date_range(start=start_date, end=end_date)
        oil_clean = self.oil.copy()
        oil_clean = oil_clean.set_index('date').reindex(all_dates).ffill().bfill().reset_index()
        oil_clean.columns = ['date', 'oil_price']
        
        oil_clean['oil_price_lag_8w'] = oil_clean['oil_price'].shift(56)
        oil_clean['oil_price_lag_8w_trend_30'] = oil_clean['oil_price_lag_8w'] - oil_clean['oil_price_lag_8w'].shift(30)
        
        oil_clean['oil_price_lag_8w'] = oil_clean['oil_price_lag_8w'].bfill()
        oil_clean['oil_price_lag_8w_trend_30'] = oil_clean['oil_price_lag_8w_trend_30'].bfill()
        return oil_clean

    def _handle_holidays(self, df):
        print("Traitement des jours fériés et priorités géographiques...")
        holidays_clean = self.holidays[self.holidays['transferred'] == False].copy()
        holidays_clean = holidays_clean.rename(columns={'type': 'holiday_type'})
        
        df = df.merge(holidays_clean, on='date', how='left')
        
        df['is_holiday_local'] = ((df['locale'] == 'Local') & (df['locale_name'] == df['city'])).astype(int)
        df['is_holiday_regional'] = ((df['locale'] == 'Regional') & (df['locale_name'] == df['state'])).astype(int)
        df['is_holiday_national'] = (df['locale'] == 'National').astype(int)
        
        priority_map = {'National': 3, 'Regional': 2, 'Local': 1}
        df['holiday_priority'] = df['locale'].map(priority_map).fillna(0)
        
        mask_not_relevant = (
            ((df['locale'] == 'Local') & (df['locale_name'] != df['city'])) |
            ((df['locale'] == 'Regional') & (df['locale_name'] != df['state']))
        )
        df.loc[mask_not_relevant, 'holiday_priority'] = 0
        
        df = df.sort_values(['date', 'store_nbr', 'family', 'holiday_priority'], ascending=[True, True, True, False])
        df = df.drop_duplicates(subset=['date', 'store_nbr', 'family'], keep='first')
        
        df.loc[df['holiday_priority'] == 0, 'holiday_type'] = 'Normal Day'
        is_work_day = (df['holiday_type'] == 'Work Day')
        df.loc[is_work_day, ['is_holiday_local', 'is_holiday_regional', 'is_holiday_national']] = 0
        df.loc[is_work_day, 'holiday_type'] = 'Normal Day'
        
        return df

    def _add_time_features(self, df):
        print("Calcul des variables temporelles et cycliques...")
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_payday'] = ((df['date'].dt.day == 15) | (df['date'].dt.is_month_end)).astype(int)
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        earthquake_date = pd.to_datetime('2016-04-16')
        df['is_earthquake_impact'] = ((df['date'] >= earthquake_date) & 
                                      (df['date'] <= earthquake_date + pd.Timedelta(weeks=4))).astype(int)
        
        df['days_to_christmas'] = (pd.to_datetime(df['date'].dt.year.astype(str) + '-12-25') - df['date']).dt.days
        df.loc[df['days_to_christmas'] < 0, 'days_to_christmas'] += 365
        return df

    def _add_lags_and_rolling(self, df):
        print("Génération des lags et fenêtres glissantes...")
        df = df.sort_values(['store_nbr', 'family', 'date'])
        grouped = df.groupby(['store_nbr', 'family'])['sales']
        
        for i in [1, 2, 3, 7, 14, 28, 56, 364]:
            df[f'sales_lag_{i}'] = grouped.shift(i)
            
        for i in [7, 28]:
            df[f'rolling_mean_{i}'] = grouped.shift(1).transform(lambda x: x.rolling(window=i).mean())
            
        df["log_sales"] = np.log1p(df["sales"])
        df['rolling_std_7'] = grouped.shift(1).transform(lambda x: x.rolling(window=7).std())
        return df

    def _handle_inactivity(self, df):
        print("Analyse de la mobilité des stocks et inactivité...")
        store_daily_total = df.groupby(['store_nbr', 'date'])['sales'].transform('sum')
        df['is_store_closed'] = (store_daily_total == 0).astype(int)
        
        df['is_family_unactive'] = 0
        for fam in df['family'].unique():
            avg_sales = self.family_stats.get(fam, 0)
            w = 3 if avg_sales >= self.q3 else (14 if avg_sales <= self.q1 else 7)
                
            mask = df['family'] == fam
            df.loc[mask, 'is_family_unactive'] = df[mask].groupby('store_nbr')['sales'].transform(
                lambda x: (x.shift(1).rolling(window=w, min_periods=1).sum() == 0).astype(int)
            )
        
        df['is_family_unactive'] = ((df['is_family_unactive'] == 1) & (df['is_store_closed'] == 0)).astype(int)
        df['is_store_closed_lag_1'] = df.groupby('store_nbr')['is_store_closed'].shift(1).fillna(0)
        df['is_family_unactive_lag_1'] = df.groupby(['store_nbr', 'family'])['is_family_unactive'].shift(1).fillna(0)
        
        return df
    
    def _optimize_memory(self, df):
        print("Optimisation de la memoire en cours...")
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                        
            elif col_type == object:
                if df[col].nunique() < 100: # Seuil pour passer en categorie
                    df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Reduction memoire: {start_mem:.1f}MB -> {end_mem:.1f}MB ({100*(start_mem-end_mem)/start_mem:.1f}%)')
        return df

    def fit(self, train_df):
        print("Calcul des statistiques de référence sur le Train...")
        self.family_stats = train_df.groupby('family')['sales'].mean()
        self.q1 = self.family_stats.quantile(0.25)
        self.q3 = self.family_stats.quantile(0.75)
        
        # On sauvegarde les 365 derniers jours pour les futurs lags du test
        last_date = train_df['date'].max()
        self.train_history = train_df[train_df['date'] > (last_date - pd.Timedelta(days=365))].copy()
        
        print(f"Stats enregistrées pour {len(self.family_stats)} familles.")
        return self

    def transform(self, input_df, is_test=False):

        # Si c'est le test, on ajoute l'historique du train pour les lags
        if is_test:
            print("\nMode TEST détecté") 
            print("Nombre de lignes avant traitement :", len(input_df))
            print("Ajout de l'historique du train pour les lags...")
            df = pd.concat([self.train_history, input_df], axis=0).reset_index(drop=True)
        else:
            print("\nMode TRAIN activé")
            print("Nombre de lignes avant traitement :", len(input_df))
            df = input_df.copy()
        
        oil_data = self._prepare_oil(df['date'].min(), df['date'].max())
        stores_data = self.stores.rename(columns={'type': 'store_type'})
        
        df = df.merge(stores_data, on='store_nbr', how='left')
        df = df.merge(oil_data, on='date', how='left')
        
        df = self._handle_holidays(df)
        df = self._add_time_features(df)
        df = self._add_lags_and_rolling(df)
        df = self._handle_inactivity(df)
        
        # Si c'est le test, on retire les lignes d'historique apres calcul
        if is_test:
            cutoff_date = input_df['date'].min()
            df = df[df['date'] >= cutoff_date].copy()
            print("Suppression de l'historique de préchauffage terminée.")

        cols_to_drop = ['locale', 'locale_name', 'description', 'transferred', 'holiday_priority']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        print("Nombre de lignes après traitement :", len(df))
        
        return self._optimize_memory(df)
    

def create_full_dataset(train_df, stores, oil, holidays, family_stats=None):
    df = train_df.copy()

    # Nettoyage et préparation des données externes
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    oil = oil.set_index('date').reindex(all_dates).ffill().bfill().reset_index()
    oil.columns = ['date', 'oil_price']
    oil['oil_price_lag_8w'] = oil['oil_price'].shift(56)
    oil['oil_price_lag_8w_trend_30'] = oil['oil_price_lag_8w'] - oil['oil_price_lag_8w'].shift(30)
    # On bfill les 56 + 30 premiers jours qui sont à NaN suite aux shifts
    oil['oil_price_lag_8w'] = oil['oil_price_lag_8w'].bfill()
    oil['oil_price_lag_8w_trend_30'] = oil['oil_price_lag_8w_trend_30'].bfill()

    holidays = holidays[holidays['transferred'] == False].copy()
    holidays = holidays.rename(columns={'type': 'holiday_type'})
    stores = stores.rename(columns={'type': 'store_type'})

    # Fusions des tables
    df = df.merge(stores, on='store_nbr', how='left')
    df = df.merge(oil, on='date', how='left')
    df = df.merge(holidays, on='date', how='left')

    # Gestion des jours fériés et fermetures
    df['is_holiday_local'] = ((df['locale'] == 'Local') & (df['locale_name'] == df['city'])).astype(int)
    df['is_holiday_regional'] = ((df['locale'] == 'Regional') & (df['locale_name'] == df['state'])).astype(int)
    df['is_holiday_national'] = (df['locale'] == 'National').astype(int)

    # score de priorité pour gérer les doublons (ex: Noël + Fête locale)
    # Plus le score est haut, plus l'événement est "fort"
    priority_map = {'National': 3, 'Regional': 2, 'Local': 1}
    df['holiday_priority'] = df['locale'].map(priority_map).fillna(0)

    # On annule la priorité si l'événement ne concerne pas le magasin
    # Si c'est une fête locale à Quito mais qu'on est à Guayaquil, priorité = 0
    mask_not_relevant = (
        ((df['locale'] == 'Local') & (df['locale_name'] != df['city'])) |
        ((df['locale'] == 'Regional') & (df['locale_name'] != df['state']))
    )
    df.loc[mask_not_relevant, 'holiday_priority'] = 0

    # On trie par date/store/family ET par priorité décroissante
    df = df.sort_values(['date', 'store_nbr', 'family', 'holiday_priority'], ascending=[True, True, True, False])

    # On ne garde que la première ligne (la plus prioritaire) pour chaque combo unique
    df = df.drop_duplicates(subset=['date', 'store_nbr', 'family'], keep='first')

    # Maintenant on peut nettoyer le holiday_type sans peur
    # Si priorité est 0, c'est un jour normal pour CE magasin
    df.loc[df['holiday_priority'] == 0, 'holiday_type'] = 'Normal Day'

    # Si c'est un jour de travail (rattrapage), on annule tous les flags de vacances
    is_work_day = (df['holiday_type'] == 'Work Day')

    df.loc[is_work_day, 'is_holiday_local'] = 0
    df.loc[is_work_day, 'is_holiday_regional'] = 0
    df.loc[is_work_day, 'is_holiday_national'] = 0

    # On peut changer le holiday_type en 'Normal Day' 
    # pour que l'embedding ne soit pas pollué
    df.loc[is_work_day, 'holiday_type'] = 'Normal Day'
    
    # Détection des fermetures réelles (si toutes les familles du magasin vendent 0)
    store_daily_total = df.groupby(['store_nbr', 'date'])['sales'].transform('sum')
    df['is_store_closed'] = (store_daily_total == 0).astype(int)

    # Features temporelles et cycliques
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['time_idx'] = (df['date'] - df['date'].min()).dt.days
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_payday'] = ((df['date'].dt.day == 15) | (df['date'].dt.is_month_end)).astype(int)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Événements spéciaux
    earthquake_date = pd.to_datetime('2016-04-16')
    df['is_earthquake_impact'] = ((df['date'] >= earthquake_date) & 
                                  (df['date'] <= earthquake_date + pd.Timedelta(weeks=4))).astype(int)
    
    df['days_to_christmas'] = (pd.to_datetime(df['date'].dt.year.astype(str) + '-12-25') - df['date']).dt.days
    df.loc[df['days_to_christmas'] < 0, 'days_to_christmas'] += 365

    # Tri et calculs basés sur l'historique (Groupby Store/Family)
    df = df.sort_values(['store_nbr', 'family', 'date'])
    grouped = df.groupby(['store_nbr', 'family'])['sales']

    # Lags et Fenêtres glissantes (Décalés de 1 pour éviter le leakage)
    for i in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 56, 364]:
        df[f'sales_lag_{i}'] = grouped.shift(i)

    for i in [7, 28, 56]:
        df[f'rolling_mean_{i}'] = grouped.shift(1).transform(lambda x: x.rolling(window=i).mean())

    df["log_sales"] = np.log1p(df["sales"]) # log transformation
    df["diff_sales"] = grouped.shift(1).transform(lambda x: x.diff()) # differenciation

    df['rolling_std_7'] = grouped.shift(1).transform(lambda x: x.rolling(window=7).std())
    df['store_family_velocity'] = grouped.shift(1).transform(lambda x: x.rolling(window=30, min_periods=1).mean())

    # Promotions 
    promo_avg = df.groupby(['store_nbr', 'family'])['onpromotion'].transform('mean')
    df['promo_ratio_vs_avg'] = df['onpromotion'] / (promo_avg + 1)

    # Interactions
    df['promo_during_payday'] = df['onpromotion'] * df['is_payday']
    df["family_store_nb"] = df["family"].astype(str) + "_" + df["store_nbr"].astype(str)
    df["family_state"] = df["family"].astype(str) + "_" + df["state"].astype(str)

    df["is_on_promo"] = df['onpromotion'] > 0
    df["family_onpromotion"] = df["family"].astype(str) + "_" + df["is_on_promo"].astype(str)


    if family_stats is None:
            # Cas du TRAIN : on calcule les statistiques
            family_stats = df.groupby('family')['sales'].mean()

    # Calculer la moyenne de ventes par jour pour chaque famille (sur tout le dataset)
    df['family_avg_sales'] = df['family'].map(family_stats)

    # Calculer les seuils dynamiques via les quartiles
    q1 = family_stats.quantile(0.25) # Les 25% plus petites familles
    q3 = family_stats.quantile(0.75) # Les 25% plus grosses familles
    
    print(f"Seuils calculés : Q1={q1:.2f} (14j), Q3={q3:.2f} (3j)")

    df['is_family_unactive'] = 0
    # Adapter dynamiquement la fenêtre de détection d'inactivité selon le volume de ventes moyen
    for fam in df['family'].unique():
        avg_sales = family_stats.get(fam, 0)
        
        if avg_sales >= q3:
            w = 3   # Haute rotation : très réactif
        elif avg_sales <= q1:
            w = 14  # Faible rotation : très tolérant
        else:
            w = 7   # Standard

        print(f"Famille '{fam}': avg_sales={avg_sales:.2f} -> fenêtre d'inactivité={w} jours")
            
        # Application du flag pour cette famille précise
        mask = df['family'] == fam
        df.loc[mask, 'is_family_unactive'] = df[mask].groupby('store_nbr')['sales'].transform(
            lambda x: (x.shift(1).rolling(window=w, min_periods=1).sum() == 0).astype(int)
        )

    # On ne marque "unactive" que si le magasin a eu des ventes (ouvert) mais que la famille n'en a pas eu.
    df['is_family_unactive'] = ((df['is_family_unactive'] == 1) & (df['is_store_closed'] == 0)).astype(int)

    # Ces deux lignes ssuivantes permettent de capturer le contexte du jour précédent pour gérer les jours fériés 
    # par exemple, pour que le modèle comprenne mieux les lags montrant des ventes nulles la veille. 
    # Contexte de fermeture du magasin 
    df['is_store_closed_lag_1'] = df.groupby('store_nbr')['is_store_closed'].shift(1).fillna(0)
    # Contexte d'inactivité spécifique à la famille (rupture de stock / arrêt catalogue)
    df['is_family_unactive_lag_1'] = df.groupby(['store_nbr', 'family'])['is_family_unactive'].shift(1).fillna(0)

    # Nettoyage final
    cols_to_drop = ['locale', 'locale_name', 'description', 'transferred', "holiday_priority"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df, family_stats


