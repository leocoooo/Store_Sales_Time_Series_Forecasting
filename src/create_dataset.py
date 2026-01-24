import pandas as pd
import numpy as np

def create_dataset(train_df, stores, oil, holidays):
    df = train_df.copy()

    # Nettoyage et préparation des données externes
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    oil = oil.set_index('date').reindex(all_dates).ffill().bfill().reset_index()
    oil.columns = ['date', 'dcoilwtico']

    # On se passe des jours fériés pour l'instant car on capte tout avec la variable is_store_closed, 
    # et surtout le merging n'est pas trivial à cause des fêtes locales / regionales / nationales différentes, 
    # on n'a pas toujours les clé ect.

    # holidays = holidays[holidays['transferred'] == False].copy()
    # holidays = holidays.rename(columns={'type': 'holiday_type'})
    stores = stores.rename(columns={'type': 'store_type'})

    # Fusions des tables
    df = df.merge(stores, on='store_nbr', how='left')
    df = df.merge(oil, on='date', how='left')
    # df = df.merge(holidays, on='date', how='left')

    # Gestion des jours fériés et fermetures
    # df['is_holiday_local'] = ((df['locale'] == 'Local') & (df['locale_name'] == df['city'])).astype(int)
    # df['is_holiday_regional'] = ((df['locale'] == 'Regional') & (df['locale_name'] == df['state'])).astype(int)
    # df['is_holiday_national'] = (df['locale'] == 'National').astype(int)
    # df['holiday_type'] = df['holiday_type'].fillna('Normal Day')
    
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
    
    # Pétrole : Tendance avec backfill pour les premières valeurs
    df['oil_trend_30'] = df['dcoilwtico'] - df['dcoilwtico'].shift(30).bfill()

    # Promotions 
    promo_avg = df.groupby(['store_nbr', 'family'])['onpromotion'].transform('mean')
    df['promo_ratio_vs_avg'] = df['onpromotion'] / (promo_avg + 1)

    # Interactions
    df['promo_during_payday'] = df['onpromotion'] * df['is_payday']
    df["family_store_nb"] = df["family"].astype(str) + "_" + df["store_nbr"].astype(str)
    df["family_state"] = df["family"].astype(str) + "_" + df["state"].astype(str)
    df["family_onpromotion"] = df["family"].astype(str) + "_" + df["onpromotion"].astype(str)

    # Calculer la moyenne de ventes par jour pour chaque famille (sur tout le dataset)
    family_volumes = df.groupby('family')['sales'].mean()

    # Calculer les seuils dynamiques via les quartiles
    q1 = family_volumes.quantile(0.25) # Les 25% plus petites familles
    q3 = family_volumes.quantile(0.75) # Les 25% plus grosses familles
    
    print(f"Seuils calculés : Q1={q1:.2f} (14j), Q3={q3:.2f} (3j)")

    df['is_family_unactive'] = 0
    # Adapter dynamiquement la fenêtre de détection d'inactivité selon le volume de ventes moyen
    for fam in df['family'].unique():
        avg_sales = family_volumes[fam]
        
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

    # Nettoyage final
    cols_to_drop = ['locale', 'locale_name', 'description', 'transferred']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

def optimize_memory(df):
    """
    Parcourt toutes les colonnes d'un dataframe et modifie leur type 
    pour réduire l'empreinte mémoire sans perte de données.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # On ne touche pas aux types 'object' (on peut les passer en category plus tard si besoin)
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Cas des entiers
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            
            # Cas des flottants
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
        # Transformer les chaînes de caractères répétitives en catégories
        elif col_type == object:
            if df[col].nunique() / len(df) > 0.50:
                continue  # Si trop de valeurs uniques, on ne convertit pas
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Mémoire initiale: {start_mem:.2f} MB')
    print(f'Mémoire finale: {end_mem:.2f} MB')
    print(f'Réduction de: {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df