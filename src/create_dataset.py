import pandas as pd
import numpy as np

def create_dataset(train_df, stores, oil, holidays):
    df = train_df.copy()

    # Oil
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    oil = oil.set_index('date').reindex(all_dates).ffill().bfill().reset_index()
    oil.columns = ['date', 'dcoilwtico']

    # Holidays
    holidays = holidays[holidays['transferred'] == False].copy()
    holidays = holidays.drop_duplicates(subset=['date'])
    holidays = holidays.rename(columns={'type': 'holiday_type'})

    # 3. Stores 
    stores = stores.rename(columns={'type': 'store_type'})

    # 4. Merges
    df = df.merge(stores, on='store_nbr', how='left')
    df = df.merge(oil, on='date', how='left')
    df = df.merge(holidays, on='date', how='left')

    # Raffinement des Jours Fériés
    df['is_holiday_local'] = ((df['locale'] == 'Local') & (df['locale_name'] == df['city'])).astype(int)
    df['is_holiday_regional'] = ((df['locale'] == 'Regional') & (df['locale_name'] == df['state'])).astype(int)
    df['is_holiday_national'] = (df['locale'] == 'National').astype(int)
    
    # On remplit les valeurs pour les jours normaux
    df['holiday_type'] = df['holiday_type'].fillna('Work Day')

    # Features Calendaires
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Coordonnées Circulaires
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Autres
    df['is_payday'] = ((df['date'].dt.day == 15) | (df['date'].dt.is_month_end)).astype(int)
    
    earthquake_date = pd.to_datetime('2016-04-16')
    df['is_earthquake_impact'] = ((df['date'] >= earthquake_date) & 
                                  (df['date'] <= earthquake_date + pd.Timedelta(weeks=4))).astype(int)
    
    df['days_to_christmas'] = (pd.to_datetime(df['date'].dt.year.astype(str) + '-12-25') - df['date']).dt.days
    df.loc[df['days_to_christmas'] < 0, 'days_to_christmas'] += 365

    # Promos & Interactions
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    # Promo Ratio
    promo_avg = df.groupby(['store_nbr', 'family'])['onpromotion'].transform('mean')
    df['promo_ratio_vs_avg'] = df['onpromotion'] / (promo_avg + 1)

    # Lags et Rolling de la target
    grouped = df.groupby(['store_nbr', 'family'])['sales']
    df['store_family_velocity'] = grouped.shift(1).transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['sales_lag_7'] = grouped.shift(7)
    df['rolling_mean_7'] = grouped.shift(1).transform(lambda x: x.rolling(window=7).mean())

    cols_to_drop = ['locale', 'locale_name', 'description', 'transferred']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df