from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np

def apply_robust_scaler(df, exclude_cols=None, drop_original=True):
    df_scaled = df.copy()
    
    # Colonnes à ne pas scaler
    if exclude_cols is None:
        exclude_cols = [
            'id', 'date', 'store_nbr', 'family', 'city', 'state', 'store_type', 'cluster',
            'sales', 'log_sales', 'diff_sales', 
            'is_holiday_local', 'is_holiday_regional', 'is_holiday_national', 
            'is_store_closed', 'is_weekend', 'is_payday', 'is_earthquake_impact'
        ]
    
    # Colonnes numériques pertinentes à scaler
    numeric_cols = df_scaled.select_dtypes(include=['float32', 'float64', 'int16', 'int32']).columns
    cols_to_scale = [c for c in numeric_cols if c not in exclude_cols]
    
    # Application du Scaler avec création de nouvelles colonnes
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_scaled[cols_to_scale])
    new_col_names = [f"scaled_{col}" for col in cols_to_scale]
    df_scaled[new_col_names] = scaled_data
    
    if drop_original:
        df_scaled.drop(columns=cols_to_scale, inplace=True)
    
    print(f"{len(new_col_names)} colonnes créées avec préfixe 'scaled_' : {new_col_names}")
    return df_scaled, scaler


def apply_target_encoding(df, cols, target, nb_previous_days_to_ignore=56, fill_first_na=True):
    df_encoded = df.copy()
    for col in cols:
        # On calcule la moyenne glissante cumulative en ignorant les nb_previous_days_to_ignore derniers jours
        group = df_encoded.groupby(col)[target]
        cum_sum = group.transform(lambda x: x.shift(nb_previous_days_to_ignore).expanding().sum())
        cum_cnt = group.transform(lambda x: x.shift(nb_previous_days_to_ignore).expanding().count())
        
        df_encoded[f'target_enc_{col}'] = cum_sum / cum_cnt

        if fill_first_na:
            # Remplissage des NaN initiaux par la moyenne globale
            print(f"{df_encoded[f'target_enc_{col}'].isnull().sum()} NaN found in target_enc_{col} before filling.")
            df_encoded[f'target_enc_{col}'].fillna(df[target].mean(), inplace=True)
            print(f"Filled NaN in target_enc_{col} with global mean: {df[target].mean()}")
        
    return df_encoded


def apply_catboost_encoding(df, cols, target, alpha=None, nb_previous_days_to_ignore=56, fill_first_na=True):
    df_encoded = df.copy()
    global_mean = df[target].mean()

    for col in cols:
        # Calcul dynamique de alpha si non fourni
        if alpha is None:
            n_categories = df[col].nunique()
            # Règle empirique : alpha = log1p de la cardinalité * 5 
            # (Ex: 33 cats -> ~17 | 1782 cats -> ~37)
            current_alpha = np.log1p(n_categories) * 5
            print(f"Variable '{col}': {n_categories} categories -> Dynamic alpha: {current_alpha:.2f}")
        else:
            current_alpha = alpha

        group = df_encoded.groupby(col)[target]
        
        # On applique le décalage (Gap) pour éviter le leakage temporel
        cum_sum = group.transform(lambda x: x.shift(nb_previous_days_to_ignore).expanding().sum())
        cum_cnt = group.transform(lambda x: x.shift(nb_previous_days_to_ignore).expanding().count())
        
        # Formule CatBoost avec alpha dynamique
        col_name = f'catboost_enc_{col}'
        df_encoded[col_name] = (cum_sum + (current_alpha * global_mean)) / (cum_cnt + current_alpha)

        if fill_first_na:
            # Les NaN apparaissent au début (période < nb_previous_days_to_ignore)
            print(f"{df_encoded[f'catboost_enc_{col}'].isnull().sum()} NaN found in catboost_enc_{col} before filling.")
            df_encoded[col_name].fillna(global_mean, inplace=True)
            print(f"Filled NaN in catboost_enc_{col} with global mean: {global_mean}") 

    return df_encoded


