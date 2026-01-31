import numpy as np

def evaluate_model(y_true, y_pred, is_store_closed, is_family_unactive):
    """
    Evalue la performance du modele en ignorant les periodes de fermeture 
    ou d'inactivite produit via un systeme de poids.
    """
    
    # Gestion des valeurs negatives : la vente ne peut pas etre inferieure a 0
    # On force y_pred à 0 pour toutes les valeurs negatives
    y_pred_clipped = np.maximum(y_pred, 0)
    
    # Definition du masque de poids (Weight Mask)
    # Poids = 0 si (ferme OU inactif), sinon 1
    weights = ((is_store_closed == 0) & (is_family_unactive == 0)).astype(int)
    
    # Nombre d'observations reellement prises en compte
    n_active = np.sum(weights)
    if n_active == 0:
        raise ValueError("Aucune observation active pour l'evaluation du modele.")
    
    # Calcul du MAE pondere
    # On multiplie l'erreur absolue par le poids, puis on divise par le nombre d'actifs
    weighted_mae = np.sum(np.abs(y_true - y_pred_clipped) * weights) / n_active
    
    # Calcul du RMSE pondere
    # On multiplie l'erreur au carre par le poids
    weighted_mse = np.sum(((y_true - y_pred_clipped) ** 2) * weights) / n_active
    weighted_rmse = np.sqrt(weighted_mse)
    
    # Calcul du RMSLE : utilise log(1+p) pour stabiliser les gros volumes
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred_clipped)
    weighted_rmsle = np.sqrt(np.sum(((log_true - log_pred) ** 2) * weights) / n_active)

    print(f"--- Evaluation (sur {n_active} points actifs) ---")
    print(f"Weighted MAE   : {weighted_mae:.4f}")
    print(f"Weighted RMSE  : {weighted_rmse:.4f}")
    print(f"Weighted RMSLE : {weighted_rmsle:.4f}")
    
    return {
        "mae": weighted_mae,
        "rmse": weighted_rmse,
        "rmsle": weighted_rmsle,
    }


def time_series_expanding_window(df, n_test_days=56, n_splits=5):
    """
    Genere des indices pour une validation croisee en fenetre croissante.
    
    Arguments:
        df: DataFrame complet (doit contenir une colonne 'date')
        n_test_days: Nombre de jours dans chaque fenetre de test (ex: 56 jours / 8 semaines)
        n_splits: Nombre de segments de validation souhaites
        
    Retourne:
        Une liste de tuples (train_index, val_index)
    """
    print(f"Preparation de la validation croisee : {n_splits} splits de {n_test_days} jours.")
    
    # Recuperation des dates uniques et triees
    unique_dates = np.sort(df['date'].unique())
    total_days = len(unique_dates)
    
    # Verification de la faisabilite
    required_days = n_splits * n_test_days
    if required_days >= total_days:
        raise ValueError(f"Pas assez de donnees pour {n_splits} splits de {n_test_days} jours.")

    splits = []
    
    # Generation des fenetres en partant de la fin (du plus récent au plus ancien)
    for i in range(n_splits):
        # Calcul des positions des dates
        # Fin du test = fin des donnees moins les splits deja calcules
        end_test_idx = total_days - (i * n_test_days)
        start_test_idx = end_test_idx - n_test_days
        
        # Le train contient tout ce qui precede le debut du test actuel
        train_dates = unique_dates[:start_test_idx]
        val_dates = unique_dates[start_test_idx:end_test_idx]
        
        # On recupere les indices correspondants dans le dataframe original
        train_indices = df[df['date'].isin(train_dates)].index
        val_indices = df[df['date'].isin(val_dates)].index
        
        # On stocke (Train, Val) - On les insere au debut pour garder l'ordre chronologique
        splits.insert(0, (train_indices, val_indices))
        
        print(f"Split {n_splits - i}: Train jusqu'au {train_dates[-1].date()}, "
              f"Val du {val_dates[0].date()} au {val_dates[-1].date()}")

    return splits