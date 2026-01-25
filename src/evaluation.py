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