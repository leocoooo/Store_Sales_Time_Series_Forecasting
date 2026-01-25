"""
Benchmark de modèles naïfs de prévision des ventes (MOSEF-compatible)

- Modèles simples / saisonniers / moyennes :
  utilisent des lags / rolling pré-calculés par (store, family)
- Si compute_missing_features=True :
  les lags / rolling manquants sont calculés automatiquement
- Modèle Naive_Drift :
  implémenté STRICTEMENT selon la formule du cours :
    ŷ_{T+h|T} = y_T + h * (y_T - y_1) / (T - 1)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import mean_absolute_error as mae


class NaiveBenchmark:

    # ==============================================================
    # INIT
    # ==============================================================
    def __init__(self, sales_df: pd.DataFrame, compute_missing_features: bool = False):
        self.sales_df = sales_df.copy()
        self.sales_df["date"] = pd.to_datetime(self.sales_df["date"])
        self.sales_df = self.sales_df.sort_values(
            ["store_nbr", "family", "date"]
        ).reset_index(drop=True)

        # Modèles basés sur colonnes (lags / rollings)
        self.column_models = {
            "Naive_Simple": "sales_lag_1",
            "Naive_Seasonal_Weekly": "sales_lag_7",
            "Naive_Seasonal_Yearly": "sales_lag_365",
            "Naive_Mean_7d": "rolling_mean_7",
            "Naive_Mean_30d": "rolling_mean_30",
            "Naive_Mean_90d": "rolling_mean_90",
        }

        # Modèles basés sur historique (théoriques)
        self.hist_models = {
            "Naive_Drift": self._naive_drift
        }

        self._validate_base_columns()

        if compute_missing_features:
            self._compute_missing_features()

        self._activate_models()

    # ==============================================================
    # VALIDATION
    # ==============================================================
    def _validate_base_columns(self):
        required = {"date", "store_nbr", "family", "sales"}
        missing = required - set(self.sales_df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans sales_df: {missing}")

    def _activate_models(self):
        """Active uniquement les modèles possibles"""
        self.models = {}

        for model, col in self.column_models.items():
            if col in self.sales_df.columns:
                self.models[model] = col

        # Le drift est toujours activable
        self.models.update(self.hist_models)

        print(f"[INFO] Modèles actifs: {list(self.models.keys())}")

    # ==============================================================
    # CONSTRUCTION DES FEATURES (OPTIONNEL)
    # ==============================================================
    def _compute_missing_features(self):
        """
        Calcule uniquement les lags / rollings manquants
        (groupby store_nbr, family + shift).
        """
        df = self.sales_df
        g = df.groupby(["store_nbr", "family"], sort=False)["sales"]

        # ---- Lags
        if "sales_lag_1" not in df.columns:
            df["sales_lag_1"] = g.shift(1)
        if "sales_lag_7" not in df.columns:
            df["sales_lag_7"] = g.shift(7)
        if "sales_lag_365" not in df.columns:
            df["sales_lag_365"] = g.shift(365)

        # ---- Rolling means (toujours shift(1) pour être causal)
        if "rolling_mean_7" not in df.columns:
            df["rolling_mean_7"] = g.shift(1).transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
        if "rolling_mean_30" not in df.columns:
            df["rolling_mean_30"] = g.shift(1).transform(
                lambda x: x.rolling(30, min_periods=1).mean()
            )
        if "rolling_mean_90" not in df.columns:
            df["rolling_mean_90"] = g.shift(1).transform(
                lambda x: x.rolling(90, min_periods=1).mean()
            )

        self.sales_df = df

    # ==============================================================
    # MODELE DRIFT (EXACT COURS)
    # ==============================================================
    def _naive_drift(self, hist: pd.DataFrame, h: int) -> float:
        """
        Drift EXACT du cours MOSEF :
        ŷ_{T+h|T} = y_T + h * (y_T - y_1) / (T - 1)
        """
        T = len(hist)
        if T < 2:
            return hist["sales"].iloc[-1] if T > 0 else 0.0

        y_T = hist["sales"].iloc[-1]
        y_1 = hist["sales"].iloc[0]

        return max(0.0, y_T + h * (y_T - y_1) / (T - 1))

    # ==============================================================
    # EVALUATION D’UN MODELE
    # ==============================================================
    def evaluate_model(self, model_name: str, n_years: int, week_horizon: int) -> pd.DataFrame:
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non reconnu")

        data_start = self.sales_df["date"].min()
        train_end = data_start + pd.Timedelta(days=365 * n_years)

        test_start = train_end + pd.Timedelta(days=1)
        test_end = train_end + pd.Timedelta(days=week_horizon * 7)

        train_df = self.sales_df[self.sales_df["date"] <= train_end]
        test_df = self.sales_df[
            (self.sales_df["date"] >= test_start) &
            (self.sales_df["date"] <= test_end)
        ]

        if test_df.empty:
            return pd.DataFrame()

        # --------------------------------------------------
        # CAS 1 : modèles à base de colonnes
        # --------------------------------------------------
        if model_name in self.column_models:
            col = self.column_models[model_name]
            tmp = test_df.copy()
            tmp["predicted"] = tmp[col].fillna(0.0)

            out = tmp[["date", "store_nbr", "family"]].copy()
            out["actual"] = tmp["sales"].values
            out["predicted"] = np.maximum(0.0, tmp["predicted"].values)
            return out

        # --------------------------------------------------
        # CAS 2 : modèle DRIFT (hist + h)
        # --------------------------------------------------
        preds = []
        model_func = self.hist_models[model_name]

        for (store, family), test_grp in test_df.groupby(["store_nbr", "family"]):
            hist = train_df[
                (train_df["store_nbr"] == store) &
                (train_df["family"] == family)
            ].sort_values("date")

            if hist.empty:
                continue

            for _, row in test_grp.iterrows():
                h = (row["date"] - train_end).days
                pred = model_func(hist, h)

                preds.append({
                    "date": row["date"],
                    "store_nbr": store,
                    "family": family,
                    "actual": row["sales"],
                    "predicted": pred
                })

        return pd.DataFrame(preds)

    # ==============================================================
    # EVALUATION GLOBALE
    # ==============================================================
    def evaluate_all_configurations(
        self,
        train_years: List[int] = [1, 2, 3, 4],
        week_horizons: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, Dict[Tuple[int, int], pd.DataFrame]]:

        all_results = {}

        for model_name in self.models:
            model_results = {}

            for n_years in train_years:
                for w in week_horizons:
                    df_pred = self.evaluate_model(model_name, n_years, w)
                    if not df_pred.empty:
                        model_results[(n_years, w)] = df_pred

            all_results[model_name] = model_results

        return all_results

    # ==============================================================
    # METRICS
    # ==============================================================
    @staticmethod
    def rmsle(y_true, y_pred) -> float:
        y_true = np.maximum(np.array(y_true), 0)
        y_pred = np.maximum(np.array(y_pred), 0)
        return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))

    # ==============================================================
    # REPORTING
    # ==============================================================
    def get_comparison_table(self, results: Dict) -> pd.DataFrame:
        rows = []

        for model, model_results in results.items():
            for (n_years, w), df in model_results.items():
                rows.append({
                    "Modèle": model,
                    "Train": f"{n_years}a",
                    "Horizon": f"w+{w}",
                    "Jours": w * 7,
                    "MAE": round(mae(df["actual"], df["predicted"]), 2),
                    "RMSLE": round(self.rmsle(df["actual"], df["predicted"]), 4)
                })

        return pd.DataFrame(rows)

    def get_global_ranking(self, results: Dict) -> pd.DataFrame:
        rows = []

        for model, model_results in results.items():
            maes, rmsles = [], []

            for df in model_results.values():
                maes.append(mae(df["actual"], df["predicted"]))
                rmsles.append(self.rmsle(df["actual"], df["predicted"]))

            if maes:
                rows.append({
                    "Modèle": model,
                    "MAE_moyen": np.mean(maes),
                    "RMSLE_moyen": np.mean(rmsles)
                })

        return pd.DataFrame(rows).sort_values("MAE_moyen").reset_index(drop=True)
