import numpy as np
import pandas as pd
from scipy.special import softmax


class PatientRetrieverV4:
    """Refactored to accept pre-loaded models instead of loading from disk."""

    CATEGORICAL_COLS = [
        "X_stage_raw", "X_figo_version", "X_histology_type", "X_grade",
        "X_cervical_involvement", "X_menopause", "X_p53", "X_mmr",
        "X_molecular_subtype", "X_stage_2023", "X_esgo_risk_group",
        "X_myometrial_invasion_ratio", "X_lvsi", "X_peritoneal_cytology",
    ]
    NUMERICAL_COLS = ["X_age", "X_myometrial_invasion_depth"]
    COMORBIDITY_COLS = [
        "X_glycemic_status", "X_hypertension", "X_bmi_status",
        "X_hyperlipidemia", "X_anemia", "X_hepatic_viral",
        "X_hepatic_dysfunction", "X_major_cv_risk",
    ]
    OTHER_BINARY_COLS = ["X_lvsi_substantial", "X_adnexal_involvement"]
    WEIGHT_COLS = ["X_major_cv_risk", "X_hepatic_viral"]
    WEIGHT_MULTIPLIER = 2.0
    LABEL_NAMES = ["radiotherapy", "chemotherapy", "targeted_therapy", "immunotherapy", "hormone_therapy"]

    def __init__(self, preprocessor, kmeans, knn, df, X_vec, patient_ids, xgb_classifiers, thresholds):
        self.preprocessor = preprocessor
        self.kmeans = kmeans
        self.knn = knn
        self.df = df
        self.X_vec = X_vec
        self.patient_ids = patient_ids
        self.classifiers = xgb_classifiers
        self.thresholds = thresholds

    def _prepare_new_patient_df(self, new_patient_dict):
        expected_cols = self.preprocessor.feature_names_in_
        full_dict = {}
        for col in expected_cols:
            raw_name = col[2:]
            if raw_name in new_patient_dict:
                full_dict[col] = new_patient_dict[raw_name]
            else:
                if col in self.NUMERICAL_COLS:
                    full_dict[col] = np.nan
                elif col in self.COMORBIDITY_COLS or col in self.OTHER_BINARY_COLS:
                    full_dict[col] = 0
                else:
                    full_dict[col] = "unknown"

        df = pd.DataFrame([full_dict])
        for col in self.CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)
        for col in self.NUMERICAL_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        for col in self.COMORBIDITY_COLS + self.OTHER_BINARY_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        for col in self.WEIGHT_COLS:
            if col in df.columns:
                df[col] = df[col] * self.WEIGHT_MULTIPLIER
        for col in self.COMORBIDITY_COLS + self.OTHER_BINARY_COLS:
            if col in df.columns and col not in self.WEIGHT_COLS:
                df[col] = df[col].astype(int)
        return df

    def _transform_patient(self, patient_df):
        X_base = self.preprocessor.transform(patient_df)
        distances = self.kmeans.transform(X_base)
        X_soft = softmax(-distances, axis=1)
        return np.column_stack([X_base, X_soft])

    def retrieve(self, new_patient_dict, top_k=3):
        new_df = self._prepare_new_patient_df(new_patient_dict)
        new_vec = self._transform_patient(new_df)
        dist, idx = self.knn.kneighbors(new_vec, n_neighbors=top_k)
        results = self.df.iloc[idx[0]].copy()
        results["distance"] = dist[0]
        return results

    def predict(self, new_patient_dict):
        if self.classifiers is None:
            return {}
        new_df = self._prepare_new_patient_df(new_patient_dict)
        X_final = self._transform_patient(new_df)
        predictions = {}
        for label in self.LABEL_NAMES:
            if label not in self.classifiers:
                continue
            clf = self.classifiers[label]
            proba = float(clf.predict_proba(X_final)[:, 1][0])
            thr = self.thresholds.get(label, 0.5)
            predictions[label] = 1 if proba > thr else 0
        return predictions
