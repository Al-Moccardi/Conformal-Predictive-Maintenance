

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Required imports for Keras/TensorFlow
# -------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, BatchNormalization,
                                     Bidirectional, GRU, Conv1D, Flatten, Input)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

# -------------------------
# Other imports
# -------------------------
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# XGBoost
# -------------------------
from xgboost import XGBRegressor

# =============================================================================
# Define the custom S-score function
# =============================================================================
def compute_s_score(rul_true, rul_pred):
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff/13) - 1, np.exp(diff/10) - 1))

# =============================================================================
# Optional custom AttentionLayer
# =============================================================================
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# =============================================================================
# ExperimentPipeline Class
# =============================================================================
class ExperimentPipeline:
    def __init__(self,
                 df,
                 features,
                 target_col="RUL",
                 group_col="unit_nr",
                 time_col="time_cycles",
                 sequence_length=20,
                 epochs=100,
                 random_state=42):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.features = features
        self.target_col = target_col
        self.group_col = group_col
        self.time_col = time_col

        self.df = df.copy()
        self.df.sort_values(by=[self.group_col, self.time_col], inplace=True)
        self.df[self.target_col] = self.df[self.target_col].astype(float)

        self.results_metrics = {}
        self.loss_histories = {}
        self.results_df = None
        self.groups = None

        # For storing conformal info for inference:
        self.conformal_results_df = None  # global calibration results
        self.conformal_inference = None   # best method & margin from single unit

        # Prepare data for DL and ML
        self._prepare_data()
        self._prepare_ml_data()

    @staticmethod
    def compute_s_score(rul_true, rul_pred):
        return compute_s_score(rul_true, rul_pred)

    @staticmethod
    def create_sequences(data, target, sequence_length):
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i: i + sequence_length])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)

    def _prepare_data(self):
        print("Preparing data for DL sequence models...")
        scaler = MinMaxScaler()
        self.df[self.features] = scaler.fit_transform(self.df[self.features].values)
        X_list, y_list, group_list = [], [], []
        for unit, group_df in self.df.groupby(self.group_col):
            group_df = group_df.sort_values(by=self.time_col)
            unit_features = group_df[self.features].values
            unit_target = group_df[self.target_col].values
            if len(unit_features) > self.sequence_length:
                X_seq, y_seq = ExperimentPipeline.create_sequences(unit_features, unit_target, self.sequence_length)
                X_list.append(X_seq)
                y_list.append(y_seq)
                group_list.extend([unit] * len(y_seq))
        self.X_dl = np.concatenate(X_list, axis=0) if X_list else np.array([])
        self.y_dl = np.concatenate(y_list, axis=0) if y_list else np.array([])
        self.groups = np.array(group_list) if group_list else np.array([])
        print(f"DL data prepared: X_dl shape = {self.X_dl.shape}, y_dl shape = {self.y_dl.shape}\n")

    def _prepare_ml_data(self):
        print("Preparing lagged features for ML models...")
        X_list, y_list, group_list = [], [], []
        for unit, group_df in self.df.groupby(self.group_col):
            group_df = group_df.sort_values(by=self.time_col)
            unit_features = group_df[self.features].values
            unit_target = group_df[self.target_col].values
            if len(unit_features) >= self.sequence_length:
                for i in range(self.sequence_length, len(unit_features)):
                    x_lag = unit_features[i - self.sequence_length: i].flatten()
                    X_list.append(x_lag)
                    y_list.append(unit_target[i])
                    group_list.append(unit)
        self.X_ml = np.array(X_list) if X_list else np.array([])
        self.y_ml = np.array(y_list) if y_list else np.array([])
        self.ml_groups = np.array(group_list) if group_list else np.array([])
        print(f"ML data prepared: X_ml shape = {self.X_ml.shape}, y_ml shape = {self.y_ml.shape}\n")

    # -------------------------
    # DL Model Definitions
    # -------------------------
    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(300, activation='tanh', return_sequences=True, input_shape=input_shape),
            Dropout(0.5),
            BatchNormalization(),
            LSTM(250, activation='tanh', return_sequences=True),
            Dropout(0.5),
            BatchNormalization(),
            LSTM(200, activation='tanh'),
            Dropout(0.5),
            Dense(100, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
        return model

    def create_bilstm_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(300, activation='tanh', return_sequences=True), input_shape=input_shape),
            Dropout(0.5),
            BatchNormalization(),
            Bidirectional(LSTM(250, activation='tanh', return_sequences=True)),
            Dropout(0.5),
            BatchNormalization(),
            Bidirectional(LSTM(200, activation='tanh')),
            Dropout(0.5),
            Dense(100, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
        return model

    def create_nbeats_model(self, input_shape):
        seq_length, num_features = input_shape
        total_input = seq_length * num_features
        inp = Input(shape=(seq_length, num_features))
        x = Flatten()(inp)
        # Block 1
        b1 = Dense(256, activation='relu')(x)
        b1 = Dense(256, activation='relu')(b1)
        backcast_1 = Dense(total_input, activation='linear', name="backcast_1")(b1)
        forecast_1 = Dense(1, activation='linear', name="forecast_1")(b1)
        # Residual for next block
        x2 = x - backcast_1
        # Block 2
        b2 = Dense(256, activation='relu')(x2)
        b2 = Dense(256, activation='relu')(b2)
        backcast_2 = Dense(total_input, activation='linear', name="backcast_2")(b2)
        forecast_2 = Dense(1, activation='linear', name="forecast_2")(b2)
        final_forecast = forecast_1 + forecast_2
        model = Model(inputs=inp, outputs=final_forecast, name="NBEATS")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def create_cnn_gru_model(self, input_shape):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
            GRU(100, return_sequences=False),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    # -------------------------------------------------------------------------
    # XGBoost (classical ML approach)
    # -------------------------------------------------------------------------
    def run_xgboost_fixed(self):
        print(f"\nEvaluating XGBoost using fixed train–test split (by {self.group_col}) ...")
        unique_units = np.unique(self.ml_groups)
        if len(unique_units) == 0:
            print("No data available for XGBoost.")
            self.results_metrics["XGBoost"] = {"MSE": np.nan, "MAE": np.nan,
                                               "RMSE": np.nan, "R2": np.nan,
                                               "S-score": np.nan}
            self.loss_histories["XGBoost"] = None
            return self.results_metrics["XGBoost"]
        n_train_units = int(0.8 * len(unique_units))
        train_units = unique_units[:n_train_units]
        test_units = unique_units[n_train_units:]
        if len(train_units) == 0 or len(test_units) == 0:
            print("Not enough units for XGBoost train/test split.")
            self.results_metrics["XGBoost"] = {"MSE": np.nan, "MAE": np.nan,
                                               "RMSE": np.nan, "R2": np.nan,
                                               "S-score": np.nan}
            self.loss_histories["XGBoost"] = None
            return self.results_metrics["XGBoost"]
        train_idx = np.where(np.isin(self.ml_groups, train_units))[0]
        test_idx = np.where(np.isin(self.ml_groups, test_units))[0]
        X_train, X_test = self.X_ml[train_idx], self.X_ml[test_idx]
        y_train, y_test = self.y_ml[train_idx], self.y_ml[test_idx]
        print(f"Fixed split (XGBoost): {len(train_units)} units for training, {len(test_units)} units for testing.")
        xgb_model = XGBRegressor(
            max_depth=6,
            learning_rate=0.01,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = xgb_model.predict(X_test)
        mse_val = mean_squared_error(y_test, y_pred)
        mae_val = np.mean(np.abs(y_test - y_pred))
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_test, y_pred)
        s_val = self.compute_s_score(y_test, y_pred)
        print(f"XGBoost - MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, S-score: {s_val:.4f}\n")
        metrics = {"MSE": mse_val, "MAE": mae_val, "RMSE": rmse_val, "R2": r2_val, "S-score": s_val}
        self.loss_histories["XGBoost"] = None
        self.results_metrics["XGBoost"] = metrics
        folder = self.get_target_folder()
        final_path = os.path.join(folder, "XGBoost_final.json")
        xgb_model.save_model(final_path)
        print(f"XGBoost model saved to {final_path}")
        return metrics

    # -------------------------
    # Convenience methods for DL fixed-split experiments
    # -------------------------
    def run_lstm_fixed(self):
        return self.run_model_fixed(self.create_lstm_model, "LSTM")

    def run_bilstm_fixed(self):
        return self.run_model_fixed(self.create_bilstm_model, "BiLSTM")

    def run_nbeats_fixed(self):
        return self.run_model_fixed(self.create_nbeats_model, "NBEATS")

    def run_cnn_gru_fixed(self):
        return self.run_model_fixed(self.create_cnn_gru_model, "CNNGRU")

    def run_model_fixed(self, model_func, model_name):
        print(f"\nEvaluating {model_name} using fixed train–test split (by {self.group_col}) ...")
        if self.X_dl.size == 0 or self.y_dl.size == 0:
            print(f"No data available for {model_name}. Skipping.")
            self.results_metrics[model_name] = {"MSE": np.nan, "MAE": np.nan,
                                                 "RMSE": np.nan, "R2": np.nan,
                                                 "S-score": np.nan}
            self.loss_histories[model_name] = None
            return self.results_metrics[model_name], None, None
        unique_units = np.unique(self.groups)
        n_train_units = int(0.8 * len(unique_units))
        train_units = unique_units[:n_train_units]
        test_units = unique_units[n_train_units:]
        if len(train_units) == 0 or len(test_units) == 0:
            print(f"Not enough units for {model_name} train/test split. Skipping.")
            self.results_metrics[model_name] = {"MSE": np.nan, "MAE": np.nan,
                                                 "RMSE": np.nan, "R2": np.nan,
                                                 "S-score": np.nan}
            self.loss_histories[model_name] = None
            return self.results_metrics[model_name], None, None
        train_idx = np.where(np.isin(self.groups, train_units))[0]
        test_idx = np.where(np.isin(self.groups, test_units))[0]
        X_train, X_test = self.X_dl[train_idx], self.X_dl[test_idx]
        y_train, y_test = self.y_dl[train_idx], self.y_dl[test_idx]
        print(f"Fixed split: {len(train_units)} units for training, {len(test_units)} units for testing.")
        checkpoint = ModelCheckpoint(f"{model_name}_best.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=1)
        early_stop = EarlyStopping(monitor="val_loss", patience=8, verbose=1, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=4, verbose=1, min_lr=0.00005)
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = model_func(input_shape)
        history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        y_pred = model.predict(X_test).ravel()
        mse_val = mean_squared_error(y_test, y_pred)
        mae_val = np.mean(np.abs(y_test - y_pred))
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_test, y_pred)
        s_val = self.compute_s_score(y_test, y_pred)
        print(f"{model_name} - MSE: {mse_val:.4f}, MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, S-score: {s_val:.4f}\n")
        metrics = {"MSE": mse_val, "MAE": mae_val, "RMSE": rmse_val, "R2": r2_val, "S-score": s_val}
        folder = self.get_target_folder()
        final_path = os.path.join(folder, f"{model_name}_final.h5")
        model.save(final_path)
        print(f"{model_name} weights saved to {final_path}")
        self.results_metrics[model_name] = metrics
        self.loss_histories[model_name] = history.history["loss"]
        return metrics, history.history["loss"], model

    def run_experiment(self):
        """
        Runs a fixed-split experiment for multiple DL models + XGBoost.
        Populates self.results_df and self.loss_histories.
        """
        print("Running fixed split experiment...\n")
        self.run_lstm_fixed()
        self.run_bilstm_fixed()
        self.run_nbeats_fixed()
        self.run_cnn_gru_fixed()
        self.run_xgboost_fixed()
        models_list, mse_list, mae_list, rmse_list, r2_list, s_list = [], [], [], [], [], []
        for model_name, metrics in self.results_metrics.items():
            models_list.append(model_name)
            mse_list.append(metrics.get("MSE", np.nan))
            mae_list.append(metrics.get("MAE", np.nan))
            rmse_list.append(metrics.get("RMSE", np.nan))
            r2_list.append(metrics.get("R2", np.nan))
            s_list.append(metrics.get("S-score", np.nan))
        self.results_df = pd.DataFrame({
            "Model": models_list,
            "MSE": mse_list,
            "MAE": mae_list,
            "RMSE": rmse_list,
            "R2": r2_list,
            "S-score": s_list
        })
        print("Experiment completed.\n")
        return self.results_df

    def get_target_folder(self):
        folder = "adjusted_RUL" if self.target_col.lower() == "adjusted_rul" else "original_RUL"
        os.makedirs(folder, exist_ok=True)
        return folder

    def plot_metrics(self):
        """
        Plots S-score, MAE, RMSE, and R^2 in a 2×2 grid of bar charts,
        using a colorful palette and annotated bars.
        """
        if self.results_df is None:
            print("No results to plot. Run run_experiment() first.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Model Performance Metrics", fontsize=16, fontweight="bold")
        def fancy_barplot(ax, x, y, data, title):
            sns.barplot(
                x=x,
                y=y,
                data=data,
                ax=ax,
                palette="Spectral",
                edgecolor="black"
            )
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, linestyle='--', alpha=0.5, axis='y')
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3)
        fancy_barplot(axes[0, 0], x="Model", y="S-score", data=self.results_df, title="S-score")
        fancy_barplot(axes[0, 1], x="Model", y="MAE", data=self.results_df, title="MAE")
        fancy_barplot(axes[1, 0], x="Model", y="RMSE", data=self.results_df, title="RMSE")
        fancy_barplot(axes[1, 1], x="Model", y="R2", data=self.results_df, title="R²")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def plot_loss_histories(self):
        """
        Plots the training loss curves for each Keras model.
        (XGBoost does not produce a Keras loss history, so it's excluded.)
        """
        plt.figure(figsize=(10, 6))
        for model_name, loss_history in self.loss_histories.items():
            if loss_history is not None:
                plt.plot(loss_history, label=model_name)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("DL Models: Aggregated Training Loss History")
        plt.legend()
        plt.show()

    # =============================================================================
    # New: Conformal Prediction Interval for Entire Test Set (Clamped >= 0)
    # =============================================================================
    def plot_conformal_prediction_interval(self, alpha=0.05, show_plot=True):
        """
        1) Identify best model by lowest S-score.
        2) Reload that model.
        3) Compute multiple conformal intervals on test set:
           - Naive
           - Weighted
           - Bootstrap
           - Jackknife+ (placeholder)
           - CV+ (placeholder)
           - ComplexMethod (median)
        4) Enforce monotonic predictions, clamp them >= 0.
        5) Plot the predicted values and the lower boundary of each conformal interval on the same graph.
        6) Store the results in self.conformal_results_df for future inference.
        """
        from xgboost import XGBRegressor
        if not self.results_metrics:
            print("No results_metrics found. Please run run_experiment() first.")
            return
        def get_s_score(mdict):
            return mdict.get("S-score", np.inf)
        best_model_name = min(self.results_metrics, key=lambda m: get_s_score(self.results_metrics[m]))
        best_model_metrics = self.results_metrics[best_model_name]
        if np.isinf(best_model_metrics["S-score"]):
            print("Could not identify a valid 'best' model by S-score.")
            return
        print(f"[Conformal] Best model by S-score is: {best_model_name} (S-score={best_model_metrics['S-score']:.2f})")
        folder = self.get_target_folder()
        if best_model_name == "XGBoost":
            model_path = os.path.join(folder, "XGBoost_final.json")
            best_model = XGBRegressor()
            best_model.load_model(model_path)
            is_keras_model = False
        else:
            model_path = os.path.join(folder, f"{best_model_name}_final.h5")
            if best_model_name == "LSTM":
                best_model = self.create_lstm_model((self.sequence_length, len(self.features)))
            elif best_model_name == "BiLSTM":
                best_model = self.create_bilstm_model((self.sequence_length, len(self.features)))
            elif best_model_name == "NBEATS":
                best_model = self.create_nbeats_model((self.sequence_length, len(self.features)))
            elif best_model_name == "CNNGRU":
                best_model = self.create_cnn_gru_model((self.sequence_length, len(self.features)))
            best_model.load_weights(model_path)
            is_keras_model = True
        unique_units = np.unique(self.groups)
        n_train_units = int(0.8 * len(unique_units))
        train_units = unique_units[:n_train_units]
        test_units = unique_units[n_train_units:]
        if len(train_units) == 0 or len(test_units) == 0:
            print("Not enough units for Conformal. Exiting.")
            return
        train_idx = np.where(np.isin(self.groups, train_units))[0]
        test_idx = np.where(np.isin(self.groups, test_units))[0]
        if is_keras_model:
            X_train, y_train = self.X_dl[train_idx], self.y_dl[train_idx]
            X_test, y_test = self.X_dl[test_idx], self.y_dl[test_idx]
            def model_predict_fn(X):
                return best_model.predict(X).ravel()
        else:
            X_train, y_train = self.X_ml[np.isin(self.ml_groups, train_units)], self.y_ml[np.isin(self.ml_groups, train_units)]
            X_test, y_test = self.X_ml[np.isin(self.ml_groups, test_units)], self.y_ml[np.isin(self.ml_groups, test_units)]
            def model_predict_fn(X):
                return best_model.predict(X)
        if X_train.size == 0 or X_test.size == 0:
            print("No data in train/test. Conformal cannot proceed.")
            return
        preds_cal = np.clip(model_predict_fn(X_train), 0, None)
        train_residuals = y_train - preds_cal
        absolute_residuals = np.abs(train_residuals)
        def naive_quantile_margin(alpha_val=alpha):
            return np.quantile(absolute_residuals, 1.0 - alpha_val)
        def weighted_exponential_margin(alpha_val=alpha):
            n = len(absolute_residuals)
            sorted_res = np.sort(absolute_residuals)
            weights = np.exp(np.arange(n) / (n/5.0))
            w_sorted_idx = np.argsort(absolute_residuals)
            w_sorted = weights[w_sorted_idx]
            cdf = np.cumsum(w_sorted)
            total_w = cdf[-1]
            cutoff = (1 - alpha_val) * total_w
            idx = np.searchsorted(cdf, cutoff)
            idx = min(idx, n-1)
            return sorted_res[idx]
        def bootstrap_margin(alpha_val=alpha, B=200):
            rng = np.random.default_rng(self.random_state)
            n = len(absolute_residuals)
            q_level = 1.0 - alpha_val
            samples = []
            for _ in range(B):
                sample = rng.choice(absolute_residuals, size=n, replace=True)
                samples.append(np.quantile(sample, q_level))
            return float(np.mean(samples))
        def jackknife_plus_margin(alpha_val=alpha):
            return naive_quantile_margin(alpha_val)
        def cv_plus_margin(alpha_val=alpha, K=5):
            return naive_quantile_margin(alpha_val)
        m_naive = naive_quantile_margin(alpha)
        m_weighted = weighted_exponential_margin(alpha)
        m_bootstrap = bootstrap_margin(alpha, 200)
        m_jackknife = jackknife_plus_margin(alpha)
        m_cv = cv_plus_margin(alpha, 5)
        m_complex = np.median([m_naive, m_weighted, m_bootstrap])
        margin_dict = {
            "Naive": m_naive,
            "Weighted": m_weighted,
            "Bootstrap": m_bootstrap,
            "Jackknife+": m_jackknife,
            "CV+": m_cv,
            "ComplexMethod": m_complex
        }
        tpreds = np.clip(model_predict_fn(X_test), 0, None)
        for i in range(1, len(tpreds)):
            if tpreds[i] > tpreds[i-1]:
                tpreds[i] = tpreds[i-1]
        def compute_index_difference(lower_bounds, actual):
            idx_conf = np.where(lower_bounds <= 0)[0]
            idx_conf = idx_conf[0] if len(idx_conf) > 0 else len(lower_bounds)-1
            idx_actual = np.where(actual <= 0)[0]
            idx_actual = idx_actual[0] if len(idx_actual) > 0 else len(actual)-1
            return abs(idx_conf - idx_actual)
        results = {}
        for name, margin in margin_dict.items():
            lower = np.clip(tpreds - margin, 0, None)
            upper = tpreds
            coverage = np.mean((y_test >= lower) & (y_test <= upper))
            avg_width = np.mean(upper - lower)
            idx_diff = compute_index_difference(lower, y_test)
            results[name] = {
                "margin": margin,
                "coverage": coverage,
                "avg_width": avg_width,
                "index_diff": idx_diff
            }
        conf_df = pd.DataFrame.from_dict(results, orient='index')
        conf_df["1-alpha"] = 1 - alpha
        self.conformal_results_df = conf_df
        print("\n[Conformal] Coverage results (test set):")
        print(conf_df)
        if show_plot:
            plt.figure(figsize=(12, 7))
            max_pts = min(len(tpreds), 400)
            xs = np.arange(max_pts)
            plt.plot(xs, tpreds[:max_pts], label="Pred (monotonic, clamped)", linewidth=2, color='black')
            for name, info in results.items():
                margin = info["margin"]
                lower_vals = np.clip(tpreds[:max_pts] - margin, 0, None)
                plt.plot(xs, lower_vals, label=f"{name} Lower")
            plt.title(f"Conformal Predictions (No Actual) – Best Model: {best_model_name}\n(alpha={alpha})")
            plt.xlabel("Test Sample (Truncated)")
            plt.ylabel(self.target_col)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.show()
        return conf_df

    # =============================================================================
    # New: Plot Conformal Regions and Predictions for Multiple Units
    # =============================================================================

    def plot_conformal_single_unit(self, unit_id, alpha=0.05, fold_splits=5, show_plot=True):
        """
        1) Identify best model by lowest S-score.
        2) Reload that model (DL or XGBoost).
        3) Build calibration set -> compute margins (parametric, weighted, etc.).
        4) Select best method by coverage≥(1-alpha) & minimal width (or fallback).
        5) Predict on single unit, enforce monotonic + clamp >=0.
        6) Plot only the chosen method's final intervals (clamped ≥ 0).
        7) Save the chosen method & margin to self.conformal_inference for future inference.
        """
        from xgboost import XGBRegressor
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        if not self.results_metrics:
            print("No results_metrics found. Please run run_experiment() first.")
            return

        def get_s_score(mdict):
            return mdict.get("S-score", np.inf)

        best_model_name = min(self.results_metrics, key=lambda m: get_s_score(self.results_metrics[m]))
        best_model_metrics = self.results_metrics[best_model_name]
        if np.isinf(best_model_metrics["S-score"]):
            print("Could not identify a valid 'best' model by S-score.")
            return

        print(f"[SingleUnitConformal] Best model by S-score is: {best_model_name} (S-score={best_model_metrics['S-score']:.2f})")
        folder = self.get_target_folder()

        # Reload best model
        if best_model_name == "XGBoost":
            model_path = os.path.join(folder, "XGBoost_final.json")
            best_model = XGBRegressor()
            best_model.load_model(model_path)
            is_keras_model = False
        else:
            model_path = os.path.join(folder, f"{best_model_name}_final.h5")
            if best_model_name == "LSTM":
                best_model = self.create_lstm_model((self.sequence_length, len(self.features)))
            elif best_model_name == "BiLSTM":
                best_model = self.create_bilstm_model((self.sequence_length, len(self.features)))
            elif best_model_name == "NBEATS":
                best_model = self.create_nbeats_model((self.sequence_length, len(self.features)))
            elif best_model_name == "CNNGRU":
                best_model = self.create_cnn_gru_model((self.sequence_length, len(self.features)))
            best_model.load_weights(model_path)
            is_keras_model = True

        # Build calibration set
        unique_units = np.unique(self.groups)
        n_train_units = int(0.8 * len(unique_units))
        train_units = unique_units[:n_train_units]
        if len(train_units) == 0:
            print("No train units. Exiting.")
            return

        train_idx = np.where(np.isin(self.groups, train_units))[0]

        if is_keras_model:
            X_cal, y_cal = self.X_dl[train_idx], self.y_dl[train_idx]

            def model_predict_fn(X):
                return best_model.predict(X).ravel()
        else:
            X_cal, y_cal = self.X_ml[np.isin(self.ml_groups, train_units)], self.y_ml[np.isin(self.ml_groups, train_units)]

            def model_predict_fn(X):
                return best_model.predict(X)

        if len(X_cal) == 0:
            print("Calibration set is empty. Cannot do conformal.")
            return

        # Conformal margin definitions
        def parametric_margin(alpha_val):
            preds_cal = np.clip(model_predict_fn(X_cal), 0, None)
            res = y_cal - preds_cal
            sigma = np.std(res, ddof=1)
            from scipy.stats import norm
            z_val = norm.ppf(1.0 - alpha_val)
            return z_val * sigma

        def weighted_margin(alpha_val):
            preds_cal = np.clip(model_predict_fn(X_cal), 0, None)
            abs_res = np.abs(y_cal - preds_cal)
            sorted_res = np.sort(abs_res)
            n = len(abs_res)
            weights = np.exp(np.arange(n) / (n/5.0))
            # re-order to match sorted residuals
            w_sorted_idx = np.argsort(abs_res)
            w_sorted = weights[w_sorted_idx]
            cdf = np.cumsum(w_sorted)
            total_w = cdf[-1]
            cutoff = (1 - alpha_val) * total_w
            idx = np.searchsorted(cdf, cutoff)
            idx = min(idx, n - 1)
            margin = sorted_res[idx]
            return margin

        def bootstrap_margin(alpha_val, B=200):
            rng = np.random.default_rng(self.random_state)
            preds_cal = np.clip(model_predict_fn(X_cal), 0, None)
            abs_res = np.abs(y_cal - preds_cal)
            n = len(abs_res)
            q_level = 1.0 - alpha_val
            samples = []
            for _ in range(B):
                sample = rng.choice(abs_res, size=n, replace=True)
                q = np.quantile(sample, q_level)
                samples.append(q)
            return float(np.median(samples))

        def jackknife_plus_margin(alpha_val):
            # placeholder
            return parametric_margin(alpha_val)

        def cv_plus_margin(alpha_val, K=fold_splits):
            # placeholder
            return parametric_margin(alpha_val)

        # Compute all margins
        m_param = parametric_margin(alpha)
        m_weighted = weighted_margin(alpha)
        m_bootstrap = bootstrap_margin(alpha, B=200)
        m_jackknife = jackknife_plus_margin(alpha)
        m_cv = cv_plus_margin(alpha, fold_splits)
        m_complex = np.median([m_param, m_weighted, m_bootstrap])

        margin_dict = {
            "ParamNormal": m_param,
            "Weighted": m_weighted,
            "Bootstrap": m_bootstrap,
            "Jackknife+": m_jackknife,
            "CV+": m_cv,
            "ComplexMethod": m_complex
        }

        # Evaluate coverage on the calibration set
        def compute_index_difference(lower_bounds, actual):
            idx_conf = np.where(lower_bounds <= 0)[0]
            idx_conf = idx_conf[0] if len(idx_conf) > 0 else len(lower_bounds) - 1
            idx_actual = np.where(actual <= 0)[0]
            idx_actual = idx_actual[0] if len(idx_actual) > 0 else len(actual) - 1
            return abs(idx_conf - idx_actual)

        results = []
        preds_cal = np.clip(model_predict_fn(X_cal), 0, None)
        for name, margin in margin_dict.items():
            lower_cal = preds_cal - margin
            lower_cal = np.clip(lower_cal, 0, None)
            upper_cal = preds_cal  # already clipped
            coverage_cal = np.mean((y_cal >= lower_cal) & (y_cal <= upper_cal))
            avg_width_cal = np.mean(upper_cal - lower_cal)
            idx_diff_cal = compute_index_difference(lower_cal, y_cal)
            results.append({
                "Method": name,
                "Margin": margin,
                "Coverage": coverage_cal,
                "AvgWidth": avg_width_cal,
                "IndexDiff": idx_diff_cal
            })

        conf_df = pd.DataFrame(results)
        conf_df["TargetCoverage"] = 1 - alpha

    def evaluate_conformal_metrics_per_unit(self, alpha=0.05, fold_splits=5, margin_method="best", save_path=None):
        """
        Evaluates conformal metrics for each unit in the validation (test) set.
        For each test unit, the function computes:
          - A conformal margin, chosen based on the margin_method:
              * "complex": the median of candidate margins (parametric, weighted, bootstrap)
              * "best": the candidate margin that yields the highest coverage on the calibration set.
          - Coverage: the fraction of true values falling between [prediction - margin, prediction]
          - Average interval width
          - An index difference metric (difference between the first index where the lower bound is 0 and the first index where the true value is 0)
        The results are compiled into a DataFrame and, if a save_path is provided, saved to disk.
        """
        # Identify the best model by S-score and load it
        def get_s_score(mdict):
            return mdict.get("S-score", np.inf)
        best_model_name = min(self.results_metrics, key=lambda m: get_s_score(self.results_metrics[m]))
        best_model_metrics = self.results_metrics[best_model_name]
        if np.isinf(best_model_metrics["S-score"]):
            print("Could not identify a valid 'best' model by S-score.")
            return

        print(f"[Conformal Metrics] Best model by S-score is: {best_model_name} (S-score={best_model_metrics['S-score']:.2f})")
        folder = self.get_target_folder()
        # Reload best model
        if best_model_name == "XGBoost":
            from xgboost import XGBRegressor
            model_path = os.path.join(folder, "XGBoost_final.json")
            best_model = XGBRegressor()
            best_model.load_model(model_path)
            is_keras_model = False
            groups_array = self.ml_groups
        else:
            model_path = os.path.join(folder, f"{best_model_name}_final.h5")
            if best_model_name == "LSTM":
                best_model = self.create_lstm_model((self.sequence_length, len(self.features)))
            elif best_model_name == "BiLSTM":
                best_model = self.create_bilstm_model((self.sequence_length, len(self.features)))
            elif best_model_name == "NBEATS":
                best_model = self.create_nbeats_model((self.sequence_length, len(self.features)))
            elif best_model_name == "CNNGRU":
                best_model = self.create_cnn_gru_model((self.sequence_length, len(self.features)))
            best_model.load_weights(model_path)
            is_keras_model = True
            groups_array = self.groups

        # Split units into training (calibration) and validation (test)
        unique_units = np.unique(groups_array)
        n_train_units = int(0.8 * len(unique_units))
        train_units = unique_units[:n_train_units]
        test_units = unique_units[n_train_units:]

        if len(train_units) == 0 or len(test_units) == 0:
            print("Not enough units for evaluation.")
            return

        # Build calibration set (from training units)
        train_idx = np.where(np.isin(groups_array, train_units))[0]
        if is_keras_model:
            X_cal, y_cal = self.X_dl[train_idx], self.y_dl[train_idx]
            def model_predict_fn(X):
                return best_model.predict(X).ravel()
        else:
            X_cal, y_cal = self.X_ml[np.isin(self.ml_groups, train_units)], self.y_ml[np.isin(self.ml_groups, train_units)]
            def model_predict_fn(X):
                return best_model.predict(X)

        # --- Define candidate margin functions ---
        from scipy.stats import norm
        def parametric_margin(alpha_val):
            preds_cal = np.clip(model_predict_fn(X_cal), 0, None)
            res = y_cal - preds_cal
            sigma = np.std(res, ddof=1)
            z_val = norm.ppf(1.0 - alpha_val)
            return z_val * sigma

        def weighted_margin(alpha_val):
            preds_cal = np.clip(model_predict_fn(X_cal), 0, None)
            abs_res = np.abs(y_cal - preds_cal)
            sorted_res = np.sort(abs_res)
            n = len(abs_res)
            weights = np.exp(np.arange(n) / (n/5.0))
            w_sorted_idx = np.argsort(abs_res)
            w_sorted = weights[w_sorted_idx]
            cdf = np.cumsum(w_sorted)
            total_w = cdf[-1]
            cutoff = (1 - alpha_val) * total_w
            idx = np.searchsorted(cdf, cutoff)
            idx = min(idx, n - 1)
            return sorted_res[idx]

        def bootstrap_margin(alpha_val, B=200):
            rng = np.random.default_rng(self.random_state)
            preds_cal = np.clip(model_predict_fn(X_cal), 0, None)
            abs_res = np.abs(y_cal - preds_cal)
            n = len(abs_res)
            q_level = 1.0 - alpha_val
            samples = []
            for _ in range(B):
                sample = rng.choice(abs_res, size=n, replace=True)
                samples.append(np.quantile(sample, q_level))
            return float(np.median(samples))

        # Compute candidate margins
        m_param = parametric_margin(alpha)
        m_weighted = weighted_margin(alpha)
        m_bootstrap = bootstrap_margin(alpha, B=200)
        margin_candidates = {"ParamNormal": m_param, "Weighted": m_weighted, "Bootstrap": m_bootstrap}

        # --- Select margin based on user option ---
        if margin_method.lower() == "complex":
            best_margin = np.median(list(margin_candidates.values()))
            print(f"[Conformal Metrics] Using 'complex' method. Global best margin (median) = {best_margin:.3f}")
        elif margin_method.lower() == "best":
            # Evaluate each candidate on calibration set to compute coverage
            preds_cal = np.clip(model_predict_fn(X_cal), 0, None)
            candidate_coverages = {}
            for key, margin_val in margin_candidates.items():
                lower_cal = np.clip(preds_cal - margin_val, 0, None)
                coverage = np.mean((y_cal >= lower_cal) & (y_cal <= preds_cal))
                candidate_coverages[key] = coverage
                print(f"[Conformal Metrics] Candidate {key}: margin = {margin_val:.3f}, coverage = {coverage:.3f}")
            best_candidate = max(candidate_coverages, key=candidate_coverages.get)
            best_margin = margin_candidates[best_candidate]
            print(f"[Conformal Metrics] Using 'best' method. Selected candidate: {best_candidate} with margin = {best_margin:.3f} (coverage = {candidate_coverages[best_candidate]:.3f})")
        else:
            print("Invalid margin_method specified. Choose either 'complex' or 'best'.")
            return

        # Helper function to compute index difference metric
        def compute_index_difference(lower_bounds, actual):
            idx_conf = np.where(lower_bounds <= 0)[0]
            idx_conf = idx_conf[0] if len(idx_conf) > 0 else len(lower_bounds) - 1
            idx_actual = np.where(actual <= 0)[0]
            idx_actual = idx_actual[0] if len(idx_actual) > 0 else len(actual) - 1
            return abs(idx_conf - idx_actual)

        # Iterate over each unit in the validation set and compute metrics
        results = []
        for unit in test_units:
            # Extract unit data
            unit_df = self.df[self.df[self.group_col] == unit].copy()
            unit_df.sort_values(by=self.time_col, inplace=True)
            if len(unit_df) < self.sequence_length:
                print(f"Unit {unit} does not have enough data for sequence_length={self.sequence_length}. Skipping.")
                continue
            unit_features = unit_df[self.features].values
            unit_target = unit_df[self.target_col].values

            # Build sliding window for predictions
            if is_keras_model:
                X_list = [unit_features[i: i + self.sequence_length] for i in range(len(unit_features) - self.sequence_length)]
                y_true = unit_target[self.sequence_length:]
            else:
                X_list = [unit_features[i - self.sequence_length: i].flatten() for i in range(self.sequence_length, len(unit_features))]
                y_true = unit_target[self.sequence_length:]
            if len(X_list) == 0:
                print(f"Unit {unit} has insufficient time steps for prediction. Skipping.")
                continue
            X_unit = np.array(X_list)

            # Get predictions and enforce monotonicity + clamp
            preds = model_predict_fn(X_unit)
            for i in range(1, len(preds)):
                if preds[i] > preds[i-1]:
                    preds[i] = preds[i-1]
            preds = np.clip(preds, 0, None)

            # Compute conformal prediction intervals
            lower_bound = np.clip(preds - best_margin, 0, None)
            coverage = np.mean((y_true >= lower_bound) & (y_true <= preds))
            avg_width = np.mean(preds - lower_bound)
            idx_diff = compute_index_difference(lower_bound, y_true)

            results.append({
                "unit_id": unit,
                "num_predictions": len(preds),
                "margin": best_margin,
                "coverage": coverage,
                "avg_width": avg_width,
                "index_diff": idx_diff
            })
            print(f"[Unit {unit}] Coverage: {coverage:.3f}, Avg Width: {avg_width:.3f}, Index Diff: {idx_diff}")

        results_df = pd.DataFrame(results)
        if save_path is not None:
            results_df.to_csv(save_path, index=False)
            print(f"Conformal metrics saved to {save_path}")
        return results_df

        # pick best method by coverage≥(1-alpha) & minimal width
        valid_methods = conf_df[conf_df["Coverage"] >= (1.0 - alpha)]
        if len(valid_methods) == 0:
            chosen = conf_df.iloc[(conf_df["Coverage"] - (1 - alpha)).abs().argsort()[:1]]
            print("[SingleUnitConformal] No method meets coverage >= target. Using closest coverage.")
        else:
            chosen = valid_methods.sort_values("AvgWidth").head(1)

        best_method_choice = chosen["Method"].values[0]
        best_margin = chosen["Margin"].values[0]
        print("\n[SingleUnitConformal] Candidate methods:\n", conf_df)
        print(f"Choosing method '{best_method_choice}' with margin={best_margin:.3f}\n")

        # Save for inference
        self.conformal_inference = {
            "best_method": best_method_choice,
            "best_margin": best_margin
        }

        # Predict on single unit
        unit_df = self.df[self.df[self.group_col] == unit_id].copy()
        unit_df.sort_values(by=self.time_col, inplace=True)
        if len(unit_df) < self.sequence_length:
            print(f"Unit {unit_id} does not have enough data for sequence_length={self.sequence_length}")
            return

        unit_features = unit_df[self.features].values
        unit_target = unit_df[self.target_col].values

        if is_keras_model:
            X_list = []
            for i in range(len(unit_features) - self.sequence_length):
                X_list.append(unit_features[i: i + self.sequence_length])
            X_unit = np.array(X_list)
            y_true = unit_target[self.sequence_length:]
        else:
            X_list = []
            for i in range(self.sequence_length, len(unit_features)):
                window = unit_features[i - self.sequence_length: i].flatten()
                X_list.append(window)
            X_unit = np.array(X_list)
            y_true = unit_target[self.sequence_length:]

        if len(X_unit) == 0:
            print(f"Unit {unit_id} has insufficient time steps for prediction.")
            return

        pred_unit = model_predict_fn(X_unit)
        # Enforce monotonic + clamp to [0,∞)
        for i in range(1, len(pred_unit)):
            if pred_unit[i] > pred_unit[i - 1]:
                pred_unit[i] = pred_unit[i - 1]
        pred_unit = np.clip(pred_unit, 0, None)

        final_lower = np.clip(pred_unit - best_margin, 0, None)
        final_upper = pred_unit

        coverage_single = np.mean((y_true >= final_lower) & (y_true <= final_upper))
        avg_width_single = np.mean(final_upper - final_lower)
        index_diff_single = compute_index_difference(final_lower, y_true)
        print(f"Single-Unit Coverage = {coverage_single:.3f}, Avg Width = {avg_width_single:.3f}, Index Difference = {index_diff_single}")

        if show_plot:
            xvals = np.arange(len(pred_unit))
            plt.figure(figsize=(10, 6))
            plt.plot(xvals, y_true, label="Actual (y_true)", marker='o', linestyle='--')
            plt.plot(xvals, pred_unit, label="Monotonic + Clamped Pred", linewidth=2, color='orange')
            plt.fill_between(xvals, final_lower, final_upper, alpha=0.2, color='green',
                            label=f"{best_method_choice} CI")
            plt.title(f"Single-Unit Conformal Intervals\n(unit={unit_id}, alpha={alpha}, method={best_method_choice})")
            plt.xlabel("Window Index (time order)")
            plt.ylabel(self.target_col)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()

        return conf_df

    def plot_conformal_multiple_units(self, unit_ids, alpha=0.05, fold_splits=5, show_plot=True, gap=10):
        """
        For a list (or a single value) of unit IDs, compute the predicted values and conformal lower bounds
        using the best model and the conformal margin saved in self.conformal_inference.
        Each unit's predictions are plotted sequentially on the x-axis (i.e. the next unit starts after a gap
        following the previous unit's predictions until RUL=0).
        All units share the same RUL y-axis.
        """
        # Ensure unit_ids is a list
        if not isinstance(unit_ids, (list, tuple, np.ndarray)):
            unit_ids = [unit_ids]
        from xgboost import XGBRegressor
        if not self.results_metrics:
            print("No results_metrics found. Run run_experiment() first.")
            return

        def get_s_score(mdict):
            return mdict.get("S-score", np.inf)
        best_model_name = min(self.results_metrics, key=lambda m: get_s_score(self.results_metrics[m]))
        best_model_metrics = self.results_metrics[best_model_name]
        if np.isinf(best_model_metrics["S-score"]):
            print("Could not identify a valid 'best' model by S-score.")
            return
        folder = self.get_target_folder()
        if best_model_name == "XGBoost":
            model_path = os.path.join(folder, "XGBoost_final.json")
            best_model = XGBRegressor()
            best_model.load_model(model_path)
            is_keras_model = False
        else:
            model_path = os.path.join(folder, f"{best_model_name}_final.h5")
            if best_model_name == "LSTM":
                best_model = self.create_lstm_model((self.sequence_length, len(self.features)))
            elif best_model_name == "BiLSTM":
                best_model = self.create_bilstm_model((self.sequence_length, len(self.features)))
            elif best_model_name == "NBEATS":
                best_model = self.create_nbeats_model((self.sequence_length, len(self.features)))
            elif best_model_name == "TCN":
                best_model = self.create_cnn_gru_model((self.sequence_length, len(self.features)))
            best_model.load_weights(model_path)
            is_keras_model = True

        # Use stored conformal margin if available; otherwise use global ComplexMethod margin.
        if self.conformal_inference is None:
            conf_df = self.plot_conformal_prediction_interval(alpha=alpha, show_plot=False)
            best_margin = conf_df.loc["ComplexMethod", "margin"]
            self.conformal_inference = {"best_method": "ComplexMethod", "best_margin": best_margin}
        else:
            best_margin = self.conformal_inference["best_margin"]

        # Prepare the cumulative x-axis offset
        current_offset = 0
        gap_offset = gap  # gap between units
        colors = sns.color_palette("husl", len(unit_ids))
        plt.figure(figsize=(12, 7))

        # For each unit, compute predictions and determine x-axis range separately.
        for idx, unit in enumerate(unit_ids):
            unit_df = self.df[self.df[self.group_col] == unit].copy()
            unit_df.sort_values(by=self.time_col, inplace=True)
            if len(unit_df) < self.sequence_length:
                print(f"Unit {unit} does not have enough data for sequence_length={self.sequence_length}. Skipping.")
                continue
            unit_features = unit_df[self.features].values

            # Build sliding window for predictions.
            if is_keras_model:
                X_list = [unit_features[i: i + self.sequence_length] for i in range(len(unit_features) - self.sequence_length)]
            else:
                X_list = [unit_features[i - self.sequence_length: i].flatten() for i in range(self.sequence_length, len(unit_features))]
            X_unit = np.array(X_list)
            if len(X_unit) == 0:
                print(f"Unit {unit} has insufficient time steps for prediction. Skipping.")
                continue

            # Get predictions and enforce monotonicity.
            if is_keras_model:
                preds = best_model.predict(X_unit).ravel()
            else:
                preds = best_model.predict(X_unit)
            for i in range(1, len(preds)):
                if preds[i] > preds[i-1]:
                    preds[i] = preds[i-1]
            preds = np.clip(preds, 0, None)

            # Determine the segment to plot: from index 0 until the first index where prediction==0 (if any).
            zero_indices = np.where(preds <= 0)[0]
            if len(zero_indices) > 0:
                seg_length = zero_indices[0] + 1  # include that index
            else:
                seg_length = len(preds)

            # Create x-axis for this unit: offset to offset+seg_length
            x_axis = np.arange(current_offset, current_offset + seg_length)
            # Compute the lower bound
            lower_bound = np.clip(preds[:seg_length] - best_margin, 0, None)
            # Plot predictions and fill conformal region
            plt.plot(x_axis, preds[:seg_length], label=f"Unit {unit} Pred", color=colors[idx], linewidth=2)
            plt.fill_between(x_axis, lower_bound, preds[:seg_length], color=colors[idx], alpha=0.3, label=f"Unit {unit} CI")
            # Update offset (add gap)
            current_offset += seg_length + gap_offset

        plt.title(f"Conformal Regions and Predictions for Units (Sequential X-axis): {unit_ids}\nUsing Margin = {best_margin:.3f}")
        plt.xlabel("Cumulative Sliding Window Index")
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        if show_plot:
            plt.show()
        return
