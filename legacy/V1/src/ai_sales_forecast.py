"""
================================================================================
Author: AI Data Analysis Team
================================================================================
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)

import warnings
warnings.filterwarnings('ignore')


# ================================================================================
#                           LOGGING CONFIGURATION
# ================================================================================

def setup_logging():
    log_dir = Path(__file__).parent.parent / "outputs" / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "sales_prediction.log", encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()


# ================================================================================
#                        SALES PREDICTION PIPELINE CLASS
# ================================================================================

class SalesPredictionPipeline:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'AVG_PRICE_JOD': 2.5,
            'MA_WINDOW': 3,
            'TEST_SIZE': 0.2,
            'VAL_SIZE': 0.15,
            'RANDOM_STATE': 42,
            'RF_N_ESTIMATORS': 400,
            'RF_MAX_DEPTH': None,
            'MIN_SAMPLES_SPLIT': 2
        }
        
        if config:
            self.config.update(config)
        
        self._setup_directories()
        self.raw_data = None
        self.processed_data = None
        self.models = {}
        self.results = {}
        
        logger.info("=" * 70)
        logger.info("SALES PREDICTION PIPELINE INITIALIZED")
        logger.info("=" * 70)
    
    def _setup_directories(self) -> None:
        self.base_dir = Path(__file__).parent.parent
        self.raw_dir = self.base_dir / "data" / "raw"
        self.prep_dir = self.base_dir / "data" / "prepared"
        self.out_charts = self.base_dir / "outputs" / "charts"
        self.out_results = self.base_dir / "outputs" / "results"
        
        for directory in [self.prep_dir, self.out_charts, self.out_results]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Directory structure created at: {self.base_dir}")
    
    def load_data(self) -> pd.DataFrame:
        csv_path = self.raw_dir / "Sales_Transactions_Dataset_Weekly - V2.csv"
        xlsx_path = self.raw_dir / "Sales_Transactions_Dataset_Weekly - V2.xlsx"
        
        if csv_path.exists():
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    self.raw_data = pd.read_csv(csv_path, encoding=encoding)
                    logger.info(f"Data loaded from: {csv_path.name} (encoding: {encoding})")
                    break
                except (UnicodeDecodeError, Exception) as e:
                    logger.debug(f"Failed with encoding {encoding}: {e}")
                    continue
            else:
                self.raw_data = pd.read_csv(csv_path, encoding='utf-8', errors='replace')
                logger.info("Data loaded with character replacement")
        
        elif xlsx_path.exists():
            self.raw_data = pd.read_excel(xlsx_path)
            logger.info(f"Data loaded from: {xlsx_path.name}")
        
        else:
            error_msg = f"Data file not found! Please place data in: {self.raw_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Data shape: {self.raw_data.shape[0]} rows, {self.raw_data.shape[1]} columns")
        
        return self.raw_data
    
    def preprocess_data(self) -> pd.DataFrame:
        if self.raw_data is None:
            self.load_data()
        
        week_cols = self._extract_week_columns()
        
        if len(week_cols) < 10:
            error_msg = f"Only found {len(week_cols)} week columns. Expected W0-W51 format."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Found {len(week_cols)} week columns")
        
        weeks_numeric = self.raw_data[week_cols].apply(
            pd.to_numeric, errors='coerce'
        ).fillna(0)
        
        total_sales = weeks_numeric.sum(axis=0)
        
        df = pd.DataFrame({
            "Week": range(len(total_sales)),
            "Total_Sales": total_sales.values.astype(float)
        })
        
        df = self._create_features(df)
        
        self.processed_data = df.dropna().reset_index(drop=True)
        
        logger.info(f"Processed data: {len(self.processed_data)} rows with {len(self.processed_data.columns)} columns")
        
        return self.processed_data
    
    def _extract_week_columns(self) -> list:
        week_cols = [
            c for c in self.raw_data.columns 
            if str(c).strip().upper().startswith("W")
        ]
        
        def week_index(col):
            s = str(col).strip().upper().replace("W", "")
            return int(s) if s.isdigit() else 9999
        
        return sorted(week_cols, key=week_index)
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        ma_window = self.config['MA_WINDOW']
        avg_price = self.config['AVG_PRICE_JOD']
        
        df["Moving_Average"] = df["Total_Sales"].rolling(ma_window).mean()
        df["Previous_Week_Sales"] = df["Total_Sales"].shift(1)
        df["Change_Rate"] = df["Total_Sales"].pct_change().replace(
            [np.inf, -np.inf], 0
        ).fillna(0)
        
        min_sales = df["Total_Sales"].min()
        max_sales = df["Total_Sales"].max()
        denom = max_sales - min_sales if max_sales != min_sales else 1.0
        df["Normalised_Sales"] = (df["Total_Sales"] - min_sales) / denom
        
        df["Revenue_Estimated"] = df["Total_Sales"] * avg_price
        df["Rolling_Std"] = df["Total_Sales"].rolling(ma_window).std().fillna(0)
        df["Lag_2"] = df["Total_Sales"].shift(2).fillna(df["Total_Sales"].mean())
        df["Lag_3"] = df["Total_Sales"].shift(3).fillna(df["Total_Sales"].mean())
        
        logger.info(f"Created {len(df.columns) - 2} new features")
        
        return df
    
    def split_data(
        self, 
        test_size: Optional[float] = None,
        val_size: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.processed_data is None:
            self.preprocess_data()
        
        test_size = test_size or self.config['TEST_SIZE']
        val_size = val_size or self.config['VAL_SIZE']
        
        n = len(self.processed_data)
        test_cutoff = int(n * (1 - test_size))
        val_cutoff = int(test_cutoff * (1 - val_size))
        
        train = self.processed_data[self.processed_data["Week"] < val_cutoff]
        val = self.processed_data[
            (self.processed_data["Week"] >= val_cutoff) & 
            (self.processed_data["Week"] < test_cutoff)
        ]
        test = self.processed_data[self.processed_data["Week"] >= test_cutoff]
        
        logger.info(f"Data split: Train={len(train)}, Validation={len(val)}, Test={len(test)}")
        
        return train, val, test
    
    def prepare_features(
        self, 
        data: pd.DataFrame,
        feature_cols: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Define default features here
        default_feature_cols = [
            "Week", 
            "Previous_Week_Sales", 
            "Moving_Average", 
            "Change_Rate", 
            "Rolling_Std", 
            "Lag_2", 
            "Lag_3"
        ]
        
        # Use provided feature_cols or default
        cols_to_use = feature_cols if feature_cols is not None else default_feature_cols
        
        X = data[cols_to_use].values
        y = data["Total_Sales"].values
        
        logger.debug(f"Prepared features: X shape={X.shape}, y shape={y.shape}")
        
        return X, y
    
    def train_models(
        self, 
        train: pd.DataFrame, 
        val: pd.DataFrame
    ) -> Dict[str, Any]:
        X_train, y_train = self.prepare_features(train)
        X_val, y_val = self.prepare_features(val)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        models_config = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(
                n_estimators=self.config['RF_N_ESTIMATORS'],
                max_depth=self.config['RF_MAX_DEPTH'],
                min_samples_split=self.config['MIN_SAMPLES_SPLIT'],
                random_state=self.config['RANDOM_STATE'],
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.config['RANDOM_STATE']
            )
        }
        
        results = {}
        best_mae = float('inf')
        best_model_name = None
        
        for name, model in models_config.items():
            logger.info(f"Training model: {name}")
            
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                pred_val = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                pred_val = model.predict(X_val)
            
            mae = mean_absolute_error(y_val, pred_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred_val))
            r2 = r2_score(y_val, pred_val)
            mape = mean_absolute_percentage_error(y_val, pred_val) * 100
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'predictions': pred_val,
                'scaler_required': name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
            }
            
            logger.info(f"   MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")
            
            if mae < best_mae:
                best_mae = mae
                best_model_name = name
        
        results['best_model'] = best_model_name
        results['best_mae'] = best_mae
        
        self.models = results
        
        logger.info("-" * 50)
        logger.info(f"BEST MODEL: {best_model_name} (MAE = {best_mae:.2f})")
        logger.info("-" * 50)
        
        return results
    
    def evaluate_on_test(
        self, 
        test: pd.DataFrame,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.models:
            error_msg = "No models trained yet. Please run train_models() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        model_name = model_name or self.models['best_model']
        model_info = self.models[model_name]
        
        X_test, y_test = self.prepare_features(test)
        
        if model_info['scaler_required']:
            train, _, _ = self.split_data()
            X_train, y_train = self.prepare_features(train)
            scaler = RobustScaler()
            scaler.fit(X_train)
            X_test_scaled = scaler.transform(X_test)
            predictions = model_info['model'].predict(X_test_scaled)
        else:
            predictions = model_info['model'].predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        
        test_results = {
            'model_name': model_name,
            'actual': y_test,
            'predictions': predictions,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        logger.info("=" * 50)
        logger.info(f"TEST SET EVALUATION - {model_name}")
        logger.info("=" * 50)
        logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
        logger.info(f"Root Mean Square Error (RMSE): {rmse:.2f}")
        logger.info(f"Coefficient of Determination (R²): {r2:.4f}")
        logger.info(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        logger.info("=" * 50)
        
        self.results['test'] = test_results
        return test_results
    
    def create_visualizations(self, test: pd.DataFrame) -> None:
        if 'test' not in self.results:
            error_msg = "No test results yet. Please run evaluate_on_test() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        predictions = self.results['test']['predictions']
        actual = self.results['test']['actual']
        model_name = self.results['test']['model_name']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Sales Prediction Analysis - {model_name}', fontsize=14, fontweight='bold')
        
        ax1 = axes[0, 0]
        weeks = test["Week"].values
        ax1.plot(weeks, actual, 'b-o', label='Actual Sales', linewidth=2, markersize=6)
        ax1.plot(weeks, predictions, 'r--s', label=f'Predicted ({model_name})', linewidth=2, markersize=6)
        ax1.fill_between(weeks, actual, predictions, alpha=0.3, color='gray', label='Error Area')
        ax1.set_title('Actual vs Predicted Weekly Sales', fontsize=12)
        ax1.set_xlabel('Week Number')
        ax1.set_ylabel('Total Sales')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        errors = actual - predictions
        colors = ['green' if e >= 0 else 'red' for e in errors]
        ax2.bar(weeks, errors, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Prediction Errors (Actual - Predicted)', fontsize=12)
        ax2.set_xlabel('Week Number')
        ax2.set_ylabel('Error')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.hist(errors, bins=10, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax3.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(errors):.2f}')
        ax3.set_title('Error Distribution', fontsize=12)
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.scatter(actual, predictions, alpha=0.7, s=100, c='steelblue', edgecolors='black')
        min_val = min(actual.min(), predictions.min())
        max_val = max(actual.max(), predictions.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction Line')
        ax4.set_title('Actual vs Predicted (Scatter)', fontsize=12)
        ax4.set_xlabel('Actual Sales')
        ax4.set_ylabel('Predicted Sales')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = self.out_charts / "sales_prediction_analysis.png"
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        logger.info(f"Chart saved: {chart_path}")
        
        plt.show()
    
    def save_results(self, test: pd.DataFrame) -> None:
        predictions = self.results['test']['predictions']
        avg_price = self.config['AVG_PRICE_JOD']
        
        result_df = pd.DataFrame({
            "Week": test["Week"].values,
            "Actual_Sales": test["Total_Sales"].values,
            "Predicted_Sales": np.round(predictions, 2),
            "Error": np.round(test["Total_Sales"].values - predictions, 2),
            "Error_Percentage": np.round(
                (test["Total_Sales"].values - predictions) / test["Total_Sales"].values * 100, 2
            ),
            "Predicted_Revenue_JOD": np.round(predictions * avg_price, 2)
        })
        
        # Save CSV
        result_csv = self.out_results / "test_results.csv"
        result_df.to_csv(result_csv, index=False, encoding='utf-8-sig')
        logger.info(f"CSV saved: {result_csv}")
        
        # Save Excel
        out_xlsx = self.prep_dir / "Prepared_AI_Data_Output.xlsx"
        
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
            if self.processed_data is not None:
                self.processed_data.to_excel(writer, index=False, sheet_name='Prepared_Data')
            result_df.to_excel(writer, index=False, sheet_name='Test_Results')
            
            model_summary = pd.DataFrame([
                {'Model': name, 'MAE': info['mae'], 'RMSE': info['rmse'], 
                 'R2': info['r2'], 'MAPE_%': info['mape']}
                for name, info in self.models.items()
                if name not in ['best_model', 'best_mae']
            ])
            model_summary.to_excel(writer, index=False, sheet_name='Model_Comparison')
        
        logger.info(f"Excel saved: {out_xlsx}")
        
        # Generate text report
        self._generate_text_report(result_df)
    
    def _generate_text_report(self, result_df: pd.DataFrame) -> None:
        report_path = self.out_results / "prediction_report.txt"
        model_name = self.results['test']['model_name']
        test_metrics = self.results['test']
        
        report = f"""
================================================================================
                    WEEKLY SALES PREDICTION REPORT
================================================================================
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

1. MODEL SUMMARY
----------------
Selected Model: {model_name}

Performance Metrics:
- Mean Absolute Error (MAE): {test_metrics['mae']:.2f}
- Root Mean Square Error (RMSE): {test_metrics['rmse']:.2f}
- Coefficient of Determination (R²): {test_metrics['r2']:.4f}
- Mean Absolute Percentage Error (MAPE): {test_metrics['mape']:.2f}%

2. MODEL COMPARISON
-------------------
"""
        
        for name, info in self.models.items():
            if name not in ['best_model', 'best_mae']:
                report += f"  {name}:\n"
                report += f"    - MAE: {info['mae']:.2f}\n"
                report += f"    - RMSE: {info['rmse']:.2f}\n"
                report += f"    - R²: {info['r2']:.4f}\n"
                report += f"    - MAPE: {info['mape']:.2f}%\n\n"
        
        report += f"""
3. DETAILED PREDICTIONS
-----------------------
{result_df.to_string(index=False)}

4. PERFORMANCE ASSESSMENT
-------------------------
"""
        if test_metrics['mape'] < 5:
            report += "EXCELLENT: Model accuracy is very high"
        elif test_metrics['mape'] < 10:
            report += "GOOD: Model is reliable for practical use"
        elif test_metrics['mape'] < 20:
            report += "ACCEPTABLE: Model needs some improvement"
        else:
            report += "NEEDS IMPROVEMENT: Model requires significant tuning"
        
        report += """

================================================================================
                           END OF REPORT
================================================================================
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Text report saved: {report_path}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        logger.info("=" * 70)
        logger.info("STARTING SALES PREDICTION PIPELINE")
        logger.info("=" * 70)
        
        try:
            self.load_data()
            self.preprocess_data()
            train, val, test = self.split_data()
            self.train_models(train, val)
            self.evaluate_on_test(test)
            self.create_visualizations(test)
            self.save_results(test)
            
            logger.info("=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            
            return {
                'models': self.models,
                'test_results': self.results['test'],
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"ERROR: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'status': 'error', 'error': str(e)}


def main():
    config = {
        'AVG_PRICE_JOD': 2.5,
        'MA_WINDOW': 4,
        'TEST_SIZE': 0.18,
        'VAL_SIZE': 0.12,
        'RANDOM_STATE': 42,
        'RF_N_ESTIMATORS': 500,
        'RF_MAX_DEPTH': 10,
        'MIN_SAMPLES_SPLIT': 5
    }
    
    pipeline = SalesPredictionPipeline(config)
    results = pipeline.run_full_pipeline()
    
    if results['status'] == 'success':
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(f"Best Model: {results['test_results']['model_name']}")
        print(f"Mean Absolute Error (MAE): {results['test_results']['mae']:.2f}")
        print(f"Root Mean Square Error (RMSE): {results['test_results']['rmse']:.2f}")
        print(f"Coefficient of Determination (R²): {results['test_results']['r2']:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {results['test_results']['mape']:.2f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()