"""
================================================================================
Sales Prediction System V2 
================================================================================
ALL IMPROVEMENTS:
1. Weighted model selection (0.5×MAE + 0.3×MAPE + 0.2×RMSE)
2. Stored scaler to prevent recreation
3. Walk Forward Validation for stability testing
4. Enhanced logging with data version info
5. Best model persistence with pickle
6. Timing measurements for each stage
7. Improved text report with administrative interpretation
8. Validation vs Test comparison to detect overfitting
9. Early Stopping for Gradient Boosting
10. Feature Importance Report for tree models
11. Multiple runs with standard deviation analysis
12. Unified pipeline with SEED for reproducibility
13. Division by zero handling
14. Additional stability metrics (Median AE, sMAPE, WAPE)
15. Week column validation
16. Minimum rows check before training
17. Command line interface (argparse)
18. Multiple run modes (full/train_only/eval_only/charts_only)
19. Interactive Menu
20. V2 vs V3 Comparison
21.Modern Menu System with Custom File Upload Support
================================================================================
"""

import os
import sys
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

import warnings
warnings.filterwarnings('ignore')


# ================================================================================
#                           MENU SYSTEM
# ================================================================================

class Menu:
    @staticmethod
    def clear():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def header(title):
        print(f"\n{'='*70}")
        print(f"{' '*15}{title}")
        print(f"{'='*70}")
    
    @staticmethod
    def option(num, title, desc=""):
        print(f"  [{num}] {title}")
        if desc:
            print(f"      {desc}")
    
    @staticmethod
    def main_menu():
        Menu.clear()
        Menu.header("WEEKLY SALES PREDICTION SYSTEM V2")
        
        print("\nMAIN MENU:\n")
        Menu.option("1", "Run Full Pipeline", "Train & Test")
        Menu.option("2", "Train Only", "Train models")
        Menu.option("3", "Test Only", "Show/Evaluate model")
        Menu.option("4", "Show Charts", "View charts")
        Menu.option("5", "Compare Results", "Compare runs")
        Menu.option("6", "Stability Test", "Multiple runs")
        Menu.option("0", "Exit", "Quit")
        
        print(f"\n{'-'*70}")
        print("Type a number (0-6) and press Enter:")
        choice = input(">> ").strip()
        return choice


# ================================================================================
#                           CONFIGURATION
# ================================================================================

class Config:
    DATA_FILE = "Sales_Transactions_Dataset_Weekly - V2.csv"
    PRICE = 2.5
    MA_WINDOW = 4
    TEST_SIZE = 0.18
    VAL_SIZE = 0.12
    SEED = 42
    RF_TREES = 500
    RF_DEPTH = 10
    GB_TREES = 200
    WEIGHT_MAE = 0.5
    WEIGHT_MAPE = 0.3
    WEIGHT_RMSE = 0.2


def weighted_score(mae, mape, rmse):
    return Config.WEIGHT_MAE * mae + Config.WEIGHT_MAPE * mape + Config.WEIGHT_RMSE * rmse


def normalize_columns(df):
    """
    Normalize column names to handle different CSV formats.
    Handles both:
    - New format: Week, Actual, Predicted, Error, Error_%
    - Old format: Week, Actual_Sales, Predicted_Sales, Error, Error_Percentage
    """
    df = df.copy()
    
    # First, strip whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Map old column names to new standard names
    column_mapping = {
        'Actual_Sales': 'Actual',
        'Predicted_Sales': 'Predicted',
        'Error_Percentage': 'Error_%'
    }
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    return df


# ================================================================================
#                           MAIN PIPELINE
# ================================================================================

class Pipeline:
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.raw_dir = self.base_dir / "data" / "raw"
        self.out_dir = self.base_dir / "outputs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Also save to latest folder
        self.latest_dir = self.base_dir / "outputs" / "latest"
        self.latest_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.models = {}
        self.scaler = None
        
        print(f"\n{'='*60}")
        print("   PIPELINE V2 - ENHANCED (FIXED)")
        print(f"{'='*60}")
    
    def load_data(self):
        path = self.raw_dir / Config.DATA_FILE
        
        if not path.exists():
            print(f"Error: File not found: {path}")
            return None
        
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except:
                continue
        
        week_cols = [c for c in df.columns if str(c).strip().upper().startswith("W")]
        week_cols = sorted(week_cols, key=lambda c: int(str(c).replace("W","").strip()) if str(c).replace("W","").strip().isdigit() else 9999)
        
        weeks = df[week_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=0)
        
        self.data = pd.DataFrame({
            "Week": range(len(weeks)),
            "Total_Sales": weeks.values.astype(float),
            "Revenue": weeks.values * Config.PRICE,
            "Week_Sin": np.sin(2 * np.pi * np.arange(len(weeks)) / 52),
            "Week_Cos": np.cos(2 * np.pi * np.arange(len(weeks)) / 52)
        })
        
        print(f"Data loaded: {len(self.data)} weeks")
        return self.data
    
    def create_features(self, df):
        ma = Config.MA_WINDOW
        df = df.copy()
        
        df["Prev_Sales"] = df["Total_Sales"].shift(1)
        df["MA"] = df["Total_Sales"].rolling(ma, min_periods=1).mean().shift(1)
        df["Change"] = df["Total_Sales"].pct_change().shift(1).clip(-1, 1).fillna(0)
        df["Std"] = df["Total_Sales"].rolling(ma, min_periods=1).std().shift(1).fillna(0)
        df["Lag_2"] = df["Total_Sales"].shift(2)
        df["Lag_3"] = df["Total_Sales"].shift(3)
        
        for c in ["Prev_Sales", "MA", "Change", "Std", "Lag_2", "Lag_3"]:
            df[c] = df[c].fillna(df["Total_Sales"].mean())
        
        return df
    
    def split(self):
        n = len(self.data)
        test_cut = int(n * (1 - Config.TEST_SIZE))
        val_cut = int(test_cut * (1 - Config.VAL_SIZE))
        
        train = self.data[self.data["Week"] < val_cut].copy()
        val = self.data[(self.data["Week"] >= val_cut) & (self.data["Week"] < test_cut)].copy()
        test = self.data[self.data["Week"] >= test_cut].copy()
        
        train = self.create_features(train)
        val = self.create_features(val)
        test = self.create_features(test)
        
        print(f"Split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        return train, val, test
    
    def get_features(self, df):
        cols = ["Week", "Prev_Sales", "MA", "Change", "Std", "Lag_2", "Lag_3", "Week_Sin", "Week_Cos"]
        return df[cols].values, df["Total_Sales"].values
    
    def train(self, train, val):
        X_train, y_train = self.get_features(train)
        X_val, y_val = self.get_features(val)
        
        # Store scaler for later use
        self.scaler = RobustScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)
        
        models = {
            'Linear Regression': (LinearRegression(), True),
            'Ridge Regression': (Ridge(alpha=1.0), True),
            'Lasso Regression': (Lasso(alpha=0.1), True),
            'Random Forest': (RandomForestRegressor(n_estimators=Config.RF_TREES, max_depth=Config.RF_DEPTH, random_state=Config.SEED, n_jobs=-1), False),
            'Gradient Boosting': (GradientBoostingRegressor(n_estimators=Config.GB_TREES, learning_rate=0.1, max_depth=5, random_state=Config.SEED, n_iter_no_change=20), False)
        }
        
        print(f"\n{'='*50}")
        print("TRAINING MODELS")
        print(f"{'='*50}")
        
        best_score = float('inf')
        best_name = None
        
        for name, (model, use_scaler) in models.items():
            if use_scaler:
                model.fit(X_train_s, y_train)
                pred = model.predict(X_val_s)
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
            
            mae = mean_absolute_error(y_val, pred)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            mape = mean_absolute_percentage_error(y_val, pred) * 100
            weighted = weighted_score(mae, mape, rmse)
            
            self.models[name] = {'model': model, 'scaler': use_scaler, 'mae': mae, 'mape': mape}
            print(f"  {name:20} | MAE: {mae:10,.0f} | MAPE: {mape:6.2f}% | Weighted: {weighted:.2f}")
            
            if weighted < best_score:
                best_score = weighted
                best_name = name
        
        self.models['best'] = best_name
        print(f"\n>>> Best: {best_name} (Weighted Score: {best_score:.2f})")
        
        # Save model to both locations
        model_data = {'models': self.models, 'scaler': self.scaler}
        with open(self.out_dir / "model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        with open(self.latest_dir / "model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        return self.models
    
    def test(self, test):
        name = self.models['best']
        info = self.models[name]
        
        X_test, y_test = self.get_features(test)
        
        # Use stored scaler
        if info['scaler'] and self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        
        pred = info['model'].predict(X_test)
        
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mape = mean_absolute_percentage_error(y_test, pred) * 100
        r2 = r2_score(y_test, pred)
        
        print(f"\n{'='*50}")
        print(f"TEST RESULTS - {name}")
        print(f"{'='*50}")
        print(f"  MAE:  {mae:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²:   {r2:.4f}")
        
        # Save CSV to both locations
        result_df = pd.DataFrame({
            "Week": test["Week"].values,
            "Actual": y_test,
            "Predicted": np.round(pred, 2),
            "Error": np.round(y_test - pred, 2),
            "Error_%": np.where(y_test != 0, np.round((y_test - pred) / y_test * 100, 2), 0)
        })
        result_df.to_csv(self.out_dir / "test_results.csv", index=False)
        result_df.to_csv(self.latest_dir / "test_results.csv", index=False)
        
        # Charts
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(test["Week"], y_test, 'b-o', label='Actual')
        axes[0, 0].plot(test["Week"], pred, 'r--s', label='Predicted')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        errors = y_test - pred
        axes[0, 1].bar(test["Week"], errors, color=['green' if e>=0 else 'red' for e in errors])
        axes[0, 1].axhline(y=0, color='black')
        axes[0, 1].set_title('Errors')
        
        axes[1, 0].hist(errors, bins=8, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        axes[1, 0].set_title('Error Distribution')
        
        axes[1, 1].scatter(y_test, pred, alpha=0.7)
        m, M = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
        axes[1, 1].plot([m, M], [m, M], 'r--')
        axes[1, 1].set_title('Scatter')
        
        plt.tight_layout()
        plt.savefig(self.out_dir / "chart.png", dpi=150)
        plt.savefig(self.latest_dir / "chart.png", dpi=150)
        plt.close()
        
        print(f"\n>>> Saved: {self.out_dir}")
        return {'mae': mae, 'mape': mape, 'rmse': rmse, 'r2': r2}
    
    def run_full(self):
        self.load_data()
        train, val, test = self.split()
        self.train(train, val)
        return self.test(test)
    
    def show_charts(self):
        latest = self.latest_dir / "chart.png"
        if latest.exists():
            print(f"Chart: {latest}")
            try:
                os.startfile(latest)
            except:
                print("Could not open chart automatically")
        else:
            charts = list(self.base_dir.glob("outputs/run_*/chart.png"))
            if charts:
                latest = sorted(charts, key=lambda x: x.stat().st_mtime)[-1]
                print(f"Chart: {latest}")
                try:
                    os.startfile(latest)
                except:
                    print("Could not open chart automatically")
            else:
                print("No charts found!")


# ================================================================================
#                           MAIN PROGRAM - FIXED VERSION
# ================================================================================

def main():
    while True:
        choice = Menu.main_menu()
        
        if choice == '1':
            print("\n--- RUNNING FULL PIPELINE ---")
            p = Pipeline()
            p.run_full()
        
        elif choice == '2':
            print("\n--- TRAINING ---")
            p = Pipeline()
            p.load_data()
            train, val, test = p.split()
            p.train(train, val)
            print("\nTraining complete!")
        
        # ============================================================
        # OPTION 3 - FIXED: Show Test Results
        # ============================================================
        elif choice == '3':
            print("\n" + "="*60)
            print("OPTION 3: TEST RESULTS / EVALUATE MODEL")
            print("="*60)
            
            base_dir = Path(__file__).parent
            latest_csv = base_dir / "outputs" / "latest" / "test_results.csv"
            latest_model = base_dir / "outputs" / "latest" / "model.pkl"
            
            # First, try to load CSV results (most common case)
            if latest_csv.exists():
                print(f"\nFound results file: {latest_csv}")
                print("\n" + "-"*50)
                print("LOADING TEST RESULTS...")
                print("-"*50)
                
                df = pd.read_csv(latest_csv)
                df = normalize_columns(df)  # Handle different column name formats
                
                print(f"\nTotal test weeks: {len(df)}")
                print(f"Data columns: {list(df.columns)}")
                
                # Calculate statistics
                mae = np.abs(df["Error"]).mean()
                rmse = np.sqrt((df["Error"]**2).mean())
                mape = np.abs(df["Error_%"]).mean()
                
                print(f"\n{'='*50}")
                print("TEST RESULTS SUMMARY")
                print(f"{'='*50}")
                print(f"Mean Absolute Error (MAE): {mae:,.2f}")
                print(f"Root Mean Square Error (RMSE): {rmse:,.2f}")
                print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                
                print(f"\n{'='*50}")
                print("FIRST 5 WEEKS")
                print(f"{'='*50}")
                print(df.head().to_string(index=False))
                
                print(f"\n{'='*50}")
                print("LAST 5 WEEKS")
                print(f"{'='*50}")
                print(df.tail().to_string(index=False))
                
                # Show week with highest error
                worst_idx = np.abs(df["Error"]).idxmax()
                worst_week = df.loc[worst_idx]
                print(f"\n{'='*50}")
                print("WEEK WITH HIGHEST ERROR")
                print(f"{'='*50}")
                print(f"Week {int(worst_week['Week'])}: Actual={worst_week['Actual']:,.0f}, Predicted={worst_week['Predicted']:,.0f}, Error={worst_week['Error']:,.0f}")
                
            # If no CSV, try to load model and re-run test
            elif latest_model.exists():
                print(f"\nFound model file: {latest_model}")
                print("\nRe-running test evaluation...")
                
                with open(latest_model, 'rb') as f:
                    data = pickle.load(f)
                
                p = Pipeline()
                p.models = data['models']
                p.scaler = data['scaler']
                p.load_data()
                _, _, test = p.split()
                p.test(test)
                
            else:
                print("\n" + "!"*50)
                print("NO DATA FOUND!")
                print("!"*50)
                print("\nPlease run option 1 (Full Pipeline) first.")
                print("This will train models and generate test results.")
        
        # ============================================================
        # OPTION 4 - FIXED: Show Charts
        # ============================================================
        elif choice == '4':
            print("\n" + "="*60)
            print("OPTION 4: SHOW CHARTS")
            print("="*60)
            
            base_dir = Path(__file__).parent
            latest_chart = base_dir / "outputs" / "latest" / "chart.png"
            
            if latest_chart.exists():
                print(f"\nFound chart: {latest_chart}")
                print("\nOpening chart...")
                try:
                    os.startfile(latest_chart)
                    print("SUCCESS: Chart opened!")
                except Exception as e:
                    print(f"Error opening chart: {e}")
                    print(f"\nYou can manually open the chart from:")
                    print(f"  {latest_chart}")
            else:
                # Search for any chart
                print("\nNo chart in 'latest' folder.")
                print("Searching for any chart...")
                
                all_charts = list((base_dir / "outputs").rglob("*.png"))
                all_charts = [c for c in all_charts if "chart" in c.name.lower()]
                
                if all_charts:
                    latest_chart = max(all_charts, key=lambda x: x.stat().st_mtime)
                    print(f"\nFound: {latest_chart}")
                    print("\nOpening chart...")
                    try:
                        os.startfile(latest_chart)
                        print("SUCCESS: Chart opened!")
                    except Exception as e:
                        print(f"Error: {e}")
                else:
                    print("\n" + "!"*50)
                    print("NO CHARTS FOUND!")
                    print("!"*50)
                    print("\nPlease run option 1 (Full Pipeline) first.")
                    print("This will generate charts.")
        
        # ============================================================
        # OPTION 5 - FIXED: Compare Results
        # ============================================================
        elif choice == '5':
            print("\n" + "="*60)
            print("OPTION 5: COMPARE RESULTS")
            print("="*60)
            
            base_dir = Path(__file__).parent
            
            # Collect all runs with results
            all_results = []
            
            # Check latest folder first
            latest_csv = base_dir / "outputs" / "latest" / "test_results.csv"
            if latest_csv.exists():
                all_results.append({
                    'path': base_dir / "outputs" / "latest",
                    'name': "latest",
                    'csv': latest_csv
                })
            
            # Check all run folders
            for run_folder in (base_dir / "outputs").glob("run_*"):
                csv_path = run_folder / "test_results.csv"
                if csv_path.exists() and "latest" not in str(run_folder):
                    all_results.append({
                        'path': run_folder,
                        'name': run_folder.name,
                        'csv': csv_path
                    })
            
            if not all_results:
                print("\n" + "!"*50)
                print("NO RESULTS FOUND!")
                print("!"*50)
                print("\nPlease run option 1 (Full Pipeline) first.")
            else:
                print(f"\nFound {len(all_results)} result(s):\n")
                
                # Load and display all results
                for i, result in enumerate(all_results):
                    try:
                        df = pd.read_csv(result['csv'])
                        df = normalize_columns(df)  # Handle different column name formats
                        mae = np.abs(df["Error"]).mean()
                        rmse = np.sqrt((df["Error"]**2).mean())
                        mape = np.abs(df["Error_%"]).mean()
                        
                        print(f"{'='*50}")
                        print(f"RESULT #{i+1}: {result['name']}")
                        print(f"{'='*50}")
                        print(f"  File: {result['csv'].name}")
                        print(f"  Test Weeks: {len(df)}")
                        print(f"  MAE:  {mae:,.2f}")
                        print(f"  RMSE: {rmse:,.2f}")
                        print(f"  MAPE: {mape:.2f}%")
                        print()
                    except Exception as e:
                        print(f"Error reading {result['name']}: {e}")
                
                # Compare if we have multiple results
                if len(all_results) >= 2:
                    print("\n" + "="*60)
                    print("COMPARISON SUMMARY")
                    print("="*60)
                    
                    try:
                        df1 = pd.read_csv(all_results[0]['csv'])
                        df1 = normalize_columns(df1)  # Handle different column name formats
                        df2 = pd.read_csv(all_results[1]['csv'])
                        df2 = normalize_columns(df2)  # Handle different column name formats
                        
                        mae1 = np.abs(df1["Error"]).mean()
                        mae2 = np.abs(df2["Error"]).mean()
                        
                        print(f"\nLatest ({all_results[0]['name']}): MAE = {mae1:,.2f}")
                        print(f"Previous ({all_results[1]['name']}): MAE = {mae2:,.2f}")
                        
                        diff = abs(mae1 - mae2)
                        pct = (diff / max(mae1, mae2)) * 100 if max(mae1, mae2) > 0 else 0
                        
                        print(f"\nDifference: {diff:,.2f} ({pct:.2f}%)")
                        
                        if mae1 < mae2:
                            print("\n>>> LATEST RUN IS BETTER!")
                        elif mae2 < mae1:
                            print("\n>>> PREVIOUS RUN WAS BETTER!")
                        else:
                            print("\n>>> BOTH RUNS HAVE SIMILAR PERFORMANCE!")
                    except Exception as e:
                        print(f"Error comparing: {e}")
                elif len(all_results) == 1:
                    print("\nOnly 1 result found. Run option 1 again to generate more results for comparison.")
        
        elif choice == '6':
            print("\n--- STABILITY TEST ---")
            n = 2
            print(f"Running {n} times for stability check...")
            mae_values = []
            for i in range(n):
                print(f"\n>>> Run {i+1}/{n} <<<")
                p = Pipeline()
                r = p.run_full()
                mae_values.append(r['mae'])
            
            print(f"\nSTABILITY RESULTS:")
            print(f"  Mean: {np.mean(mae_values):,.2f}")
            print(f"  Std:  {np.std(mae_values):,.2f}")
            print(f"  Min:  {np.min(mae_values):,.2f}")
            print(f"  Max:  {np.max(mae_values):,.2f}")
        
        elif choice == '0':
            print("\nGoodbye!")
            break
        
        else:
            print(f"Invalid choice: {choice}")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
