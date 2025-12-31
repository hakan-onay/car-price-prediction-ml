import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")
plt.style.use("ggplot")
sns.set_palette("husl")
np.random.seed(42)



# Plot helpers (SAVE ONLY)
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tl_formatter(x, pos):
    """Format large numbers as TL with K/M suffix for readability."""
    x = float(x)
    ax = abs(x)
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{x:.0f}"


def apply_price_axis_format(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(tl_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(tl_formatter))


def save_fig(fig, out_dir: str, filename: str, dpi: int = 300):
    """Save figure with consistent high quality settings and close it."""
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def clip_series_for_plot(series: pd.Series, low_q=0.01, high_q=0.99):
    """Clip only for plotting (keeps the core distribution readable)."""
    low = series.quantile(low_q)
    high = series.quantile(high_q)
    return series.clip(lower=low, upper=high), low, high



# Main
def main():
    print("=" * 80)
    print("CAR PRICE PREDICTION PROJECT - PROCESS STEPS")
    print("=" * 80)

    plots_dir = os.path.join(os.getcwd(), "plots")
    ensure_dir(plots_dir)

    # 1. DATA LOADING
    print("\n1. DATA LOADING")
    print("-" * 40)
    try:
        df = pd.read_csv("cardekho_dataset.csv")
        print("Dataset loaded successfully")
        print(f"  - Initial dataset shape: {df.shape}")
        print(f"  - Number of columns: {len(df.columns)}")
        print(f"  - Number of rows: {len(df)}")
        print(f"  - Column names: {list(df.columns)}")
    except FileNotFoundError:
        print("ERROR: 'cardekho_dataset.csv' file not found.")
        return

    print("\n  First 3 records:")
    print(df.head(3).to_string())

    # 2. DATA CLEANING
    print("\n\n2. DATA CLEANING")
    print("-" * 40)

    print("Removing unnecessary columns...")
    initial_cols = len(df.columns)
    df = df.drop(["Unnamed: 0", "car_name"], axis=1, errors="ignore")
    final_cols = len(df.columns)
    print(f"  - Columns removed: {initial_cols - final_cols}")
    print(f"  - Remaining columns: {final_cols}")
    print(f"  - Remaining columns: {list(df.columns)}")

    print("\nChecking for missing data...")
    missing_data = df.isnull().sum()
    total_missing = int(missing_data.sum())
    print(f"  - Total missing data: {total_missing}")

    if total_missing > 0:
        print("  - Distribution of missing data by column:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"    * {col}: {count} missing ({count / len(df) * 100:.1f}%)")

    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    removed_rows = initial_rows - final_rows
    print(f"  - Rows cleaned: {removed_rows}")
    print(f"  - Remaining rows: {final_rows}")
    print(f"  - Percentage of rows cleaned: {removed_rows / initial_rows * 100:.2f}%")

    print("\nUnique value analysis...")
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count < 20:
            print(f"  - {col}: {unique_count} unique values")

    # 3. OUTLIER ANALYSIS AND CLEANING
    print("\n\n3. OUTLIER ANALYSIS")
    print("-" * 40)

    print("Target variable (selling_price) analysis...")
    price_stats = df["selling_price"].describe()
    print(f"  - Minimum price: {price_stats['min']:,.0f} TL")
    print(f"  - Maximum price: {price_stats['max']:,.0f} TL")
    print(f"  - Average price: {price_stats['mean']:,.0f} TL")
    print(f"  - Median price: {price_stats['50%']:,.0f} TL")
    print(f"  - Standard deviation: {price_stats['std']:,.0f} TL")

    Q1 = df["selling_price"].quantile(0.25)
    Q3 = df["selling_price"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print("\n  Outlier boundaries (IQR method):")
    print(f"  - Q1 (25th percentile): {Q1:,.0f} TL")
    print(f"  - Q3 (75th percentile): {Q3:,.0f} TL")
    print(f"  - IQR: {IQR:,.0f} TL")
    print(f"  - Lower bound: {lower_bound:,.0f} TL")
    print(f"  - Upper bound: {upper_bound:,.0f} TL")

    outliers = df[(df["selling_price"] < lower_bound) | (df["selling_price"] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percentage = outlier_count / len(df) * 100

    print("\n  Outlier statistics:")
    print(f"  - Number of outliers: {outlier_count}")
    print(f"  - Percentage of outliers: {outlier_percentage:.1f}%")

    if outlier_count > 0:
        print(
            f"  - Price range of outliers: "
            f"{outliers['selling_price'].min():,.0f} - {outliers['selling_price'].max():,.0f} TL"
        )

    print("\nCleaning outliers...")
    initial_size = len(df)
    df = df[(df["selling_price"] >= lower_bound) & (df["selling_price"] <= upper_bound)]
    final_size = len(df)
    removed_outliers = initial_size - final_size

    print(f"  - Outliers removed: {removed_outliers}")
    print(f"  - Remaining data: {final_size}")
    print(f"  - Percentage of data cleaned: {removed_outliers / initial_size * 100:.1f}%")

    clean_price_stats = df["selling_price"].describe()
    print("\n  Price statistics after cleaning:")
    print(f"  - New minimum price: {clean_price_stats['min']:,.0f} TL")
    print(f"  - New maximum price: {clean_price_stats['max']:,.0f} TL")
    print(f"  - New average price: {clean_price_stats['mean']:,.0f} TL")
    print(f"  - New standard deviation: {clean_price_stats['std']:,.0f} TL")

    # 4. DATA TYPES AND PREPROCESSING
    print("\n\n4. DATA TYPES AND PREPROCESSING")
    print("-" * 40)

    print("Variable type analysis...")
    categorical_cols = ["brand", "seller_type", "fuel_type", "transmission_type"]
    numerical_cols = [c for c in df.columns if c not in categorical_cols + ["selling_price", "model"]]

    print(f"  - Categorical variables ({len(categorical_cols)}): {categorical_cols}")
    print(f"  - Numerical variables ({len(numerical_cols)}): {numerical_cols}")
    print("  - Target variable: selling_price")
    print("  - Special variable to process: model")

    print("\nProcessing 'model' column...")
    model_unique_count = df["model"].nunique()
    print(f"  - Number of unique values in 'model' column: {model_unique_count}")
    print("  - Applying Label Encoding due to too many categories")

    le_model = LabelEncoder()
    df["model_encoded"] = le_model.fit_transform(df["model"])
    numerical_cols.append("model_encoded")

    print("  - Label Encoding completed")
    print("  - New numerical variable: model_encoded")
    print(f"  - Current number of numerical variables: {len(numerical_cols)}")

    # 5. SEPARATING TARGET VARIABLE AND FEATURES
    print("\n\n5. SEPARATING TARGET VARIABLE AND FEATURES")
    print("-" * 40)

    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]

    print("Features (X) and target variable (y) separated")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - Total number of features: {X.shape[1]}")

    # 6. DATA VISUALIZATION (SKIP INITIAL MULTIPLOT)
    print("\n\n6. DATA VISUALIZATION")
    print("-" * 40)
    print("Initial exploratory multi-plot is skipped as requested.")

    # Correlation matrix (SAVED)
    print("Creating correlation matrix (saved to file)...")
    fig = plt.figure(figsize=(12, 10))
    numerical_df = df[numerical_cols + ["selling_price"]]
    correlation_matrix = numerical_df.corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    ax = sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("CORRELATION MATRIX OF NUMERICAL VARIABLES", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, plots_dir, "correlation_matrix.png")

    # 7. SPLITTING DATA INTO TRAIN AND TEST SETS
    print("\n\n7. SPLITTING DATA INTO TRAIN AND TEST SETS")
    print("-" * 40)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data successfully split:")
    print(f"  - Training set (X_train): {X_train.shape}")
    print(f"  - Test set (X_test): {X_test.shape}")
    print(f"  - Training target (y_train): {y_train.shape}")
    print(f"  - Test target (y_test): {y_test.shape}")
    print("  - Split ratio: 80% training, 20% test")

    # remove raw model text, keep encoded column
    X_train = X_train.drop("model", axis=1)
    X_test = X_test.drop("model", axis=1)

    print("\n'model' column removed (model_encoded will be used)")
    print(f"  - Current X_train shape: {X_train.shape}")
    print(f"  - Current X_test shape: {X_test.shape}")

    # 8. CREATING PREPROCESSOR
    print("\n\n8. CREATING PREPROCESSOR")
    print("-" * 40)
    print("Creating preprocessor...")

    try:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ]
        )
        print("  - Using current scikit-learn version (sparse_output=False)")
    except TypeError:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )
        print("  - Using older scikit-learn version")

    print("  - For numerical variables: StandardScaler")
    print("  - For categorical variables: OneHotEncoder")
    print("  - Total transformation steps: 2")

    # 9. CREATING MODELS
    print("\n\n9. CREATING MODELS")
    print("-" * 40)
    print("Creating 5 different models...")

    models = {
        "Linear Regression": Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())]),
        "Ridge Regression": Pipeline([("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0, random_state=42))]),
        "Lasso Regression": Pipeline(
            [("preprocessor", preprocessor), ("regressor", Lasso(alpha=0.01, random_state=42, max_iter=10000))]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=15,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42
                    ),
                ),
            ]
        ),
    }

    print("  - Models created:")
    for i, name in enumerate(models.keys(), 1):
        print(f"    {i}. {name}")

    # 10. MODEL TRAINING AND EVALUATION + SAVE PLOTS
    print("\n\n10. MODEL TRAINING AND EVALUATION")
    print("-" * 40)

    results = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nEvaluating {name} model...")
        print("  - Training model...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)

        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2", n_jobs=-1)

        results.append(
            {
                "Model": name,
                "Train R2": train_r2,
                "Test R2": test_r2,
                "R2 Difference": train_r2 - test_r2,
                "CV R2 Mean": cv_scores.mean(),
                "CV R2 Std": cv_scores.std(),
                "RMSE": rmse,
                "MAE": mae,
            }
        )

        print(f"  - Training R²: {train_r2:.4f}")
        print(f"  - Test R²: {test_r2:.4f}")
        print(f"  - R² Difference: {train_r2 - test_r2:.4f}")
        print(f"  - CV R² Average: {cv_scores.mean():.4f}")
        print(f"  - RMSE: {rmse:,.0f} TL")
        print(f"  - MAE: {mae:,.0f} TL")

        # Performance plots (SAVED)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Actual vs Predicted
        axes[0].scatter(y_test, y_test_pred, alpha=0.25, s=10, edgecolor="black", linewidth=0.3)
        minv, maxv = float(y_test.min()), float(y_test.max())
        axes[0].plot([minv, maxv], [minv, maxv], "r--", lw=2, label="Ideal")
        axes[0].set_xlabel("Actual Price (TL)")
        axes[0].set_ylabel("Predicted Price (TL)")
        axes[0].set_title(f"{name}\nTest R²: {test_r2:.3f}")
        axes[0].legend()
        apply_price_axis_format(axes[0])

        # Right: Error distribution (clipped for plotting readability)
        errors = y_test - y_test_pred
        errors_plot, lo, hi = clip_series_for_plot(pd.Series(errors), 0.01, 0.99)

        axes[1].hist(errors_plot, bins=30, edgecolor="black", alpha=0.7)
        axes[1].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[1].set_xlabel("Error (Actual - Predicted) (TL)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Error Distribution\nRMSE: {rmse:,.0f} TL\n(Plot clipped to 1–99%)")
        axes[1].xaxis.set_major_formatter(FuncFormatter(tl_formatter))

        fig.suptitle(f"{name} PERFORMANCE ANALYSIS", fontsize=14, fontweight="bold")
        fig.tight_layout()

        safe_name = name.lower().replace(" ", "_")
        save_fig(fig, plots_dir, f"{safe_name}_performance.png")

    # 11. RESULTS TABLE
    print("\n\n11. RESULTS TABLE")
    print("=" * 80)

    results_df = pd.DataFrame(results).sort_values("Test R2", ascending=False)

    print("\nMODEL PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    # 12. OVERFITTING CONTROL GRAPH (SAVED)
    print("\n\n12. OVERFITTING CONTROL ANALYSIS")
    print("-" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    models_list = results_df["Model"].values
    x_pos = np.arange(len(models_list))

    axes[0].bar(x_pos - 0.2, results_df["Train R2"], width=0.4, label="Train R²", edgecolor="black", alpha=0.8)
    axes[0].bar(x_pos + 0.2, results_df["Test R2"], width=0.4, label="Test R²", edgecolor="black", alpha=0.8)

    axes[0].set_xlabel("Models")
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("Train vs Test R² Comparison")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(models_list, rotation=45, ha="right")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    diffs = results_df["R2 Difference"].values
    bars = axes[1].bar(models_list, diffs, edgecolor="black")
    axes[1].axhline(y=0.15, color="r", linestyle="--", alpha=0.7, label="High Risk Threshold")
    axes[1].axhline(y=0.05, color="orange", linestyle="--", alpha=0.7, label="Medium Risk Threshold")

    axes[1].set_xlabel("Models")
    axes[1].set_ylabel("R² Difference (Train - Test)")
    axes[1].set_title("Overfitting Level Analysis")
    axes[1].set_xticklabels(models_list, rotation=45, ha="right")
    axes[1].legend()

    for bar, diff in zip(bars, diffs):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{diff:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("OVERFITTING CONTROL ANALYSIS", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, plots_dir, "overfitting_control_analysis.png")

    # 13. BEST MODEL ANALYSIS
    print("\n\n13. BEST MODEL ANALYSIS")
    print("=" * 80)

    best_model_info = results_df.iloc[0]
    best_model_name = best_model_info["Model"]
    best_model = models[best_model_name]

    print(f"\nBEST MODEL: {best_model_name}")
    print("-" * 40)
    print(f"Test R²: {best_model_info['Test R2']:.4f}")
    print(f"Train R²: {best_model_info['Train R2']:.4f}")
    print(f"R² Difference: {best_model_info['R2 Difference']:.4f}")
    print(f"CV R² (Average): {best_model_info['CV R2 Mean']:.4f}")
    print(f"CV R² (Std): {best_model_info['CV R2 Std']:.4f}")
    print(f"RMSE: {best_model_info['RMSE']:,.0f} TL")
    print(f"MAE: {best_model_info['MAE']:,.0f} TL")

    diff = best_model_info["R2 Difference"]
    if diff > 0.15:
        print("\nWARNING: Significant overfitting detected in the model!")
    elif diff > 0.05:
        print("\nINFO: Slight overfitting risk in the model.")
    else:
        print("\nSUCCESS: Model is balanced and reliable.")

    # 14. FEATURE IMPORTANCE ANALYSIS (SAVED)
    print("\n\n14. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 80)

    feat_imp_df = None
    if best_model_name in ["Random Forest", "Gradient Boosting"]:
        print(f"\n{best_model_name} model feature importance analysis:")

        try:
            if hasattr(best_model.named_steps["preprocessor"], "transformers_"):
                feature_names = []
                feature_names.extend(numerical_cols)

                cat_encoder = best_model.named_steps["preprocessor"].named_transformers_["cat"]
                for i, col in enumerate(categorical_cols):
                    categories = cat_encoder.categories_[i]
                    for cat in categories:
                        feature_names.append(f"{col}_{cat}")

                importances = best_model.named_steps["regressor"].feature_importances_

                feat_imp_df = (
                    pd.DataFrame({"Feature": feature_names, "Importance": importances})
                    .sort_values("Importance", ascending=False)
                    .head(15)
                )

                print("\nTop 15 Most Important Features:")
                print("-" * 40)
                print(feat_imp_df.to_string(index=False))

                fig = plt.figure(figsize=(12, 8))
                ax = plt.gca()
                ax.barh(feat_imp_df["Feature"][::-1], feat_imp_df["Importance"][::-1], edgecolor="black", linewidth=0.5)
                ax.set_xlabel("Importance Score", fontsize=12)
                ax.set_title(f"{best_model_name} - FEATURE IMPORTANCE LEVELS (TOP 15)", fontsize=14, fontweight="bold")
                fig.tight_layout()
                save_fig(fig, plots_dir, f"{best_model_name.lower().replace(' ', '_')}_feature_importance.png")

        except Exception as e:
            print(f"Feature importance analysis error: {e}")

    # 15. PRICE SEGMENT ANALYSIS
    print("\n\n15. PRICE SEGMENT ANALYSIS")
    print("-" * 80)

    y_test_df = pd.DataFrame({"Actual": y_test, "Predicted": best_model.predict(X_test)})
    y_test_df["Error"] = y_test_df["Actual"] - y_test_df["Predicted"]
    y_test_df["Error Percentage"] = (y_test_df["Error"] / y_test_df["Actual"]) * 100

    bins = [0, 300000, 600000, 1000000, 2000000, 5000000, 15000000]
    labels = ["Very Low", "Low", "Medium", "High", "Very High", "Premium"]
    y_test_df["Price Segment"] = pd.cut(y_test_df["Actual"], bins=bins, labels=labels)

    segment_stats = y_test_df.groupby("Price Segment").agg({"Actual": "count", "Error Percentage": ["mean", "std"]}).round(2)

    print("\nSegment-Based Error Analysis:")
    print("-" * 40)
    print(segment_stats)

    # 16. PROJECT SUMMARY
    print("\n\n" + "=" * 80)
    print("PROJECT SUMMARY")
    print("=" * 80)

    print("\nDATA PROCESSING SUMMARY:")
    print("   - Initial dataset size: 15,411 records")
    print(f"   - Missing data cleaned: {removed_rows} records")
    print(f"   - Outliers cleaned: {removed_outliers} records")
    print(f"   - Final dataset size: {len(df)} records")
    print(
        f"   - Total cleaned: {removed_rows + removed_outliers} records "
        f"({(removed_rows + removed_outliers) / 15411 * 100:.1f}%)"
    )

    print("\nMODEL PERFORMANCE SUMMARY:")
    print(f"   - Best model: {best_model_name}")
    print(f"   - Test accuracy (R²): {best_model_info['Test R2']:.3f} ({best_model_info['Test R2'] * 100:.1f}%)")
    print(f"   - Average error (RMSE): {best_model_info['RMSE']:,.0f} TL")
    print(f"   - Overfitting status: {'LOW' if diff < 0.05 else 'MEDIUM' if diff < 0.15 else 'HIGH'}")

    print("\nMOST IMPORTANT FACTORS:")
    if feat_imp_df is not None:
        top_features = feat_imp_df.head(3)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"   {i}. {row['Feature']}: {row['Importance'] * 100:.1f}%")

    print("\nPROJECT STATUS: SUCCESSFULLY COMPLETED")
    print(f"\nPLOTS SAVED TO: {plots_dir}")


if __name__ == "__main__":
    main()
