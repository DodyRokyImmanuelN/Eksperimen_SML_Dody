import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# =====================================================
# Load dataset
# =====================================================
def load_data(path):
    print("📂 Loading dataset...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# =====================================================
# Handle duplicated rows
# =====================================================
def handle_duplicates(df):
    print("\n🔍 Handling duplicated rows...")

    dup_before = df.duplicated().sum()
    print(f"   - Duplicated rows sebelum drop: {dup_before}")

    if dup_before > 0:
        df = df.drop_duplicates()
        dup_after = df.duplicated().sum()
        print(f"   - Duplicated rows setelah drop: {dup_after}")
        print(f"   - Shape baru: {df.shape}")
    else:
        print("   - Tidak ada duplicated rows.")

    print("✅ Duplicate handling complete.")
    return df


# =====================================================
# Handle missing values
# =====================================================
def handle_missing(df):
    """Fill missing values dengan median (jika ada)"""
    print("\n🔧 Handling missing values...")

    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        print(f"   - Total missing values: {total_missing}")

        for col in df.columns:
            miss = df[col].isnull().sum()
            if miss > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"   - {col}: {miss} NaN diisi median = {median_val:.4f}")

        print("✅ Missing values handled.")
    else:
        print("   - No missing values found.")
        print("✅ Missing value step skipped.")

    return df


# =====================================================
# Handle outlier (IQR Winsorizing)
# =====================================================
def handle_outliers(df, target_col="target"):
    """Handle outliers menggunakan IQR Winsorizing"""
    print("\n🎯 Handling outliers (IQR Winsorizing)...")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Jangan sentuh kolom target
    if target_col in num_cols:
        num_cols.remove(target_col)

    outliers_found = False

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Hitung outlier
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if outliers > 0:
            outliers_found = True
            print(f"   - {col}: {outliers} outliers winsorized")
            df[col] = np.where(
                df[col] < lower_bound, lower_bound,
                np.where(df[col] > upper_bound, upper_bound, df[col])
            )

    if outliers_found:
        print("✅ Outliers handled.")
    else:
        print("   - No outliers found (berdasarkan aturan IQR).")
        print("✅ Outlier step selesai tanpa perubahan.")

    return df


# =====================================================
# Scaling numeric features (target tidak di-scale)
# =====================================================
def scale_numeric(df,
                  target_col="target",
                  save_scaler=True,
                  scaler_path="preprocessing/heart_scaler.pkl"):
    """
    Scale numeric features menggunakan StandardScaler.
    Kolom target tidak di-scale.
    Scaler disimpan untuk dipakai saat inference / modelling.
    """
    print("\n🔄 Scaling numeric features (StandardScaler)...")

    # Ambil semua kolom numerik
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Keluarkan target dari scaling
    if target_col in num_cols:
        num_cols.remove(target_col)
        print(f"   - Target column '{target_col}' excluded from scaling")

    # Inisialisasi scaler
    scaler = StandardScaler()

    # Fit dan transform
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"   - Scaled {len(num_cols)} numeric features")

    # Simpan scaler
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved to: {scaler_path}")

    print(f"✅ Scaling complete. Final shape: {df.shape}")
    return df, scaler, num_cols


# =====================================================
# Save preprocessing statistics
# =====================================================
def save_preprocessing_stats(scaler, feature_names,
                             output_path="preprocessing/heart_preprocessing_stats.txt"):
    """Simpan statistik scaler (mean & std) untuk dokumentasi"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("HEART DATASET PREPROCESSING STATISTICS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Scaler Type: StandardScaler\n\n")
        f.write("Feature Statistics (Mean and Std):\n")
        f.write("-" * 60 + "\n")

        for i, feature in enumerate(feature_names):
            mean = scaler.mean_[i]
            std = scaler.scale_[i]
            f.write(f"{feature:25s}: mean={mean:10.4f}, std={std:10.4f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"✅ Preprocessing stats saved to: {output_path}")


# =====================================================
# Main preprocessing pipeline
# =====================================================
def preprocess_heart(input_path,
                     output_path,
                     scaler_path="preprocessing/heart_scaler.pkl"):
    """
    Complete preprocessing pipeline untuk heart disease dataset.

    Urutan pipeline:
    1. Load data
    2. Handle duplicated rows
    3. Handle missing values (median)
    4. Handle outliers (IQR Winsorizing)
    5. Scale numeric features (simpan scaler)
    6. Simpan data hasil preprocessing
    """

    print("\n" + "=" * 70)
    print("🚀 HEART DATASET PREPROCESSING PIPELINE")
    print("=" * 70 + "\n")

    try:
        # Step 1: Load data
        df = load_data(input_path)

        # Step 2: Handle duplicates
        df = handle_duplicates(df)

        # Step 3: Handle missing values
        df = handle_missing(df)

        # Step 4: Handle outliers
        df = handle_outliers(df, target_col="target")

        # Step 5: Scale numeric features
        df, scaler, scaled_features = scale_numeric(
            df,
            target_col="target",
            save_scaler=True,
            scaler_path=scaler_path
        )

        # Step 6: Save preprocessing statistics
        save_preprocessing_stats(
            scaler,
            feature_names=scaled_features,
            output_path="preprocessing/heart_preprocessing_stats.txt"
        )

        # Step 7: Save preprocessed data
        out_dir = os.path.dirname(output_path)
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"\n✅ Preprocessed data saved to: {output_path}")
        print(f"✅ Scaler object saved to: {scaler_path}")
        print("\n" + "=" * 70)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 70 + "\n")

        return df, scaler

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("=" * 70 + "\n")
        raise

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70 + "\n")
        raise


# =====================================================
# Jalankan otomatis
# =====================================================
if __name__ == "__main__":
    # Sesuaikan dengan lokasi file di project-mu
    INPUT_PATH = "heart_raw.csv"                                  
    OUTPUT_PATH = "preprocessing/heart_preprocessing.csv"
    SCALER_PATH = "preprocessing/heart_scaler.pkl"

    try:
        df_processed, scaler_obj = preprocess_heart(
            input_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            scaler_path=SCALER_PATH
        )

        print("📊 Preprocessed Data Summary:")
        print(f"   - Shape : {df_processed.shape}")
        print(f"   - Columns: {list(df_processed.columns)}")
        print("\n   First 3 rows (scaled):")
        print(df_processed.head(3))

    except Exception as e:
        print(f"\n❌ Preprocessing failed: {e}")
        exit(1)
