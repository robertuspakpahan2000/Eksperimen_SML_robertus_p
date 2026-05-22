import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with median for numeric columns."""
    logger.info("Handling missing values...")
    missing_before = df.isnull().sum().sum()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"  Filled '{col}' with median: {median_val}")
    
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} → {missing_after}")
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    logger.info("Handling duplicates...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"Duplicates removed: {before - after} rows")
    return df


def handle_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Cap outliers using IQR method."""
    logger.info("Handling outliers with IQR method...")
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        logger.info(f"  '{col}': {before} outliers capped [{lower:.2f}, {upper:.2f}]")
    return df


def encode_categorical(df: pd.DataFrame, cat_columns: list) -> pd.DataFrame:
    """Encode categorical columns using LabelEncoder."""
    logger.info("Encoding categorical columns...")
    for col in cat_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"  Encoded '{col}'")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features."""
    logger.info("Performing feature engineering...")
    # Age group feature
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 100],
                              labels=[0, 1, 2, 3]).astype(int)
    # Chol to age ratio
    df['chol_age_ratio'] = df['chol'] / (df['age'] + 1)
    logger.info("  Added: age_group, chol_age_ratio")
    return df


def scale_features(df: pd.DataFrame, target_col: str,
                   scale_cols: list) -> pd.DataFrame:
    """Standardize numerical features."""
    logger.info("Scaling features with StandardScaler...")
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    logger.info(f"  Scaled {len(scale_cols)} columns")
    return df


def split_and_save(df: pd.DataFrame, target_col: str,
                   output_dir: str, test_size: float = 0.2,
                   random_state: int = 42):
    """Split dataset and save train/test CSV."""
    logger.info(f"Splitting data: test_size={test_size}")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(output_dir, "heart_train.csv")
    test_path = os.path.join(output_dir, "heart_test.csv")
    full_path = os.path.join(output_dir, "heart_preprocessing.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)

    logger.info(f"Train saved: {train_path} ({len(train_df)} rows)")
    logger.info(f"Test saved:  {test_path}  ({len(test_df)} rows)")
    logger.info(f"Full saved:  {full_path}  ({len(df)} rows)")
    return train_df, test_df


def preprocess(raw_filepath: str, output_dir: str) -> pd.DataFrame:
    """Main preprocessing pipeline."""
    logger.info("=" * 50)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 50)

    # Config
    TARGET_COL = "target"
    NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    CATEGORICAL_COLS = []  # Heart dataset is already numeric
    SCALE_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                  'chol_age_ratio']

    # Pipeline
    df = load_data(raw_filepath)
    df = handle_duplicates(df)
    df = handle_missing_values(df)
    df = handle_outliers(df, NUMERIC_COLS)
    df = encode_categorical(df, CATEGORICAL_COLS)
    df = feature_engineering(df)
    df = scale_features(df, TARGET_COL, SCALE_COLS)
    train_df, test_df = split_and_save(df, TARGET_COL, output_dir)

    logger.info("=" * 50)
    logger.info("Preprocessing pipeline completed successfully!")
    logger.info("=" * 50)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automate Heart Disease Preprocessing")
    parser.add_argument("--input", type=str, default="heart_raw.csv",
                        help="Path to raw CSV file")
    parser.add_argument("--output", type=str, default="preprocessing/heart_preprocessing",
                        help="Output directory for preprocessed data")
    args = parser.parse_args()

    preprocess(args.input, args.output)
