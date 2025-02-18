import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from IPython.display import display


class FraudFeatureEngineer:
    def __init__(self, fraud_data_path, creditcard_data_path, seed=42):
        self.fraud_data_path = fraud_data_path
        self.creditcard_data_path = creditcard_data_path
        self.seed = seed
        self.scaler = MinMaxScaler()
        self.label_encoders = {}

    def load_data(self, file_path):
        print(f"Loading data from {file_path}...")
        return pd.read_csv(file_path)

    def process_fraud_data(self):
        print("Processing fraud data...")
        df = self.load_data(self.fraud_data_path)

        print("Converting timestamps and extracting time-based features...")
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        df["hour_of_day"] = df["purchase_time"].dt.hour
        df["day_of_week"] = df["purchase_time"].dt.weekday
        df["time_since_signup"] = (
            df["purchase_time"] - df["signup_time"]
        ).dt.total_seconds()

        print("Calculating transaction frequency and velocity...")
        df["transaction_count"] = df.groupby("user_id")["user_id"].transform("count")
        df["purchase_time_unix"] = df["purchase_time"].astype(np.int64) // 10**9
        df["velocity"] = df.groupby("device_id")["purchase_time_unix"].diff().fillna(0)

        print("Dropping unnecessary columns...")
        df.drop(
            columns=[
                "user_id",
                "signup_time",
                "purchase_time",
                "device_id",
                "ip_address",
                "purchase_time_unix",
            ],
            inplace=True,
        )

        print("Encoding categorical features...")
        categorical_columns = ["source", "browser", "sex", "country"]
        for col in categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])

        print("Normalizing numerical features...")
        numerical_columns = [
            "purchase_value",
            "age",
            "hour_of_day",
            "day_of_week",
            "time_since_signup",
            "transaction_count",
            "velocity",
        ]
        df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])

        print("First 3 rows of transformed fraud data:")
        display(df.head(3))

        print("Splitting and saving data...")
        X = df.drop(columns=["class"])
        y = df["class"]
        return self.split_and_save(X, y, "fraud_data")

    def process_creditcard_data(self):
        print("Processing credit card data...")
        df = self.load_data(self.creditcard_data_path)

        print("Normalizing features V1 to V28 and Amount...")
        feature_columns = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])

        print("First 3 rows of transformed credit card data:")
        display(df.head(3))

        print("Splitting and saving data...")
        X = df.drop(columns=["Class"])
        y = df["Class"]
        return self.split_and_save(X, y, "creditcard_data")

    def split_and_save(self, X, y, filename):
        print(f"Splitting data for {filename}...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=self.seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.seed, stratify=y_temp
        )

        print(f"Saving split datasets for {filename}...")
        np.save(f"../data/processed/{filename}_X_train.npy", X_train)
        np.save(f"../data/processed/{filename}_X_val.npy", X_val)
        np.save(f"../data/processed/{filename}_X_test.npy", X_test)
        np.save(f"../data/processed/{filename}_y_train.npy", y_train)
        np.save(f"../data/processed/{filename}_y_val.npy", y_val)
        np.save(f"../data/processed/{filename}_y_test.npy", y_test)

        print("Data processing complete.")
