import shap
import lime
import lime.lime_tabular
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


class ModelExplainability:
    """
    A class for explaining machine learning models using SHAP and LIME.
    Supports both classical and deep learning models.
    """

    def __init__(self, dataset_name, model_type):
        """
        Initializes the ModelExplainability class with dataset and model details.

        Args:
            dataset_name (str): The name of the dataset.
            model_type (str): The type of model (e.g., "RandomForest", "CNN").
        """
        self.dataset_name = dataset_name
        self.model_type = model_type.replace(" ", "_")
        self.data_path = f"../data/processed/{dataset_name}_X_test.npy"
        self.label_path = f"../data/processed/{dataset_name}_y_test.npy"
        self.model_path = f"../checkpoints/best_{self.model_type}_{dataset_name}.joblib"
        self.deep_model_path = (
            f"../checkpoints/best_{self.model_type}_{dataset_name}.h5"
        )
        self.load_data()
        self.load_model()

    def load_data(self):
        """
        Loads the test dataset from the specified file paths.
        """
        print(f"Loading test data for {self.dataset_name}...")
        self.X_test = np.load(self.data_path, allow_pickle=True)
        self.y_test = np.load(self.label_path, allow_pickle=True)
        print(f"Test data shape: {self.X_test.shape}")

    def load_model(self):
        """
        Loads the trained model (either classical or deep learning) from disk.
        """
        if (
            "CNN" in self.model_type
            or "RNN" in self.model_type
            or "LSTM" in self.model_type
        ):
            print(f"Loading deep learning model from {self.deep_model_path}")
            self.model = load_model(self.deep_model_path)
            self.model_type = "Deep Learning"
        else:
            print(f"Loading classical model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.model_type = "Classical"

    def explain_with_shap(self):
        """
        Generates and displays SHAP explanations for model predictions.
        Uses different SHAP explainers based on the model type.
        """
        print("Generating SHAP explanations...")
        explainer = (
            shap.Explainer(self.model.predict, self.X_test)
            if self.model_type == "Deep Learning"
            else shap.TreeExplainer(self.model)
        )
        shap_values = explainer(self.X_test)

        # Summary plot
        shap.summary_plot(shap_values, self.X_test)

    def explain_with_lime(self):
        """
        Generates and displays LIME explanations for a randomly chosen test sample.
        """
        print("Generating LIME explanations...")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_test, mode="classification"
        )
        idx = np.random.randint(0, len(self.X_test))
        exp = explainer.explain_instance(
            self.X_test[idx],
            (
                self.model.predict_proba
                if self.model_type == "Classical"
                else lambda x: self.model.predict(x).flatten()
            ),
        )
        exp.show_in_notebook()
        exp.as_pyplot_figure()
        plt.show()
