from joblib import dump, load

import os
import json
from datetime import datetime

def save_model(model, model_name, directory="../artifacts/models/", metadata=None):
    model_path = os.path.join(directory, f"{model_name}.joblib")
    dump(model, model_path)
    print(f"Modèle sauvegardé : {model_path}")

    if metadata is None:
        metadata = {}
    metadata["model_name"] = model_name
    metadata["date_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metadata_path = os.path.join(directory, f"{model_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Métadonnées sauvegardées : {metadata_path}")

def load_model(filename, directory='../artifacts/models/'):
    model_path = os.path.join(directory, filename)
    return load(model_path)
