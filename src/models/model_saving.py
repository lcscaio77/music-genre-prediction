import os
import json
from joblib import dump, load
from datetime import datetime


def save_model(model, model_name, directory='../models/', metadata=None):
    model_path = os.path.join(directory, f'{model_name}.joblib')
    dump(model, model_path)

    if metadata is None:
        metadata = {}
    metadata['model_name'] = model_name
    metadata['date_saved'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    metadata_path = os.path.join(directory, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f'Modèle et ses métadonnées sauvegardées dans : {os.path.abspath(directory)}')


def load_model(filename, directory='../models/'):
    model_path = os.path.join(directory, filename + '.joblib')

    try:
        model = load(model_path)
        print('Modèle chargé avec succès !')
        return model
    except Exception as e:
        raise ValueError(f'Erreur lors du chargement du modèle : {e}')

