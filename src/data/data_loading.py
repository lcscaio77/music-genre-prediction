import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print('Données chargées avec succès.')
        return data
    except Exception as e:
        raise ValueError(f'Erreur lors du chargement des données : {e}')
