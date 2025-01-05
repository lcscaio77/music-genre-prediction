import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    log_loss,
    confusion_matrix,
)


import pandas as pd
from sklearn.metrics import (accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score, 
                             roc_curve, auc, confusion_matrix)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, classes=None, multiclass=True, print_results=True, plot_roc_curve=True, plot_conf_mat=True):
    if classes is None:
        classes = model.classes_
    
    n_classes = len(classes)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_prob)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_prob, average='macro', multi_class='ovr')

    global_results = {
        'Accuracy': accuracy,
        'Log Loss': logloss,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC Score': roc_auc,
    }
    global_results = pd.DataFrame.from_dict(global_results, orient='index', columns=['Score'])

    per_class_results = None

    if multiclass:
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        roc_auc_per_class = roc_auc_score(y_test, y_prob, average=None, multi_class='ovr')

        class_metrics = {
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1 Score': f1_per_class,
            'ROC-AUC Score': roc_auc_per_class
        }

        if isinstance(classes, dict):
            per_class_results = pd.DataFrame(class_metrics, index=list(classes.values()))
        else:
            per_class_results = pd.DataFrame(class_metrics, index=classes)

    if print_results:
        print('-'*100 + '\n')
        print("Métriques globales :\n")
        print(global_results)

        if multiclass:
            print("\nMétriques par classe :\n")
            print(per_class_results)
        
        print('\n' + '-'*100)

        if plot_roc_curve:
            fpr = {}; tpr = {}; roc_auc_curve = {}
            y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
                roc_auc_curve[i] = auc(fpr[i], tpr[i])

            plt.figure(figsize=(8, 6))

            colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'{classes[i]} (AUC = {roc_auc_curve[i]:.2f})')

            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

            plt.title('Courbes ROC Multiclasse (One-vs-Rest)')
            plt.xlabel('Taux de Faux Positifs (FPR)')
            plt.ylabel('Taux de Vrais Positifs (TPR)')
            plt.legend(loc='lower right')
            plt.show()
            
            print('-'*100)

        if plot_conf_mat:
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            cm *= 100

            _, ax = plt.subplots(figsize=(8, 6))

            annot = np.array([['{:.0f}%'.format(val) for val in row] for row in cm])
            if isinstance(classes, dict):
                sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True, xticklabels=classes.values(), yticklabels=classes.values(), ax=ax, vmin=0, vmax=100)
            else:
                sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True, xticklabels=classes, yticklabels=classes, ax=ax, vmin=0, vmax=100)

            cbar = ax.collections[0].colorbar
            cbar.set_ticks(np.linspace(0, 100, 6))
            cbar.set_ticklabels([f'{x:.0f}%' for x in cbar.get_ticks()])

            plt.xticks(rotation=90)
            plt.title('Matrice de confusion')
            plt.grid(False)
            plt.show()

            print('-'*100)

    return global_results, per_class_results


def plot_gridsearch(grid_search, params_grid):
    from operator import itemgetter

    best_params = grid_search.best_params_
    for param in best_params:
        print(f'Meilleure valeur de {param} : {best_params[param]}')

    results = grid_search.cv_results_['mean_test_score']
    params = []
    for param in params_grid:
        params.append(grid_search.cv_results_[f'param_{param}'])

    combinations = list(zip(*params))

    sorted_results = sorted(enumerate(zip(results, combinations)), key=itemgetter(1), reverse=True)

    top_results = sorted_results[:50]

    top_scores, top_combinations = zip(*[(result[1][0], result[1][1]) for result in top_results])

    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(len(top_scores)), top_scores, color='#0c6696')

    max_score_idx = np.argmax(top_scores)
    bars[max_score_idx].set_color('#961a0c')

    params_comb_str = f'({", ".join([f"{param}" for param in params_grid])})'
    plt.xticks(
        np.arange(len(top_scores)),
        [f'({", ".join(map(str, comb))})' for comb in top_combinations],
        rotation=90
    )
    plt.xlabel(f'Combinaison des hyperparamètres : {params_comb_str}')
    plt.ylim(min(top_scores) - 0.05, max(top_scores) + 0.01)
    plt.ylabel('Score moyen')
    plt.title('Scores obtenus pour les 50 meilleures combinaisons de paramètres')
    plt.tight_layout()
    plt.show()

    return best_params

