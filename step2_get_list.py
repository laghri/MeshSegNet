import numpy as np  # Importe le module numpy pour effectuer des opérations numériques.
import os  # Importe le module os pour effectuer des opérations liées au système de fichiers.
from sklearn.model_selection import KFold  # Importe la fonction KFold de sklearn pour diviser les données en ensembles.
from sklearn.model_selection import train_test_split  # Importe la fonction train_test_split de sklearn pour diviser les données en ensembles d'entraînement et de validation.
import pandas as pd  # Importe le module pandas pour travailler avec des données tabulaires.

if __name__ == '__main__':
    # Le code suivant est exécuté uniquement lorsque le script est exécuté en tant que programme principal.

    data_path = './augmentation_vtk_data/'  # Spécifie le chemin vers les données augmentées en VTK.
    output_path = './'  # Spécifie le répertoire de sortie.
    num_augmentations = 5  # Définit le nombre d'augmentations par échantillon.
    train_size = 0.8  # Définit la proportion d'échantillons à utiliser pour l'ensemble d'entraînement.
    with_flip = True  # Indique si les échantillons inversés (flipped) doivent être inclus.

    num_samples = 20  # Définit le nombre d'échantillons.
    sample_list = list(range(1, num_samples + 1))  # Crée une liste d'échantillons de 1 à num_samples.
    sample_name = 'A{0}_Sample_0{1}_d.vtp'  # Modèle de nom de fichier pour les échantillons.

    # Obtient la liste des échantillons valides existants dans le répertoire de données.
    valid_sample_list = []
    for i_sample in sample_list:
        for i_aug in range(num_augmentations):
            if os.path.exists(os.path.join(data_path, sample_name.format(i_aug, i_sample))):
                valid_sample_list.append(i_sample)

    # Supprime les doublons de la liste des échantillons valides.
    sample_list = list(dict.fromkeys(valid_sample_list))
    sample_list = np.asarray(sample_list)  # Convertit la liste en un tableau numpy.

    i_cv = 0  # Initialisation d'un compteur pour la validation croisée (cross-validation).
    kf = KFold(n_splits=6, shuffle=False)  # Crée un objet KFold pour effectuer une validation croisée en 6 parties sans mélanger les données.

    # Boucle pour la validation croisée.
    for train_idx, test_idx in kf.split(sample_list):
        i_cv += 1  # Incrémente le compteur de validation croisée.
        print('Round:', i_cv)  # Affiche le numéro de la validation croisée en cours.

        train_list, test_list = sample_list[train_idx], sample_list[test_idx]  # Divise les échantillons en ensembles d'entraînement et de test.
        train_list, val_list = train_test_split(train_list, train_size=0.8, shuffle=True)  # Divise l'ensemble d'entraînement en ensembles d'entraînement et de validation.

        # Affiche les listes d'échantillons pour chaque ensemble.
        print('Training list:\n', train_list, '\nValidation list:\n', val_list, '\nTest list:\n', test_list)

        # Crée la liste des fichiers d'entraînement.
        train_name_list = []
        for i_sample in train_list:
            for i_aug in range(num_augmentations):
                subject_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample)
                train_name_list.append(os.path.join(data_path, subject_name))
                if with_flip:
                    subject2_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug + 1000, i_sample)
                    train_name_list.append(os.path.join(data_path, subject2_name))

        # Écrit la liste des fichiers d'entraînement dans un fichier CSV.
        with open(os.path.join(output_path, 'train_list_{0}.csv'.format(i_cv)), 'w') as file:
            for f in train_name_list:
                file.write(f + '\n')

        # Crée la liste des fichiers de validation.
        val_name_list = []
        for i_sample in val_list:
            for i_aug in range(num_augmentations):
                subject_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug, i_sample)
                val_name_list.append(os.path.join(data_path, subject_name))
                if with_flip:
                    subject2_name = 'A{}_Sample_0{}_d.vtp'.format(i_aug + 1000, i_sample)
                    val_name_list.append(os.path.join(data_path, subject2_name))

        # Écrit la liste des fichiers de validation dans un fichier CSV.
        with open(os.path.join(output_path, 'val_list_{0}.csv'.format(i_cv)), 'w') as file:
            for f in val_name_list:
                file.write(f + '\n')

        # Crée un DataFrame pandas pour la liste des fichiers de test.
        test_df = pd.DataFrame(data=test_list, columns=['Test ID'])
        test_df.to_csv('test_list_{}.csv'.format(i_cv), index=False)

        # Affiche des informations sur les ensembles d'entraînement, de validation et de test.
        print('--------------------------------------------')
        print('with flipped samples:', with_flip)
        print('# of train:', len(train_name_list))
        print('# of validation:', len(val_name_list))
        print('--------------------------------------------')
