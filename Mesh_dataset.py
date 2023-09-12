from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix

class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=15, patch_size=7000):
        """
        Args:
            data_list_path (string): Chemin vers le fichier CSV contenant la liste des fichiers de maillage.
            num_classes (int, optional): Nombre de classes pour la segmentation. Par défaut, 15 classes.
            patch_size (int, optional): Taille de la mosaïque de données. Par défaut, 7000.
        """
        # Charge la liste des fichiers de maillage depuis le fichier CSV
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        """
        Renvoie la longueur du jeu de données, c'est-à-dire le nombre d'exemples de maillage.
        """
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        """
        Renvoie un échantillon de données à partir de l'ensemble de données.

        Args:
            idx (int): L'indice de l'échantillon à récupérer.

        Returns:
            sample (dict): Un dictionnaire contenant les données d'entrée, les étiquettes et les matrices d'adjacence.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Récupère le nom du fichier de maillage à partir de la liste des données
        i_mesh = self.data_list.iloc[idx][0]

        # Charge le fichier de maillage au format VTK
        mesh = load(i_mesh)
        
        # Extrait les étiquettes (labels) des cellules du maillage
        labels = mesh.celldata['Label'].astype('int32').reshape(-1, 1)

        # ... (Suite du prétraitement des données)
        
        # Construit le dictionnaire d'échantillon contenant les données d'entrée, les étiquettes et les matrices d'adjacence
        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                  'A_S': torch.from_numpy(S1), 'A_L': torch.from_numpy(S2)}

        return sample

if __name__ == '__main__':
    # Crée une instance de la classe Mesh_Dataset en utilisant un fichier CSV spécifique
    dataset = Mesh_Dataset('./train_list_1.csv')
    
    # Récupère et affiche un échantillon du jeu de données
    print(dataset.__getitem__(0))
