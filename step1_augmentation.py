import os
import numpy as np
import vtk
from vedo import *

def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                               translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    '''
    Cette fonction génère une matrice de transformation VTK 4x4 en effectuant des opérations de rotation,
    de translation et d'échelle aléatoires.
    :param rotate_X: Intervalle de rotation autour de l'axe X.
    :param rotate_Y: Intervalle de rotation autour de l'axe Y.
    :param rotate_Z: Intervalle de rotation autour de l'axe Z.
    :param translate_X: Intervalle de translation le long de l'axe X.
    :param translate_Y: Intervalle de translation le long de l'axe Y.
    :param translate_Z: Intervalle de translation le long de l'axe Z.
    :param scale_X: Intervalle d'échelle le long de l'axe X.
    :param scale_Y: Intervalle d'échelle le long de l'axe Y.
    :param scale_Z: Intervalle d'échelle le long de l'axe Z.
    :return: Matrice de transformation VTK 4x4.
    '''
    Trans = vtk.vtkTransform()

    # Générer des drapeaux aléatoires pour déterminer quelles transformations effectuer
    ry_flag = np.random.randint(0, 2)  # Si 0, pas de rotation autour de Y
    rx_flag = np.random.randint(0, 2)  # Si 0, pas de rotation autour de X
    rz_flag = np.random.randint(0, 2)  # Si 0, pas de rotation autour de Z
    if ry_flag == 1:
        # Rotation aléatoire autour de l'axe Y
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # Rotation aléatoire autour de l'axe X
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # Rotation aléatoire autour de l'axe Z
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    # Générer un drapeau aléatoire pour déterminer si une translation doit être effectuée
    trans_flag = np.random.randint(0, 2)
    if trans_flag == 1:
        # Translation aléatoire le long de X, Y et Z
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                         np.random.uniform(translate_Y[0], translate_Y[1]),
                         np.random.uniform(translate_Z[0], translate_Z[1])])

    # Générer un drapeau aléatoire pour déterminer si une mise à l'échelle doit être effectuée
    scale_flag = np.random.randint(0, 2)
    if scale_flag == 1:
        # Mise à l'échelle aléatoire le long de X, Y et Z
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])

    matrix = Trans.GetMatrix()

    return matrix

if __name__ == "__main__":
    num_samples = 20  # Définir le nombre d'échantillons
    vtk_path = 'dataset_dental_VTP/vtp'  # Spécifier le chemin du répertoire contenant les fichiers VTK
    output_save_path = './augmentation_vtk_data'
    if not os.path.exists(output_save_path):
        os.mkdir(output_save_path)

    sample_list = list(range(1, num_samples + 1))
    num_augmentations = 20

    for i_sample in sample_list:
        for i_aug in range(num_augmentations):
            # Charger le fichier VTK d'origine
            file_name = 'Sample_0{0}_d.vtp'.format(i_sample)
            output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample)
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                  translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                  scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2])
            mesh = load(os.path.join(vtk_path, file_name))
            mesh.apply_transform(vtk_matrix)
            write(mesh, os.path.join(output_save_path, output_file_name))

        # Mesh inversé
        for i_aug in range(num_augmentations):
            file_name = 'Sample_0{0}_d.vtp'.format(i_sample)
            output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample + 1000)
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                  translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                  scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2])
            mesh = load(os.path.join(vtk_path, file_name))
            mesh.apply_transform(vtk_matrix)
            write(mesh, os.path.join(output_save_path, output_file_name))
