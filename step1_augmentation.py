import os  # Importe le module os pour effectuer des opérations liées au système d'exploitation.
import numpy as np  # Importe le module numpy pour effectuer des opérations numériques.
import vtk  # Importe le module vtk pour travailler avec des fichiers VTK.
from vedo import *  # Importe le module vedo pour visualiser et manipuler des données VTK.

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
    Trans = vtk.vtkTransform()  # Crée un objet de transformation VTK.

    ry_flag = np.random.randint(0, 2)  # Génère un drapeau aléatoire pour la rotation autour de Y (0 ou 1).
    rx_flag = np.random.randint(0, 2)  # Génère un drapeau aléatoire pour la rotation autour de X (0 ou 1).
    rz_flag = np.random.randint(0, 2)  # Génère un drapeau aléatoire pour la rotation autour de Z (0 ou 1).

    if ry_flag == 1:
        # Effectue une rotation aléatoire autour de l'axe Y si le drapeau est à 1.
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))

    if rx_flag == 1:
        # Effectue une rotation aléatoire autour de l'axe X si le drapeau est à 1.
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))

    if rz_flag == 1:
        # Effectue une rotation aléatoire autour de l'axe Z si le drapeau est à 1.
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    trans_flag = np.random.randint(0, 2)  # Génère un drapeau aléatoire pour la translation (0 ou 1).

    if trans_flag == 1:
        # Effectue une translation aléatoire si le drapeau est à 1.
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                         np.random.uniform(translate_Y[0], translate_Y[1]),
                         np.random.uniform(translate_Z[0], translate_Z[1])])

    scale_flag = np.random.randint(0, 2)  # Génère un drapeau aléatoire pour l'échelle (0 ou 1).

    if scale_flag == 1:
        # Effectue une mise à l'échelle aléatoire si le drapeau est à 1.
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])

    matrix = Trans.GetMatrix()  # Obtient la matrice de transformation résultante.

    return matrix

if __name__ == "__main__":
    num_samples = 20  # Définir le nombre d'échantillons.
    vtk_path = 'dataset_dental_VTP/vtp'  # Spécifier le chemin du répertoire contenant les fichiers VTK.
    output_save_path = './augmentation_vtk_data'

    if not os.path.exists(output_save_path):
        os.mkdir(output_save_path)  # Crée un répertoire de sortie s'il n'existe pas déjà.

    sample_list = list(range(1, num_samples + 1))  # Crée une liste d'échantillons de 1 à num_samples.
    num_augmentations = 20  # Définir le nombre d'augmentations à effectuer par échantillon.

    for i_sample in sample_list:
        for i_aug in range(num_augmentations):
            # Génère le nom de fichier d'entrée et de sortie en fonction de l'échantillon et de l'augmentation.
            file_name = 'Sample_0{0}_d.vtp'.format(i_sample)
            output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample)
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                  translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                  scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2])
            mesh = load(os.path.join(vtk_path, file_name))  # Charge le fichier VTK d'origine.
            mesh.apply_transform(vtk_matrix)  # Applique la transformation à la mesh.
            write(mesh, os.path.join(output_save_path, output_file_name))  # Écrit la mesh transformée dans un fichier.

        # Mesh inversé
        for i_aug in range(num_augmentations):
            # Génère le nom de fichier d'entrée et de sortie pour la mesh inversée.
            file_name = 'Sample_0{0}_d.vtp'.format(i_sample)
            output_file_name = 'A{0}_Sample_0{1}_d.vtp'.format(i_aug, i_sample + 1000)
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                  translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                  scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2])
            mesh = load(os.path.join(vtk_path, file_name))  # Charge le fichier VTK d'origine.
            mesh.apply_transform(vtk_matrix)  # Applique la transformation à la mesh.
            write(mesh, os.path.join(output_save_path, output_file_name))  # Écrit la mesh transformée dans un fichier.
