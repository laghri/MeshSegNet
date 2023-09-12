import os  # Module pour l'interaction avec le système de fichiers
import numpy as np  # Bibliothèque pour le calcul numérique en Python
import torch  # Bibliothèque pour l'apprentissage automatique (PyTorch)
from torch.utils.data import DataLoader  # Classe pour la gestion des données d'entraînement et de validation
from torch.optim.lr_scheduler import StepLR  # Pour la mise à jour dynamique du taux d'apprentissage
import torch.optim as optim  # Classes d'optimisation pour PyTorch
import torch.nn as nn  # Classes pour la définition de réseaux neuronaux
from Mesh_dataset import *  # Importe des classes personnalisées pour la gestion des données
from meshsegnet import *  # Importe des classes personnalisées pour la définition du modèle
from losses_and_metrics_for_mesh import *  # Importe des fonctions personnalisées pour le calcul de pertes et de métriques
import utils  # Module personnalisé contenant des utilitaires
import pandas as pd  # Bibliothèque pour la manipulation de données tabulaires (dataframes)

if __name__ == '__main__':
    # Sélectionne le GPU à utiliser (fonction personnalisée)
    torch.cuda.set_device(utils.get_avail_gpu())
    
    # Option pour l'utilisation de Visdom (outil de visualisation)
    use_visdom = False
    
    # Chemins des fichiers de listes d'entraînement et de validation
    train_list = './train_list_1.csv'  # Utilise 1-fold comme exemple
    val_list = './val_list_1.csv'  # Utilise 1-fold comme exemple

    # Répertoire où sauvegarder les modèles
    model_path = './models/'
    # Nom du modèle
    model_name = 'Mesh_Segmentation_MeshSegNet_15_classes_60_samples'  # À définir
    # Nom du fichier de checkpoint
    checkpoint_name = 'latest_checkpoint.tar'

    # Nombre de classes
    num_classes = 15
    # Nombre de canaux (caractéristiques)
    num_channels = 15
    # Nombre d'époques d'entraînement
    num_epochs = 100
    # Nombre de travailleurs pour le chargement des données
    num_workers = 0
    # Taille du lot d'entraînement
    train_batch_size = 3
    # Taille du lot de validation
    val_batch_size = 3
    # Nombre de lots à afficher pendant l'entraînement
    num_batches_to_print = 20

    if use_visdom:
        # Configuration du plotter pour Visdom
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=model_name)

    # Création du répertoire 'models' s'il n'existe pas
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Configuration des ensembles de données d'entraînement et de validation
    training_dataset = Mesh_Dataset(data_list_path=train_list,
                                    num_classes=num_classes,
                                    patch_size=6000)
    val_dataset = Mesh_Dataset(data_list_path=val_list,
                               num_classes=num_classes,
                               patch_size=6000)

    # Création des chargeurs de données pour l'entraînement et la validation
    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    # Configuration du modèle et de l'optimiseur
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    opt = optim.Adam(model.parameters(), amsgrad=True)

    # Listes pour stocker les pertes et les métriques
    losses, mdsc, msen, mppv = [], [], [], []
    val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []

    # Meilleure métrique pour la validation
    best_val_dsc = 0.0

    # Configuration de cuDNN pour accélérer l'entraînement sur GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Affichage de l'entraînement en cours
    print('Entraînement du modèle...')
    
    # Poids de classe pour la perte
    class_weights = torch.ones(15).to(device, dtype=torch.float)
    
    # Boucle sur les époques d'entraînement
    for epoch in range(num_epochs):

        # Mode d'entraînement
        model.train()
        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0
        
        # Boucle sur les mini-lots d'entraînement
        for i_batch, batched_sample in enumerate(train_loader):

            # Envoie le mini-lot sur le périphérique (GPU)
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            labels = batched_sample['labels'].to(device, dtype=torch.long)
            A_S = batched_sample['A_S'].to(device, dtype=torch.float)
            A_L = batched_sample['A_L'].to(device, dtype=torch.float)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            # Réinitialisation des gradients
            opt.zero_grad()

            # Propagation avant + rétropropagation + optimisation
            outputs = model(inputs, A_S, A_L)
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()

            # Affichage des statistiques
            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()
            
            if i_batch % num_batches_to_print == num_batches_to_print-1:  
                # Affichage toutes les N mini-lots
                print('[Époque: {0}/{1}, Mini-lot: {2}/{3}] perte: {4}, DSC: {5}, SEN: {6}, PPV: {7}'.format(
                    epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print,
                    running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print))
                if use_visdom:
                    # Affichage avec Visdom
                    plotter.plot('perte', 'entraînement', 'Perte', epoch+(i_batch+1)/len(train_loader), running_loss/num_batches_to_print)
                    plotter.plot('DSC', 'entraînement', 'DSC', epoch+(i_batch+1)/len(train_loader), running_mdsc/num_batches_to_print)
                    plotter.plot('SEN', 'entraînement', 'SEN', epoch+(i_batch+1)/len(train_loader), running_msen/num_batches_to_print)
                    plotter.plot('PPV', 'entraînement', 'PPV', epoch+(i_batch+1)/len(train_loader), running_mppv/num_batches_to_print)
                running_loss = 0.0
                running_mdsc = 0.0
                running_msen = 0.0
                running_mppv = 0.0

        # Enregistrement des pertes et des métriques
        losses.append(loss_epoch/len(train_loader))
        mdsc.append(mdsc_epoch/len(train_loader))
        msen.append(msen_epoch/len(train_loader))
        mppv.append(mppv_epoch/len(train_loader))

        # Réinitialisation des valeurs pour la prochaine époque
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0

        # Validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_mdsc = 0.0
            running_val_msen = 0.0
            running_val_mppv = 0.0
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            for i_batch, batched_val_sample in enumerate(val_loader):

                # Envoie le mini-lot de validation sur le périphérique (GPU)
                inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
                labels = batched_val_sample['labels'].to(device, dtype=torch.long)
                A_S = batched_val_sample['A_S'].to(device, dtype=torch.float)
                A_L = batched_val_sample['A_L'].to(device, dtype=torch.float)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

                outputs = model(inputs, A_S, A_L)
                loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_val_loss += loss.item()
                running_val_mdsc += dsc.item()
                running_val_msen += sen.item()
                running_val_mppv += ppv.item()
                val_loss_epoch += loss.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()

                if i_batch % num_batches_to_print == num_batches_to_print-1:  
                    # Affichage toutes les N mini-lots de validation
                    print('[Époque: {0}/{1}, Mini-lot de validation: {2}/{3}] perte de validation: {4}, DSC de validation: {5}, SEN de validation: {6}, PPV de validation: {7}'.format(
                        epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print,
                        running_val_mdsc/num_batches_to_print, running_val_msen/num_batches_to_print,
                        running_val_mppv/num_batches_to_print))
                    running_val_loss = 0.0
                    running_val_mdsc = 0.0
                    running_val_msen = 0.0
                    running_val_mppv = 0.0

            # Enregistrement des pertes et des métriques de validation
            val_losses.append(val_loss_epoch/len(val_loader))
            val_mdsc.append(val_mdsc_epoch/len(val_loader))
            val_msen.append(val_msen_epoch/len(val_loader))
            val_mppv.append(val_mppv_epoch/len(val_loader))

            # Réinitialisation des valeurs pour la prochaine époque de validation
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0

            # Affichage de l'état actuel
            print('*****\nÉpoque: {}/{}, perte: {}, DSC: {}, SEN: {}, PPV: {}\n         perte de validation: {}, DSC de validation: {}, SEN de validation: {}, PPV de validation: {}\n*****'.format(
                epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1],
                val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
            if use_visdom:
                # Affichage avec Visdom
                plotter.plot('perte', 'entraînement', 'Perte', epoch+1, losses[-1])
                plotter.plot('DSC', 'entraînement', 'DSC', epoch+1, mdsc[-1])
                plotter.plot('SEN', 'entraînement', 'SEN', epoch+1, msen[-1])
                plotter.plot('PPV', 'entraînement', 'PPV', epoch+1, mppv[-1])
                plotter.plot('perte', 'validation', 'Perte', epoch+1, val_losses[-1])
                plotter.plot('DSC', 'validation', 'DSC', epoch+1, val_mdsc[-1])
                plotter.plot('SEN', 'validation', 'SEN', epoch+1, val_msen[-1])
                plotter.plot('PPV', 'validation', 'PPV', epoch+1, val_mppv[-1])

        # Sauvegarde du checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                    model_path+checkpoint_name)

        # Sauvegarde du meilleur modèle basé sur la métrique de validation
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'msen': msen,
                        'mppv': mppv,
                        'val_losses': val_losses,
                        'val_mdsc': val_mdsc,
                        'val_msen': val_msen,
                        'val_mppv': val_mppv},
                        model_path+'{}_best.tar'.format(model_name))

        # Sauvegarde de toutes les données de pertes et de métriques
        pd_dict = {'perte': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv, 'perte de validation': val_losses, 'DSC de validation': val_mdsc, 'SEN de validation': val_msen, 'PPV de validation': val_mppv}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv('pertes_metriques_vs_epoque.csv')
