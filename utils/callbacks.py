"""Code permettant d'automatiser la génération et la sauvegarde d'images à intervalles réguliers pendant
 l'entraînement d'un VAE, tout en ajustant dynamiquement le taux d'apprentissage selon une stratégie 
 de décroissance par paliers(Technique d'optimisation des réseaux de neurones)."""


from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os

#### CALLBACKS
class CustomCallback(Callback):  # Définition d'une classe personnalisée dérivée de Callback
    
    def __init__(self, run_folder, print_every_n_batches, initial_epoch, vae):
        # Initialisation de la classe avec les paramètres nécessaires
        self.epoch = initial_epoch  # L'epoch de départ
        self.run_folder = run_folder  # Dossier où sauvegarder les images générées
        self.print_every_n_batches = print_every_n_batches  # Fréquence de sauvegarde des images
        self.vae = vae  # Modèle VAE (Variational Autoencoder)

    def on_batch_end(self, batch, logs={}):  
        # Fonction appelée à la fin de chaque batch d'entraînement
        if batch % self.print_every_n_batches == 0:  # Si c'est le moment de sauvegarder une image
            z_new = np.random.normal(size=(1, self.vae.z_dim))  # Génère un nouveau vecteur latent aléatoire
            reconst = self.vae.decoder.predict(np.array(z_new))[0].squeeze()  # Génère une image à partir de ce vecteur

            # Détermine le chemin pour sauvegarder l'image
            filepath = os.path.join(self.run_folder, 'images', 'img_' + str(self.epoch).zfill(3) + '_' + str(batch) + '.jpg')
            
            # Sauvegarde l'image en tant que fichier .jpg
            if len(reconst.shape) == 2:  # Si l'image est en niveaux de gris
                plt.imsave(filepath, reconst, cmap='gray_r')
            else:  # Si l'image est en couleur
                plt.imsave(filepath, reconst)

    def on_epoch_begin(self, epoch, logs={}):
        # Fonction appelée au début de chaque epoch
        self.epoch += 1  # Incrémente le compteur d'epochs


def step_decay_schedule(initial_lr, decay_factor=0.5, step_size=1):
    '''
    Fonction qui crée un LearningRateScheduler avec une stratégie de décroissance du taux d'apprentissage par paliers.
    '''
    def schedule(epoch):
        # Calcule le nouveau taux d'apprentissage en fonction de l'epoch
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))
        return new_lr

    return LearningRateScheduler(schedule)  # Retourne un LearningRateScheduler avec la stratégie de décroissance définie
