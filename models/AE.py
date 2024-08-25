# Importation des modules nécessaires de Keras
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Activation, BatchNormalization, LeakyReLU, Dropout
from keras.models import Model
from keras import backend as K
from keras.optimizers.legacy import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model

# Importation de modules utilitaires
from utils.callbacks import CustomCallback, step_decay_schedule

# Importation de modules supplémentaires
import numpy as np
import json
import os
import pickle


class Autoencoder():
    def __init__(self, input_dim, encoder_conv_filters, encoder_conv_kernel_size, encoder_conv_strides, 
                 decoder_conv_t_filters, decoder_conv_t_kernel_size, decoder_conv_t_strides, z_dim, 
                 use_batch_norm=False, use_dropout=False):
        """
        Initialisation de l'autoencodeur avec les paramètres spécifiés.

        Arguments :
        - input_dim : Dimension d'entrée du réseau.
        - encoder_conv_filters : Liste du nombre de filtres pour chaque couche convolutionnelle de l'encodeur.
        - encoder_conv_kernel_size : Liste des tailles de kernel pour chaque couche convolutionnelle de l'encodeur.
        - encoder_conv_strides : Liste des strides pour chaque couche convolutionnelle de l'encodeur.
        - decoder_conv_t_filters : Liste du nombre de filtres pour chaque couche transposée de convolution du décodeur.
        - decoder_conv_t_kernel_size : Liste des tailles de kernel pour chaque couche transposée de convolution du décodeur.
        - decoder_conv_t_strides : Liste des strides pour chaque couche transposée de convolution du décodeur.
        - z_dim : Dimension de l'espace latent (code).
        - use_batch_norm : Booléen indiquant si la normalisation par lot doit être utilisée.
        - use_dropout : Booléen indiquant si le dropout doit être utilisé.
        """
        self.name = 'autoencoder'

        # Stockage des hyperparamètres du modèle
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        # Construction du modèle
        self._build()

    def _build(self):
        """
        Construction du modèle d'autoencodeur.
        Comprend un encodeur et un décodeur.
        """

        ### CONSTRUCTION DE L'ENCODEUR ###
        encoder_input = Input(shape=self.input_dim, name='encoder_input')  # Entrée de l'encodeur

        x = encoder_input

        # Ajout de couches convolutionnelles selon les paramètres fournis
        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters=self.encoder_conv_filters[i],
                kernel_size=self.encoder_conv_kernel_size[i],
                strides=self.encoder_conv_strides[i],
                padding='same',
                name='encoder_conv_' + str(i)
            )

            x = conv_layer(x)
            x = LeakyReLU()(x)  # Activation LeakyReLU

            # Application de la normalisation par lot si spécifié
            if self.use_batch_norm:
                x = BatchNormalization()(x)

            # Application du dropout si spécifié
            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

            print(x)  # Pour débogage

        # Capture de la forme du tenseur avant aplatissement
        shape_before_flattening = K.int_shape(x)[1:]
        print(shape_before_flattening)  # Pour débogage

        x = Flatten()(x)  # Aplatissement du tenseur
        encoder_output = Dense(self.z_dim, name='encoder_output')(x)  # Couche dense menant à l'espace latent

        self.encoder = Model(encoder_input, encoder_output)  # Création du modèle d'encodeur

        ### CONSTRUCTION DU DECODEUR ###
        decoder_input = Input(shape=(self.z_dim,), name='decoder_input')  # Entrée du décodeur

        x = Dense(np.prod(shape_before_flattening))(decoder_input)  # Couche dense suivant l'entrée du décodeur
        x = Reshape(shape_before_flattening)(x)  # Reshape pour correspondre à la forme avant aplatissement

        # Ajout de couches de convolution transposée selon les paramètres fournis
        for i in range(self.n_layers_decoder):
            conv_t_layer = Conv2DTranspose(
                filters=self.decoder_conv_t_filters[i],
                kernel_size=self.decoder_conv_t_kernel_size[i],
                strides=self.decoder_conv_t_strides[i],
                padding='same',
                name='decoder_conv_t_' + str(i)
            )

            x = conv_t_layer(x)

            # Application d'une activation LeakyReLU sauf pour la dernière couche
            if i < self.n_layers_decoder - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout(rate=0.25)(x)
            else:
                x = Activation('sigmoid')(x)  # Activation sigmoid pour la dernière couche

            print(x)  # Pour débogage
            print(shape_before_flattening)  # Pour débogage

        decoder_output = x  # Sortie du décodeur

        self.decoder = Model(decoder_input, decoder_output)  # Création du modèle de décodeur

        ### CONSTRUCTION DE L'AUTOENCODEUR COMPLET ###
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)  # Création du modèle complet (autoencodeur)

    def compile(self, learning_rate):
        """
        Compilation du modèle avec l'optimiseur et la fonction de perte.

        Arguments :
        - learning_rate : Taux d'apprentissage pour l'optimiseur.
        """
        self.learning_rate = learning_rate

        optimizer = Adam(lr=learning_rate)  # Utilisation de l'optimiseur Adam

        def r_loss(y_true, y_pred):
            # Fonction de perte basée sur l'erreur quadratique moyenne
            return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])

        self.model.compile(optimizer=optimizer, loss=r_loss)  # Compilation du modèle

    def save(self, folder):
        """
        Sauvegarde du modèle et des paramètres dans un dossier spécifié.

        Arguments :
        - folder : Chemin du dossier où sauvegarder les données.
        """
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        # Sauvegarde des paramètres dans un fichier pickle
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.encoder_conv_filters,
                self.encoder_conv_kernel_size,
                self.encoder_conv_strides,
                self.decoder_conv_t_filters,
                self.decoder_conv_t_kernel_size,
                self.decoder_conv_t_strides,
                self.z_dim,
                self.use_batch_norm,
                self.use_dropout
            ], f)

        self.plot_model(folder)  # Génération des graphiques du modèle

    def load_weights(self, filepath):
        """
        Chargement des poids du modèle à partir d'un fichier spécifié.

        Arguments :
        - filepath : Chemin du fichier de poids.
        """
        self.model.load_weights(filepath)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0, lr_decay=1):
        """
        Entraînement du modèle d'autoencodeur.

        Arguments :
        - x_train : Données d'entraînement.
        - batch_size : Taille du lot.
        - epochs : Nombre d'époques d'entraînement.
        - run_folder : Dossier où sauvegarder les résultats de l'entraînement.
        - print_every_n_batches : Fréquence d'affichage des informations d'entraînement.
        - initial_epoch : Époque initiale de l'entraînement.
        - lr_decay : Facteur de décroissance du taux d'apprentissage.
        """
        # Définition des callbacks
        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)
        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only=True, verbose=1)

        callbacks_list = [checkpoint2, custom_callback, lr_sched]

        # Entraînement du modèle
        self.model.fit(
            x_train, 
            x_train, 
            batch_size=batch_size, 
            shuffle=True, 
            epochs=epochs, 
            initial_epoch=initial_epoch, 
            callbacks=callbacks_list
        )

    def plot_model(self, run_folder):
        """
        Génération des graphiques des modèles et sauvegarde dans le dossier spécifié.

        Arguments :
        - run_folder : Dossier où sauvegarder les graphiques.
        """
        plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.encoder, to_file=os.path.join(run_folder, 'viz/encoder.png'), show_shapes=True, show_layer_names=True)
        plot_model(self.decoder, to_file=os.path.join(run_folder, 'viz/decoder.png'), show_shapes=True, show_layer_names=True)
