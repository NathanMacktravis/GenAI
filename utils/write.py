#!/usr/bin/env python

from collections import Counter
import csv
import random
import numpy as np

# Définition de constantes pour la taille maximale d'un batch et la longueur maximale d'un document
_MAX_BATCH_SIZE = 128
_MAX_DOC_LENGTH = 200

# Mots spéciaux utilisés pour le padding, les mots inconnus, le début et la fin des séquences
PADDING_WORD = "<PAD>"
UNKNOWN_WORD = "<UNK>"
START_WORD = "<START>"
END_WORD = "<END>"

# Dictionnaire pour mapper les mots à leurs indices et liste pour mapper les indices à leurs mots
_word_to_idx = {}
_idx_to_word = []

# Fonction pour ajouter un mot dans les dictionnaires _word_to_idx et _idx_to_word
def _add_word(word):
    idx = len(_idx_to_word)
    _word_to_idx[word] = idx
    _idx_to_word.append(word)
    return idx

# Ajout des tokens spéciaux dans les dictionnaires
PADDING_TOKEN = _add_word(PADDING_WORD)
UNKNOWN_TOKEN = _add_word(UNKNOWN_WORD)
START_TOKEN = _add_word(START_WORD)
END_TOKEN = _add_word(END_WORD)

# Chemin vers le fichier des embeddings GloVe
embeddings_path = './data/glove/glove.6B.100d.trimmed.txt'

# Lecture des embeddings GloVe depuis le fichier
with open(embeddings_path) as f:
    # Lecture de la première ligne pour déterminer la dimension des embeddings
    line = f.readline()
    chunks = line.split(" ")
    dimensions = len(chunks) - 1
    f.seek(0)

    # Calcul de la taille du vocabulaire
    vocab_size = sum(1 for line in f)
    vocab_size += 4  # Ajout des 4 tokens spéciaux
    f.seek(0)

    # Initialisation de la matrice d'embeddings
    glove = np.ndarray((vocab_size, dimensions), dtype=np.float32)
    # Attribution d'embeddings aléatoires pour les tokens spéciaux
    glove[PADDING_TOKEN] = np.random.normal(0, 0.02, dimensions)
    glove[UNKNOWN_TOKEN] = np.random.normal(0, 0.02, dimensions)
    glove[START_TOKEN] = np.random.normal(0, 0.02, dimensions)
    glove[END_TOKEN] = np.random.normal(0, 0.02, dimensions)

    # Remplissage de la matrice d'embeddings avec les valeurs du fichier GloVe
    for line in f:
        chunks = line.split(" ")
        idx = _add_word(chunks[0])
        glove[idx] = [float(chunk) for chunk in chunks[1:]]
        if len(_idx_to_word) >= vocab_size:
            break

# Fonction pour retrouver l'index d'un mot, retourne le token inconnu si le mot n'est pas trouvé
def look_up_word(word):
    return _word_to_idx.get(word, UNKNOWN_TOKEN)

# Fonction pour retrouver le mot correspondant à un index donné
def look_up_token(token):
    return _idx_to_word[token]

# Fonction pour tokeniser une chaîne de caractères en une liste de mots
def _tokenize(string):
    return [word.lower() for word in string.split(" ")]

# Préparation d'un batch de données pour l'entraînement
def _prepare_batch(batch):
    id_to_indices = {}
    document_ids = []
    document_text = []
    document_words = []
    answer_text = []
    answer_indices = []
    question_text = []
    question_input_words = []
    question_output_words = []
    
    # Boucle pour extraire et organiser les informations du batch
    for i, entry in enumerate(batch):
        id_to_indices.setdefault(entry["document_id"], []).append(i)
        document_ids.append(entry["document_id"])
        document_text.append(entry["document_text"])
        document_words.append(entry["document_words"])
        answer_text.append(entry["answer_text"])
        answer_indices.append(entry["answer_indices"])
        question_text.append(entry["question_text"])

        question_words = entry["question_words"]
        question_input_words.append([START_WORD] + question_words)
        question_output_words.append(question_words + [END_WORD])

    # Calcul des longueurs maximales pour normaliser les séquences
    batch_size = len(batch)
    max_document_len = max((len(document) for document in document_words), default=0)
    max_answer_len = max((len(answer) for answer in answer_indices), default=0)
    max_question_len = max((len(question) for question in question_input_words), default=0)

    # Initialisation des matrices et vecteurs pour stocker les tokens, longueurs, etc.
    document_tokens = np.zeros((batch_size, max_document_len), dtype=np.int32)
    document_lengths = np.zeros(batch_size, dtype=np.int32)
    answer_labels = np.zeros((batch_size, max_document_len), dtype=np.int32)
    answer_masks = np.zeros((batch_size, max_answer_len, max_document_len), dtype=np.int32)
    answer_lengths = np.zeros(batch_size, dtype=np.int32)
    question_input_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_output_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_lengths = np.zeros(batch_size, dtype=np.int32)

    # Remplissage des matrices avec les tokens et leurs informations associées
    for i in range(batch_size):
        for j, word in enumerate(document_words[i]):
            document_tokens[i, j] = look_up_word(word)
        document_lengths[i] = len(document_words[i])

        for j, index in enumerate(answer_indices[i]):
            for shared_i in id_to_indices[batch[i]["document_id"]]:
                answer_labels[shared_i, index] = 1
            answer_masks[i, j, index] = 1
        answer_lengths[i] = len(answer_indices[i])

        for j, word in enumerate(question_input_words[i]):
            question_input_tokens[i, j] = look_up_word(word)
        for j, word in enumerate(question_output_words[i]):
            question_output_tokens[i, j] = look_up_word(word)
        question_lengths[i] = len(question_input_words[i])

    # Retourne un dictionnaire contenant toutes les informations préparées pour le batch
    return {
        "size": batch_size,
        "document_ids": document_ids,
        "document_text": document_text,
        "document_words": document_words,
        "document_tokens": document_tokens,
        "document_lengths": document_lengths,
        "answer_text": answer_text,
        "answer_indices": answer_indices,
        "answer_labels": answer_labels,
        "answer_masks": answer_masks,
        "answer_lengths": answer_lengths,
        "question_text": question_text,
        "question_input_tokens": question_input_tokens,
        "question_output_tokens": question_output_tokens,
        "question_lengths": question_lengths,
    }

# Fonction pour éliminer les doublons de documents dans un batch
def collapse_documents(batch):
    seen_ids = set()  # Ensemble pour suivre les IDs déjà vus
    keep = []  # Liste des indices des documents à garder

    # Boucle pour vérifier les documents du batch
    for i in range(batch["size"]):
        id = batch["document_ids"][i]
        if id in seen_ids:
            continue

        keep.append(i)
        seen_ids.add(id)

    # Création d'un nouveau batch ne contenant que les documents uniques
    result = {}
    for key, value in batch.items():
        if key == "size":
            result[key] = len(keep)
        elif isinstance(value, np.ndarray):
            result[key] = value[keep]
        else:
            result[key] = [value[i] for i in keep]
    return result

# Fonction pour étendre les réponses dans un batch
def expand_answers(batch, answers):
    new_batch = []

    # Boucle pour traiter chaque élément du batch
    for i in range(batch["size"]):
        split_answers = []
        last = None
        for j, tag in enumerate(answers[i]):
            if tag:
                if last != j - 1:
                    split_answers.append([])
                split_answers[-1].append(j)
                last = j

        if len(split_answers) > 0:
            answer_indices = split_answers[0]
            document_id = batch["document_ids"][i]
            document_text = batch["document_text"][i]
            document_words = batch["document_words"][i]
            answer_text = " ".join(document_words[i] for i in answer_indices)
            new_batch.append({
                "document_id": document_id,
                "document_text": document_text,
                "document_words": document_words,
                "answer_text": answer_text,
                "answer_indices": answer_indices,
                "question_text": "",
                "question_words": [],
            })
        else:
            new_batch.append({
                "document_id": batch["document_ids"][i],
                "document_text": batch["document_text"][i],
                "document_words": batch["document_words"][i],
                "answer_text": "",
                "answer_indices": [],
                "question_text": "",
                "question_words": [],
            })

    # Retourne le batch étendu, préparé pour l'entraînement
    return _prepare_batch(new_batch)

# Fonction pour lire les données à partir d'un fichier CSV
def _read_data(path):
    stories = {}

    # Ouverture et lecture du fichier CSV
    with open(path) as f:
        header_seen = False
        for row in csv.reader(f):
            if not header_seen:
                header_seen