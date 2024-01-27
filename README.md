# Project Readme

## Detection de Dangerosité de Message Textuel

Ce projet vise à développer une API capable d'analyser des messages textuels pour détecter leur dangerosité, en mettant l'accent sur l'analyse sémantique, la détection de langage inapproprié, la détection de menaces, l'identification de la nature de la dangerosité, et le retour d'un pourcentage de dangerosité. Cette API peut être intégrée dans des applications de messagerie, des médias sociaux ou d'autres plateformes pour aider à filtrer les contenus potentiellement dangereux.

### Fonctionnalités

1. **Analyse sémantique:** Mise en place d'un modèle d'analyse sémantique capable de comprendre le contexte et le ton du message textuel.

2. **Détection de langage inapproprié:** Intégration d'un module pour détecter l'utilisation de langage offensant, vulgaire ou inapproprié dans le texte.

3. **Détection de menaces:** Création d'un mécanisme pour identifier les messages contenant des menaces ou des intentions nuisibles.

4. **Identification de la nature de la dangerosité:** Retour d'une liste spécifiant le type de dangerosité détectée, telle que la violence verbale, la discrimination, la cyberintimidation, etc.

5. **Retour de pourcentage de dangerosité:** Génération d'un score de dangerosité exprimé en pourcentage pour chaque message analysé.

### Spécificités 

- Utilisation de modèles de traitement du langage naturel pré-entraînés avec explication des adaptations ou améliorations spécifiques au projet.
- L'API est accessible via une interface RESTful.


### Bonus

- **Intégration de techniques de renforcement automatique** pour améliorer la détection au fil du temps.
- **Mise en place d'une interface utilisateur simple** pour tester l'API.

### Dépendances

- `fastapi`
- `uvicorn`
- `python-dotenv`
- `pytest==7.1.2`
- `httpx`
- `requests`
- `alt-profanity-check`
- `profanity-check`
- `spacy`
- `transformers`
- `datasets`
- `torch`
- `annoy`
- `accelerate`

### Membres du Projet

- Jounayd MOSBAH
- Clément SCHOBERT
- Rémy PENICHON
- Lucas AYMARD
- Enzo MOYON
- Raphaël PLASSART

### Utilisation des Datasets

- [Intuit-GenSRF/combined_toxicity_profanity_v2_train_eval](https://huggingface.co/datasets/Intuit-GenSRF/combined_toxicity_profanity_v2_train_eval)
- [mrmorj/hate-speech-and-offensive-language-dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

### Prérequis

1. Obtenez le [moteur Docker](https://docs.docker.com/engine/install/)
2. Obtenez la commande `make`

### Utilisation des Commandes

- `build` → Construit l'image Docker du projet
- `up` → Lance le projet à [cette URL](http://localhost/docs)
- `down` → Arrête le projet s'il est en cours d'exécution

- `lint` → Vérifie le formatage des fichiers du projet
- `format` → Formate les fichiers du projet

- `shell` → Accède au terminal bash du conteneur Docker
- `test` → Teste le projet

