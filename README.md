# Support Ticket AI API

API backend permettant d'automatiser la première analyse des réclamations
e-commerce reçues via messagerie (audio, photo, texte), afin d'aiguiller
instantanément les équipes support.

Ce document est un **guide complet**, pensé pour que n'importe qui (même
sans avoir touché au projet) puisse comprendre l'architecture, l'installer,
le lancer, le tester et comprendre chaque choix technique.

---

## Table des matières

1. [Vue d'ensemble fonctionnelle](#1-vue-densemble-fonctionnelle)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Schéma de flux (pipeline)](#3-schéma-de-flux-pipeline)
4. [Choix techniques & justifications](#4-choix-techniques--justifications)
5. [Installation](#5-installation)
6. [Lancer l'API](#6-lancer-lapi)
7. [Utilisation de l'API](#7-utilisation-de-lapi)
8. [Détail du endpoint POST /support-ticket](#8-détail-du-endpoint-post-support-ticket)
9. [Gestion des erreurs](#9-gestion-des-erreurs)
10. [Optimisation mémoire (Singleton / lru_cache)](#10-optimisation-mémoire-singleton--lru_cache)
11. [Tests](#11-tests)
12. [Configuration (.env)](#12-configuration-env)
13. [Base de connaissances (RAG)](#13-base-de-connaissances-rag)
14. [Git Flow & gestion de projet](#14-git-flow--gestion-de-projet)
15. [Limites connues & pistes d'amélioration](#15-limites-connues--pistes-damélioration)

---

## 1. Vue d'ensemble fonctionnelle

Un client envoie une réclamation via une app de messagerie : parfois une
note vocale, parfois une photo du produit endommagé, parfois juste du
texte. Cette API expose **un seul endpoint** (`POST /support-ticket`) qui :

1. **Transcrit** l'audio en texte (si fourni) avec Whisper.
2. **Analyse** la photo du produit (si fournie) avec un modèle de vision
   (ViT) pour détecter un éventuel défaut.
3. **Recherche** dans les CGV/FAQ internes (RAG) la règle applicable, à
   partir du texte transcrit ou de la description libre.
4. **Propose un statut** de ticket (`Remboursable`, `À vérifier`,
   `Refusé`) avec une justification, dans une réponse JSON structurée.

## 2. Architecture du projet

```
support-ticket-api/
├── app/
│   ├── main.py                  # Point d'entrée FastAPI, exception handlers
│   ├── routers/
│   │   └── support.py           # Route POST /support-ticket (orchestration)
│   ├── services/
│   │   ├── asr_service.py       # Transcription audio (Whisper)
│   │   ├── vision_service.py    # Analyse image (ViT)
│   │   ├── rag_service.py       # Recherche documentaire (embeddings)
│   │   └── decision_service.py  # Logique métier -> statut du ticket
│   ├── models/
│   │   └── schemas.py           # Schémas Pydantic (requêtes/réponses)
│   ├── core/
│   │   └── config.py            # Configuration centralisée (Settings)
│   ├── utils/
│   │   ├── exceptions.py        # Exceptions métier personnalisées
│   │   └── file_utils.py        # Validation fichiers + fichiers temporaires
│   └── data/
│       └── knowledge_base.json  # Base CGV/FAQ interne (pour le RAG)
├── tests/
│   └── test_api.py              # Tests API (services IA mockés)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

**Principe de séparation des responsabilités :**
- `routers/` ne fait qu'**orchestrer** : lire les fichiers, appeler les
  services dans l'ordre, assembler la réponse. Il ne contient aucune
  logique d'inférence IA.
- `services/` contient toute la logique IA et métier, **indépendante de
  FastAPI** (aucun `services/*.py` n'importe `fastapi`). On pourrait les
  réutiliser tels quels dans un script CLI ou un worker asynchrone.
- `models/schemas.py` définit le contrat de données, validé
  automatiquement par Pydantic à l'entrée et à la sortie de l'API.
- `utils/` regroupe le code transverse (gestion fichiers, exceptions).

## 3. Schéma de flux (pipeline)

```
Client (multipart/form-data: audio? + image? + description?)
        │
        ▼
POST /support-ticket  (routers/support.py)
        │
        ├── Validation fichiers (extension, taille) ──► erreur 422 si invalide
        │
        ├── [si audio] ─► asr_service.transcribe_audio()  ─► texte transcrit
        │
        ├── [si image] ─► vision_service.analyze_image()  ─► label + confiance
        │
        ├── texte_client = transcription OU description
        │
        ├── [si texte_client] ─► rag_service.find_applicable_rule()
        │                              (recherche sémantique dans CGV/FAQ)
        │
        ├── decision_service.decide_status(vision, rag) ─► statut proposé
        │
        ▼
Réponse JSON structurée (SupportTicketResponse)
```

## 4. Choix techniques & justifications

| Besoin | Choix | Justification |
|---|---|---|
| Framework API | **FastAPI** | Validation native via Pydantic, génération automatique de Swagger (`/docs`), support natif de `async`/`multipart`. |
| ASR | **openai/whisper-small** (transformers) | Bon compromis vitesse/qualité pour du CPU, multilingue, disponible directement via `pipeline()`. |
| Vision | **google/vit-base-patch16-224** | Modèle ViT léger et standard ; en production, on le remplacerait par une version **fine-tunée** sur des photos de produits endommagés (voir §15). |
| RAG - embeddings | **sentence-transformers/all-MiniLM-L6-v2** | Modèle d'embeddings compact (~80 Mo), rapide sur CPU, très utilisé pour de la recherche sémantique légère sans base vectorielle externe. |
| RAG - stockage | **JSON + matrice numpy en mémoire** | Base de connaissances de taille modeste (quelques dizaines de règles CGV/FAQ) : une vraie base vectorielle (FAISS, Chroma...) serait sur-dimensionnée. Le code est néanmoins isolé dans `rag_service.py`, donc remplaçable facilement. |
| Chargement des modèles | **Singleton via `@lru_cache()`** | Voir section dédiée [§10](#10-optimisation-mémoire-singleton--lru_cache). |
| Fichiers temporaires | **`tempfile` + context manager** | Garantit la suppression du fichier même en cas d'exception (`try/finally`). |
| Gestion d'erreurs | **Exceptions métier + `@app.exception_handler`** | Les services ne connaissent pas les codes HTTP ; `main.py` centralise la traduction exception → réponse HTTP. |

## 5. Installation

### Prérequis
- Python 3.11 ou supérieur
- ~5 Go d'espace disque libre (poids des modèles Whisper + ViT +
  sentence-transformers téléchargés au premier lancement)
- Connexion internet au premier démarrage (téléchargement des modèles
  depuis Hugging Face Hub, puis mis en cache localement)



## 5. Lancer l'API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger UI (documentation interactive + test des endpoints) :
  **http://localhost:8000/docs**
- Redoc : **http://localhost:8000/redoc**
- Health check : **http://localhost:8000/health**

> ⚠️ Le premier appel à `/support-ticket` déclenchera le téléchargement
> des modèles (Whisper, ViT, sentence-transformers) — cela peut prendre
> plusieurs minutes selon la connexion. Les appels suivants sont rapides
> car les modèles restent chargés en mémoire (voir §10) et mis en cache
> sur disque par `transformers`/`sentence-transformers`.

## 6. Utilisation de l'API

### Exemple avec `curl` (texte seul)

```bash
curl -X POST "http://localhost:8000/support-ticket" \
  -F "description=Mon colis est arrivé avec la boîte cassée et le produit rayé"
```

### Exemple avec `curl` (audio + image)

```bash
curl -X POST "http://localhost:8000/support-ticket" \
  -F "audio=@note_vocale_client.wav" \
  -F "image=@photo_produit.jpg"
```

### Exemple avec `httpx` (Python)

```python
import httpx

files = {
    "audio": open("note_vocale_client.wav", "rb"),
    "image": open("photo_produit.jpg", "rb"),
}
response = httpx.post("http://localhost:8000/support-ticket", files=files, timeout=120)
print(response.json())
```

0
## 9. Gestion des erreurs

Toutes les erreurs renvoient un JSON structuré `{ "detail": ..., "error_code": ... }` :

| Code HTTP | error_code | Cas |
|---|---|---|
| 400 | `no_input_provided` | Aucun audio/image/texte fourni |
| 422 | `invalid_file` | Extension non supportée ou fichier trop volumineux |
| 502 | `model_inference_error` | Échec d'un modèle IA (audio corrompu, image illisible...) |
| 500 | `unhandled_exception` | Erreur serveur inattendue (loggée côté serveur, jamais de stack trace exposée au client) |

## 10. Optimisation mémoire (Singleton / lru_cache)

Les 3 modèles (Whisper, ViT, sentence-transformers) sont **coûteux à
charger** (plusieurs centaines de Mo à charger en RAM). Si on les
rechargeait à chaque requête HTTP, l'API saturerait rapidement la
mémoire et deviendrait très lente.

Chaque service expose une fonction de chargement décorée par
`@lru_cache()`, ex. dans `asr_service.py` :

```python
@lru_cache()
def _load_asr_pipeline():
    ...
    return pipeline(task="automatic-speech-recognition", model=..., device=...)
```

`lru_cache()` (sans argument, donc taille illimitée mais un seul jeu
d'arguments possible ici : aucun argument → un seul résultat mis en
cache) garantit que la fonction ne s'exécute réellement **qu'une seule
fois** par process Python, quel que soit le nombre de requêtes reçues
ensuite : les appels suivants retournent instantanément l'objet déjà en
mémoire. C'est l'équivalent d'un pattern **Singleton**, sans avoir à
écrire de classe dédiée.

De la même façon, la base de connaissances RAG est **encodée en
vecteurs une seule fois** au premier appel (`_load_knowledge_base()`),
puis réutilisée pour toutes les recherches suivantes.





## 13. Base de connaissances (RAG)

Le fichier `app/data/faq.txt` contient des règles CGV/FAQ
factices (retour, produit endommagé, retard de livraison...). Chaque
entrée a la forme :


## 14. Git Flow & gestion de projet

- Branches : `main` (production), `develop` (intégration),
  `feature/xxx` (une branche par fonctionnalité), fusionnées vers
  `develop` via Pull Request après revue.
- Aucun commit direct sur `main`.
- Tableau Kanban (Backlog / In Progress / Review / Done) : **lien à
  ajouter ici par l'équipe** → `[Lien vers le tableau Kanban](À_COMPLETER)`
- Répartition des tâches (si travail en binôme) : voir le document
  séparé `TASKS.md` (non inclus dans ce README, conformément à la
  consigne).

