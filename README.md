# Benchmark Mistral STT sur des vidéos YouTube en Darija

Proof of Concept indépendant, sans frontend, Docker ni base propre. Il lit uniquement les liens déjà collectés dans le projet `darija_youtube_extraction`, prépare 20 extraits audio de 120 secondes pris au milieu des vidéos, puis les transcrit avec l'API officielle Mistral.

Le dataset source n'est jamais modifié. Les connexions SQLite sont ouvertes avec `mode=ro` (lecture seule), et aucune nouvelle vidéo n'est recherchée sur YouTube.

## Dataset réellement identifié

Source principale inspectée :

`C:\Users\BadrJOULALI\Downloads\darija_youtube_extraction\darija_youtube_clean.sqlite`

- table : `videos`
- nombre de lignes observé pendant la validation : plus de 655 000 (la base source continue d'être alimentée)
- colonne URL : `video_url`
- filtre appliqué : `is_darija = 1`
- colonnes utiles déjà présentes : `video_id`, `title`, `duration`, `availability`, `darija_score`, `is_darija`

Le loader accepte aussi CSV, JSON, JSONL/NDJSON et TXT. Il détecte la colonne URL sans supposer qu'elle s'appelle `url`.

## Architecture

- `notebooks/00_inspect_dataset.ipynb` : inspecte la source en lecture seule et affiche quelques URLs existantes.
- `notebooks/01_prepare_audio_samples.ipynb` : sélectionne 20 vidéos valides et extrait leur segment central de 120 secondes.
- `notebooks/02_run_mistral_benchmark.ipynb` : lance les transcriptions et reprend après interruption.
- `benchmark.py` : orchestration réutilisée par les notebooks.
- `dataset_loader.py` : lecture seule SQLite/CSV/JSON/JSONL/TXT et détection de la colonne URL.
- `youtube_downloader.py` : métadonnées et téléchargement audio via `yt-dlp`.
- `audio_extractor.py` : calcul du segment central, extraction et contrôle de durée via FFmpeg/ffprobe.
- `transcriber.py` : appel officiel `audio.transcriptions.complete` et messages d'erreur sûrs.
- `config.py` : chemins et constantes (`20` extraits de `120` secondes).
- `utils.py` : écriture UTF-8 des CSV/JSON.
- `main.py` : alternative en ligne de commande au workflow notebooks.
- `selected_videos.csv` : journal des URLs examinées et statut de chaque tentative.
- `audio/` : extraits `<video_id>.mp3` (ignorés par Git).
- `outputs/transcripts/` : transcription texte par vidéo.
- `outputs/results/` : résultat JSON par vidéo.
- `outputs/summary.csv` et `outputs/summary.json` : résumé global.
- `tests/test_core.py` : tests locaux `unittest`, sans YouTube ni appel Mistral.

## Installation sous Windows PowerShell

```powershell
cd C:\Users\BadrJOULALI\Downloads\mistral_darija_stt_benchmark
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Vérifiez FFmpeg :

```powershell
ffmpeg -version
ffprobe -version
```

S'il manque :

```powershell
winget install --id Gyan.FFmpeg -e
```

Ouvrez ensuite un nouveau PowerShell.

## Utilisation recommandée : notebooks

Lancez Jupyter depuis la racine du projet :

```powershell
jupyter notebook
```

Exécutez dans l'ordre :

1. `notebooks/00_inspect_dataset.ipynb`
2. `notebooks/01_prepare_audio_samples.ipynb`
3. configurez les variables ci-dessous vous-même ;
4. `notebooks/02_run_mistral_benchmark.ipynb`

Si le SQLite se trouve sur une autre machine mais que `audio/` contient déjà
les extraits, ouvrez directement les notebooks 01 et 02 dans leur nouveau mode
**audios existants**. Ce mode ne lit pas le SQLite, ne contacte pas YouTube et
ne nécessite ni `yt-dlp` ni FFmpeg. Le notebook 02 sélectionne par défaut un
seul audio sans résultat Mistral réussi afin de limiter le coût du test API.

La préparation audio n'appelle jamais Mistral. Seul le troisième notebook effectue des appels authentifiés.

## Configuration Mistral — à faire localement

Dans le PowerShell séparé qui lancera Jupyter, saisissez vous-même :

```powershell
$env:MISTRAL_API_KEY="votre_cle"
$env:MISTRAL_MODEL="voxtral-mini-latest"
jupyter notebook
```

La documentation officielle actuelle recommande `voxtral-mini-latest` pour la transcription hors ligne. Le projet exige tout de même que `MISTRAL_MODEL` soit explicitement configuré. La clé n'est ni affichée ni enregistrée.

## Alternative en ligne de commande

```powershell
python main.py --dataset "C:\Users\BadrJOULALI\Downloads\darija_youtube_extraction\darija_youtube_clean.sqlite"
```

Forcer le retraitement des résultats réussis :

```powershell
python main.py --dataset "C:\Users\BadrJOULALI\Downloads\darija_youtube_extraction\darija_youtube_clean.sqlite" --force
```

Pour un petit essai préalable :

```powershell
python main.py --dataset "C:\Users\BadrJOULALI\Downloads\darija_youtube_extraction\darija_youtube_clean.sqlite" --count 1
```

## Reprise et résultats

Si `outputs/results/<video_id>.json` existe avec `status: success`, la vidéo est ignorée sauf avec `--force` ou `FORCE = True` dans le notebook 02. Les résultats partiels sont écrits après chaque vidéo.

Avec 20 réussites, la durée totale transcrite est de 2 400 secondes, soit 40 minutes.

## Erreurs fréquentes

- `MISTRAL_API_KEY is not configured.` : définissez la variable dans le PowerShell qui lance Jupyter.
- `MISTRAL_MODEL is not configured.` : définissez par exemple `voxtral-mini-latest`.
- FFmpeg introuvable : installez-le avec `winget`, puis rouvrez PowerShell.
- Vidéo privée/supprimée : elle est marquée `unavailable`, puis le dataset continue.
- Vidéo trop courte : elle est marquée `too_short`.
- Échec de téléchargement/extraction : elle est marquée `download_error`.
- Erreurs API 401/403/404/413/429/500+ : le résultat JSON contient un message compréhensible, sans secret, puis le benchmark continue.

Documentation utilisée : [transcription hors ligne Mistral](https://docs.mistral.ai/studio-api/audio/speech_to_text/offline_transcription) et [endpoint Audio Transcriptions](https://docs.mistral.ai/api/endpoint/audio/transcriptions).

## Phase 2 — contrôle des pseudo-labels avec un modèle CTC

Cette phase n'appelle pas l'API Mistral. Elle réutilise uniquement les résultats `status: success` déjà présents dans `outputs/results/`.

Le modèle pilote est `facebook/omniASR-CTC-300M`, résolu vers la carte
`omniASR_CTC_300M_v2`. Les variantes 1B, 3B et 7B sont déclarées dans
l'interface, mais ne sont jamais téléchargées automatiquement. Le modèle
`boumehdi/wav2vec2-large-xlsr-moroccan-darija` est également disponible comme
baseline Transformers.

OmniASR dépend de `fairseq2n`, qui ne fournit pas de wheel Windows natif. Sous
Windows, utilisez la baseline Transformers ci-dessus. Pour OmniASR, utilisez
Linux ou une distribution WSL2 (Ubuntu) avec accès GPU.

Les extraits de 120 secondes sont automatiquement découpés en blocs de 30
secondes avant l'inférence, puis les transcriptions sont fusionnées. Cela
respecte la limite de 40 secondes du pipeline OmniASR non streaming et réduit
la consommation VRAM de la baseline Transformers.

### Nouveaux composants

- `audit_ctc_readiness.py` : audit non destructif des fichiers, outils, dépendances et ressources GPU.
- `ctc_transcriber.py` : inférence CTC limitée, reprise, cache et écriture d'un JSON par segment.
- `text_normalizer.py` : normalisation reproductible arabe, latin, Arabizi et code-switching.
- `quality_scorer.py` : CER/WER d'accord, similarités et score provisoire non calibré.
- `calibrate_threshold.py` : seuil manuel, régression logistique et calibration isotonic après annotation humaine.
- `notebooks/03_run_ctc_quality_check.ipynb` : vérification et pilote protégé par `RUN_CTC = False`.
- `requirements-ctc.txt` : dépendances optionnelles de la phase CTC.
- `tests/test_ctc_pipeline.py` : tests sans téléchargement de modèle ni appel réseau.

### Audit avant installation

```powershell
python audit_ctc_readiness.py
python ctc_transcriber.py --dry-run --prepare-annotations
```

Le rapport est écrit dans `outputs/ctc_readiness_audit.json`. Aucun secret n'est lu ou enregistré.

### Installation sur une machine GPU

Installez d'abord la version de PyTorch correspondant au pilote CUDA avec le sélecteur officiel de PyTorch, puis :

```powershell
pip install -r requirements-ctc.txt
```

Pour une RTX 50xx (`sm_120`), utilisez une distribution PyTorch compilée avec
CUDA 12.8 ou plus récent. Exemple conservateur avec un pilote CUDA 12.9 :

```powershell
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -r requirements-ctc.txt
```

Le pilote CTC vérifie explicitement que `sm_120` figure dans les architectures
prises en charge par PyTorch avant de télécharger le modèle.

Vérifiez :

```powershell
nvidia-smi
ffmpeg -version
ffprobe -version
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Le code refuse par défaut l'inférence lourde si CUDA n'est pas détecté. `--allow-cpu` existe uniquement pour un essai CPU volontaire et très limité.

### Pilote sécurisé

Commencez par un seul audio :

```powershell
python ctc_transcriber.py --model boumehdi/wav2vec2-large-xlsr-moroccan-darija --limit 1 --batch-size 1
```

Sous Linux/WSL2, remplacez le modèle par `facebook/omniASR-CTC-300M`.

Après validation du premier résultat, passez au maximum à trois :

```powershell
python ctc_transcriber.py --model facebook/omniASR-CTC-300M --limit 3 --batch-size 1
```

Un résultat CTC `status: success` utilisant le même modèle est ignoré lors d'une reprise, sauf avec `--force`.

### Sorties

- `outputs/ctc_results/<segment_id>.json` : transcription brute/normalisée, modèle, appareil, temps, confiance et taux de blank si accessibles.
- `outputs/quality_results/<segment_id>.json` : comparaison Mistral/CTC.
- `outputs/ctc_human_annotations.csv` : feuille d'annotation humaine.

Le CER et le WER de cette phase mesurent **l'accord entre Mistral et CTC**, pas l'exactitude par rapport à une vérité humaine.

### Annotation humaine

Les valeurs autorisées pour `human_label` sont :

- `good` : utilisable presque sans correction ;
- `medium` : globalement utile, mais correction nécessaire ;
- `reject` : très incorrect, mauvaise langue, absence de parole ou audio inutilisable.

Les colonnes `human_label`, `corrected_transcript` et `reviewer_notes` restent vides jusqu'à leur remplissage par un humain. Une reprise du pipeline préserve ces colonnes.

### Calibration

```powershell
python calibrate_threshold.py --minimum 30 --target-precision 0.90
```

La calibration :

- refuse un jeu insuffisant ou sans diversité de labels ;
- sépare calibration et test par `video_id` ;
- compare un seuil manuel, une régression logistique et, avec assez de données, une calibration isotonic ;
- rapporte précision, rappel, F1, matrice de confusion et taux de rétention ;
- favorise une forte précision pour la classe `good`.

Les 20 extraits actuels permettent seulement de vérifier le pipeline. Ils ne suffisent pas pour valider scientifiquement un seuil de production. Prévoir au moins 100 à 300 segments courts, stratifiés et annotés humainement.

### Tests

```powershell
python -m unittest discover -s tests -v
```
