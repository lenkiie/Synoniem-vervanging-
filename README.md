# Lightweight Synonym Pipeline

Een efficiënte aanpak voor synoniemvervanging die lightweight word embeddings combineert met slimme filters, waardoor complexe taalmodellen vaak niet nodig zijn. 

## Features
- Hybride aanpak die contextloze word embeddings (Word2Vec/GloVe) verrijkt met contextuele filters
- Filtering via:
  - POS-tagging voor grammaticale correctheid
  - N-gram context windows voor lokale context analyse
  - Lemmatization om vervoegingen van hetzelfde woord te voorkomen
  - Configureerbare similarity thresholds
- Automatische werkwoordsvervoeging na synoniemkeuze
- Ondersteuning voor zowel Word2Vec als GloVe embeddings
- Lichtgewicht alternatief voor grote taalmodellen, geschikt voor edge devices en lokale implementaties

## Prerequisites
- Python 3.10+
- pip
- Voldoende schijfruimte voor de embedding modellen (~4GB)

## Installatie

1. Clone de repository:
```bash
git clone [https://github.com/lenkiie/Synoniem-vervanging-]

2. Installeer de benodigde libraries:
```bash
pip install -r requirements.txt

3. Download de benodigde embedding modellen:

**Word2Vec:**
- Download de binary file van Google Word2Vec model:
  https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
- Dit is het originele Google News model (ongeveer 3.5GB)

**GloVe:**
- Download de vectoren van Stanford's GloVe project:
  https://nlp.stanford.edu/data/glove.6B.zip
- We raden aan om de 300d versie te gebruiken (glove.6B.300d.txt)
- Dit model is getraind op Wikipedia 2014 + Gigaword 5 (ongeveer 2GB)
 