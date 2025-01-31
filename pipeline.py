import string
import numpy as np
import pandas as pd
import spacy
import nltk

from gensim.models import KeyedVectors
from nltk import word_tokenize, pos_tag, ngrams
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
from datasets import load_dataset
import pyinflect

# Laad Word2Vec
word2vec = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
nlp = spacy.load("en_core_web_sm")

# Laad Glove
glove_file = "glove.6B.300d.txt"

# Functie om de GloVe embeddings te laden
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]  # Eerste woord is de sleutel
            vector = np.asarray(values[1:], dtype='float32')  # De rest is de vector
            embeddings[word] = vector
    return embeddings

# Laden van embeddings
glove = load_glove_embeddings(glove_file)

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None  # Geen correcte WordNet POS-tag
        
def conjugate_verb(verb, target_tag):
    # Eerst terugbrengen naar de stamvorm
    base_form = lemmatizer.lemmatize(verb, wordnet.VERB)

    # Gebruik spaCy voor de juiste vervoeging
    doc = nlp(base_form)
    
    if target_tag == "VBD":  # Verleden tijd
        past_tense = doc[0]._.inflect("VBD")
        return past_tense if past_tense else base_form + "ed"  # Fallback
    
    elif target_tag == "VBN":  # Voltooid deelwoord
        past_participle = doc[0]._.inflect("VBN")
        return past_participle if past_participle else base_form + "ed"
    
    elif target_tag == "VBZ":  # Derde persoon enkelvoud tegenwoordige tijd
        vbz_form = doc[0]._.inflect("VBZ")
        if vbz_form:
            return vbz_form

        # Fallback voor regelmatige werkwoorden
        if base_form.endswith(("ch", "sh", "x", "s", "o")):
            return base_form + "es"
        elif base_form.endswith("y") and base_form[-2] not in "aeiou":
            return base_form[:-1] + "ies"
        else:
            return base_form + "s"

    return verb  # Geen verandering nodig

def get_embedding(woord, model): 
    if model == word2vec:
        return word2vec[woord] if woord in word2vec else None
    elif model == glove:
        return model.get(woord) if woord in glove else None

def find_best_synonym(zin, doelwoord, model, threshold=0.3, n=3):
    # Tokeniseer en tag de zin
    tokens = word_tokenize(zin)
    tags = pos_tag(tokens)

    # Zoek POS-tag van het doelwoord in de zin
    doelwoord_tag = None
    for token, tag in tags:
        if token.lower() == doelwoord.lower():  
            doelwoord_tag = tag
            break

    if not doelwoord_tag:
        return "Geen POS-tag gevonden voor doelwoord", 0

    # print(f"DEBUG: POS-tag van '{doelwoord}': {doelwoord_tag}")

    # Converteer naar WordNet POS
    wordnet_pos = get_wordnet_pos(doelwoord_tag)
    if not wordnet_pos:
        return "Geen geschikte WordNet POS-tag", 0

    # Haal de synoniemen op uit WordNet met correcte POS
    synsets = wordnet.synsets(doelwoord, pos=wordnet_pos)

    # Verzamel synoniemen en filter ongewenste woorden
    wordnet_synoniemen = set()
    base_form = lemmatizer.lemmatize(doelwoord, wordnet_pos)
    
    for synset in synsets:
        for lemma in synset.lemmas():
            synoniem = lemma.name().replace("_", " ").lower()
            if synoniem != doelwoord and synoniem != base_form:  # Filter identieke woorden
                wordnet_synoniemen.add(synoniem)

    # print(f"DEBUG: Gevonden synoniemen voor '{doelwoord}': {wordnet_synoniemen}")

    # Maak n-grams
    context_ngrams = list(nltk.ngrams(tokens, n))

    # Verkrijg de embedding van het doelwoord
    if doelwoord in model:
        doelwoord_embedding = get_embedding(doelwoord, model)
    else:
        return "Geen embedding voor het doelwoord", 0


    # Bereken cosine similarity met context-n-grams
    similarity_scores = {}
    for synoniem in wordnet_synoniemen:
        if synoniem in model:
            synoniem_embedding = get_embedding(synoniem, model)

            total_similarity = 0
            valid_ngram_count = 0

            for ngram in context_ngrams:
                valid_words = [word for word in ngram if word in model]
                
                if valid_words:
                    ngram_embedding = np.mean([get_embedding(word, model) for word in valid_words], axis=0)
                    
                    if ngram_embedding.ndim == 1:
                        similarity_score = 1 - cosine(synoniem_embedding, ngram_embedding)

                        if similarity_score > threshold:
                            # Extra straf als het synoniem te dicht bij de stam van het doelwoord ligt
                            if lemmatizer.lemmatize(synoniem, wordnet.VERB) == base_form:
                                similarity_score *= 0.9
                            
                            total_similarity += similarity_score
                            valid_ngram_count += 1

            if valid_ngram_count > 0:
                average_similarity = total_similarity / valid_ngram_count
                similarity_scores[synoniem] = average_similarity

    # Sorteer op hoogste similarity
    sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # print(f"DEBUG: Similarity scores: {sorted_similarity_scores}")

    if not sorted_similarity_scores:
        return "Geen geschikt synoniem", 0

    beste_synoniem = sorted_similarity_scores[0][0]

    # **Vervoeg het synoniem correct!**
    if wordnet_pos == wordnet.VERB:
        vervoegd_synoniem = conjugate_verb(beste_synoniem, doelwoord_tag)
        # print(f"DEBUG: Oorspronkelijk synoniem: {beste_synoniem}, Vervoegd synoniem: {vervoegd_synoniem}")
        beste_synoniem = vervoegd_synoniem

    return beste_synoniem, sorted_similarity_scores[0][1]

zin = 'The car decelerates as it approaches the red light.'
doelwoord = 'decelerates'
synoniem =  find_best_synonym(zin, doelwoord, word2vec)
print(f'gevonden synoniem voor {doelwoord}: {synoniem}')