import re
import unicodedata

class TextPreprocessor:
    def __init__(self, lemma_file="lemmas.txt", contraction_file="contractions.txt",stop_file="stopwords.txt"):
        self.lemma_dict = {}
        self.contraction_dict = {}
        self._load_dictionary(lemma_file, self.lemma_dict)
        self._load_dictionary(contraction_file, self.contraction_dict)
        self._load_stopwords(stop_file)
    def _load_dictionary(self, file_path, target_dict):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.strip().split(',')
                    for part in parts:
                        if ':' in part:
                            key, val = part.split(':')
                            target_dict[key.strip().lower()] = val.strip().lower()
        except FileNotFoundError:
            print(f"Warning: {file_path} not found.")
    def _load_stopwords(self, file_path):
        """Loads stopwords from a comma-separated text file into a set."""
        try:
            with open(file_path, 'r') as f:
                content = f.read().split(',')
                # Strip spaces and convert to set for O(1) lookup
                self.stop_words = {w.strip().lower() for w in content if w.strip()}
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Stopword removal will be skipped.")

    def clean_noise(self, text):
        # 1. Lowercase and Normalize Accents (résumé -> resume)
        if isinstance(text, list):
         text = " ".join(map(str, text))
    
    # Now it's safe to call .lower()
        text = text.lower()
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # 2. Expand Contractions 
        # IMPROVEMENT: Use regex to find contractions even if they are next to punctuation
        for contraction, expansion in self.contraction_dict.items():
            # Use word boundaries \b to ensure we don't replace parts of other words
            text = re.sub(rf'\b{re.escape(contraction)}\b', expansion, text)

        # 3. Remove structural noise
        text = re.sub(r'<.*?>', ' ', text)  # HTML
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text) # URLs
        text = re.sub(r'\S+@\S+', ' ', text) # Emails

        # 4. Clean Punctuation
        text = re.sub(r'[-_]', ' ', text) # Hyphens to spaces
        text = re.sub(r'[^\w\s]', ' ', text) # Remove all other punctuation
        text = re.sub(r'\d+', ' ', text) # Remove numbers

        # 5. Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _apply_lemmatization(self, word):
        word = word.lower().strip()
        if word in self.lemma_dict:
            return self.lemma_dict[word]

        if len(word) > 3:
            # Move -ly lower or make it stricter
            if word.endswith("ies"): 
                return word[:-3] + "y"
            
            if word.endswith("ing") and len(word) > 4: 
                stem = word[:-3]
                if len(stem) > 1 and stem[-1] == stem[-2]:
                    return stem[:-1]
                return stem
            
            if word.endswith("ed") and len(word) > 4: 
                return word[:-2]
            
            if word.endswith("es") and not word.endswith("ses"):
                return word[:-2]

            if word.endswith("s") and not word.endswith("ss"):
                return word[:-1]
            
            # Adverb handling
            if word.endswith("ly") and len(word) > 5:
                return word[:-2]

            if word.endswith("i"):
                return word[:-1] + "y"

        return word
    def tokenize(self, cleaned_text):
        """
        Takes cleaned text and returns a list of individual words.
        """
        # 1. Split by whitespace
        raw_tokens = cleaned_text.split()

        # 2. Filter out empty strings and very short noise
        # (e.g., if a word was "a", noise cleaning might leave a space)
        tokens = [t for t in raw_tokens if len(t) > 1]
        
        return tokens
    def remove_stopwords(self, tokens):
        """
        Input: List of raw tokens
        Output: List of tokens with stopwords removed
        """
        # We also filter out tokens with length < 2 as a safety measure 
        # to remove stray punctuation or single-letter noise.
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]
    def preprocess(self, text):
        """Processes raw text into final vector-ready tokens."""
        # Step 1: Noise Removal
        cleaned = self.clean_noise(text)
        # Step 2: Tokenization
        tokens = self.tokenize(cleaned)
        # Step 3: Stopword Removal
        filtered = self.remove_stopwords(tokens)
        # Step 4: Lemmatization
        final = [self._apply_lemmatization(t) for t in filtered]
        return final
import math
from collections import defaultdict

import math
from collections import defaultdict



class TfidfVectorizer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.vocabulary = set()
        self.idf_vector = {}
        self.num_docs = 0

    def fit(self, raw_documents):
        """LEARNING PHASE: Learns IDF from a collection of documents."""
        self.vocabulary = set()
        self.idf_vector = {}
        self.num_docs = len(raw_documents)
        doc_freq_counts = defaultdict(int)

        for doc in raw_documents:
            tokens = self.preprocessor.preprocess(doc)
            self.vocabulary.update(tokens)
            for t in set(tokens):
                doc_freq_counts[t] += 1

        # Calculate Global IDF
        for term in self.vocabulary:
            df = doc_freq_counts[term]
            self.idf_vector[term] = math.log10((self.num_docs + 1) / (df + 1)) + 1

    def transform(self, text):
        """PRODUCTION PHASE: Turns any string into a (vector, norm) tuple."""
        tokens = self.preprocessor.preprocess(text)
        if not tokens:
            return {}, 0.0

        # Calculate TF
        term_counts = defaultdict(int)
        for t in tokens:
            term_counts[t] += 1
        
        total_tokens = len(tokens)
        tfidf_map = {}
        sum_sq = 0.0

        for term, count in term_counts.items():
            if term in self.idf_vector:
                tf = count
                tfidf_score = tf * self.idf_vector[term]
                tfidf_map[term] = tfidf_score
                sum_sq += tfidf_score ** 2

        return tfidf_map, math.sqrt(sum_sq)
    
def calculate_cosine_similarity(vec1, norm1, vec2, norm2):
    """Universal math utility: No knowledge of text or TF-IDF."""
    if norm1 == 0 or norm2 == 0: return 0.0
    dot_product = 0.0
    # Iterate over the smaller vector for efficiency
    small, large = (vec1, vec2) if len(vec1) < len(vec2) else (vec2, vec1)
    for term, weight in small.items():
        if term in large:
            dot_product += weight * large[term]
    return dot_product / (norm1 * norm2)