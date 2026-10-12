import re
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import distance
from rapidfuzz import fuzz

# Ensure stopwords are downloaded
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)


CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have",
    "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
    "how'll": "how will", "how's": "how is", "i'd": "i would",
    "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
    "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
    "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
    "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
    "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
    "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have", "so's": "so as",
    "that'd": "that would", "that'd've": "that would have", "that's": "that is",
    "there'd": "there would", "there'd've": "there would have", "there's": "there is",
    "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
    "they'll've": "they will have", "they're": "they are", "they've": "they have",
    "to've": "to have", "wasn't": "was not", "we'd": "we would",
    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what'll've": "what will have", "what're": "what are",
    "what's": "what is", "what've": "what have", "when's": "when is",
    "when've": "when have", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is",
    "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
    "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
    "y'all'd've": "you all would have", "y'all're": "you all are",
    "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
    "you'll": "you will", "you'll've": "you will have", "you're": "you are",
    "you've": "you have"
}


def preprocess(q):
    """Clean and standardize input question text."""
    q = str(q).lower().strip()

    # Special characters replacement
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # Remove [math] tags
    q = q.replace('[math]', '')

    # Standardize numbers
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontract words
    q_decontracted = []
    for word in q.split():
        if word in CONTRACTIONS:
            word = CONTRACTIONS[word]
        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Remove HTML tags
    q = BeautifulSoup(q, "html.parser").get_text()

    # Remove punctuation
    pattern = re.compile(r'\W')
    q = re.sub(pattern, ' ', q).strip()

    return q


def test_common_words(q1, q2):
    """Count common unique words between q1 and q2."""
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    w1.discard('')
    w2.discard('')
    return len(w1 & w2)


def test_total_words(q1, q2):
    """Count total unique words between q1 and q2."""
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    w1.discard('')
    w2.discard('')
    return len(w1) + len(w2)


def test_fetch_token_features(q1, q2):
    """Extract 8 token-based features."""
    SAFE_DIV = 0.0001
    stop_words = stopwords.words("english")
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in stop_words])
    q2_words = set([word for word in q2_tokens if word not in stop_words])

    q1_stops = set([word for word in q1_tokens if word in stop_words])
    q2_stops = set([word for word in q2_tokens if word in stop_words])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


def test_fetch_length_features(q1, q2):
    """Extract 3 length-based features."""
    length_features = [0.0] * 3
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2.0

    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1.0) if (strs and len(strs[0]) > 0) else 0.0

    return length_features


def test_fetch_fuzzy_features(q1, q2):
    """Extract 4 fuzzy matching features."""
    fuzzy_features = [0.0] * 4
    fuzzy_features[0] = float(fuzz.QRatio(q1, q2))
    fuzzy_features[1] = float(fuzz.partial_ratio(q1, q2))
    fuzzy_features[2] = float(fuzz.token_sort_ratio(q1, q2))
    fuzzy_features[3] = float(fuzz.token_set_ratio(q1, q2))
    return fuzzy_features


def query_point_creator(q1, q2, cv):
    """Build the exact 6,022-dimensional feature vector for a pair of questions."""
    input_query = []

    # 1. Preprocessing
    q1_prep = preprocess(q1)
    q2_prep = preprocess(q2)

    # 2. Basic features (7 features)
    # Length of strings
    input_query.append(len(q1_prep))
    input_query.append(len(q2_prep))

    # Word counts
    q1_num_words = len([w for w in q1_prep.split(" ") if w])
    q2_num_words = len([w for w in q2_prep.split(" ") if w])
    input_query.append(q1_num_words)
    input_query.append(q2_num_words)

    # Common and total words
    common_w = test_common_words(q1_prep, q2_prep)
    total_w = test_total_words(q1_prep, q2_prep)
    word_share = round(common_w / total_w, 2) if total_w > 0 else 0.0

    input_query.append(common_w)
    input_query.append(total_w)
    input_query.append(word_share)

    # 3. Token features (8 features)
    token_features = test_fetch_token_features(q1_prep, q2_prep)
    input_query.extend(token_features)

    # 4. Length-based features (3 features)
    length_features = test_fetch_length_features(q1_prep, q2_prep)
    input_query.extend(length_features)

    # 5. Fuzzy features (4 features)
    fuzzy_features = test_fetch_fuzzy_features(q1_prep, q2_prep)
    input_query.extend(fuzzy_features)

    # 6. Bag of words features (3,000 + 3,000 features)
    q1_bow = cv.transform([q1_prep]).toarray()
    q2_bow = cv.transform([q2_prep]).toarray()

    feature_vec = np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))

    feature_details = {
        "q1_preprocessed": q1_prep,
        "q2_preprocessed": q2_prep,
        "common_words": common_w,
        "total_words": total_w,
        "word_share": word_share,
        "fuzz_ratio": fuzzy_features[0],
        "fuzz_partial_ratio": fuzzy_features[1],
        "token_sort_ratio": fuzzy_features[2],
        "token_set_ratio": fuzzy_features[3],
        "abs_len_diff": length_features[0],
        "longest_substr_ratio": length_features[2],
    }

    return feature_vec, feature_details
