# -*- coding: UTF-8
"""A selection of unicode string distance metrics including uni/bi/trigrams, 
   cosine similarity and edit distances"""
import Levenshtein  # http://pypi.python.org/pypi/python-Levenshtein/ via "pip-2.7 install python-Levenshtein"
import numpy as np

# Also - FuzzyWuzzy
# http://seatgeek.com/blog/dev/fuzzywuzzy-fuzzy-string-matching-in-python
# might have value for messy sub-string matching


def ngrams(sequence, n):
    """Create ngrams from sequence, e.g. ([1,2,3], 2) -> [(1,2), (2,3)]
       Note that fewer sequence items than n results in an empty list being returned"""
    # credit: http://stackoverflow.com/questions/2380394/simple-implementation-of-n-gram-tf-idf-and-cosine-similarity-in-python
    sequence = list(sequence)
    count = max(0, len(sequence) - n + 1)
    return [tuple(sequence[i:i+n]) for i in range(count)] 


def make_terms_from_string(s):
    """turn string s into a list of unicode terms"""
    u = unicode(s)
    return u.split()


def distance_levenshtein_distance(t1, t2):
    """Return Levenshtein Distance for title strings, 
       0 means identical, >1 represents the edit distance"""
    return Levenshtein.distance(unicode(t1), unicode(t2))


def distance_levenshtein_jaro_winkler(t1, t2):
    """Return Jaro Winkler co-efficient of title strings,
       0 means that they're identical, >0 represents a distance"""
    return 1.0 - Levenshtein.jaro_winkler(unicode(t1), unicode(t2))


def distance_levenshtein_ratio(t1, t2):
    """Return Levenshtein Ratio of title strings, 
       0 means that they're identical, >0 represents a distance"""
    return 1.0 - Levenshtein.ratio(unicode(t1), unicode(t2))


def distance_jaro(t1, t2):
    """Return Jaro distance measure of title strings,
       0 means that they're identical, >0 represents a distance"""
    return 1.0 - Levenshtein.jaro(unicode(t1), unicode(t2))


def distance_title_len(t1, t2):
    """Count difference in length of title strings,
       0 means that they're identical, >0 represents the length distance"""
    return abs(len(t1) - len(t2))


def distance_nbr_title_terms(t1, t2):
    """Count difference of number of terms in titles,
       e.g. ['a', 'title'], ['title'] -> 1"""
    t1_terms = make_terms_from_string(t1)
    t2_terms = make_terms_from_string(t2)
    return abs(len(t1_terms) - len(t2_terms))


def distance_unigrams_same(t1, t2):
    """Unigram distance metric, term frequency is ignored,
       0 if unigrams are identical, 1.0 if no unigrams are common"""
    t1_terms = make_terms_from_string(t1)
    t2_terms = make_terms_from_string(t2)
    terms1 = set(t1_terms)
    terms2 = set(t2_terms)
    shared_terms = terms1.intersection(terms2)
    all_terms = terms1.union(terms2)
    dist = 1.0 - (len(shared_terms) / float(len(all_terms)))
    return dist


def distance_bigrams_same(t1, t2):
    """Bigram distance metric, term frequency is ignored,
       0 if bigrams are identical, 1.0 if no bigrams are common"""
    t1_terms = make_terms_from_string(t1)
    t2_terms = make_terms_from_string(t2)
    terms1 = set(ngrams(t1_terms, 2)) # was using nltk.bigrams
    terms2 = set(ngrams(t2_terms, 2))
    shared_terms = terms1.intersection(terms2)
    all_terms = terms1.union(terms2)
    dist = 1.0
    if len(all_terms) > 0:
        dist = 1.0 - (len(shared_terms) / float(len(all_terms)))
    return dist


def distance_trigrams_same(t1, t2):
    """Trigram distance metric, term frequency is ignored,
       0 if trigrams are identical, 1.0 if no trigrams are common"""
    t1_terms = make_terms_from_string(t1)
    t2_terms = make_terms_from_string(t2)
    terms1 = set(ngrams(t1_terms, 3)) # was using nltk.trigrams
    terms2 = set(ngrams(t2_terms, 3))
    shared_terms = terms1.intersection(terms2)
    all_terms = terms1.union(terms2)
    dist = 1.0
    if len(all_terms) > 0:
        dist = 1.0 - (len(shared_terms) / float(len(all_terms)))
    return dist


def distance_cosine_measure(t1, t2):
    """Calculate cosine similarity of two strings, return as a distance (1==absolutely different, 0==no difference),
       Background: http://en.wikipedia.org/wiki/Cosine_similarity"""
    
    def __cosine_similarity(v1, v2):
        norm_a = np.linalg.norm(v1)  # sqrt of sum of all-items-squared
        norm_b = np.linalg.norm(v2)
        abs_a_abs_b = norm_a * norm_b
        if abs_a_abs_b == 0:
            return 0
        else:
            a_dot_b = np.dot(v1, v2)
            result = a_dot_b / abs_a_abs_b
            return result
    
    # convert titles in to term lists
    if len(t1) == 0 and len(t2) == 0:
        cos_sim = 1.0  # empty strings are absolutely the same for our cosine similarity score
    else:
        words = set()
        t1_terms = make_terms_from_string(t1)
        t2_terms = make_terms_from_string(t2)
        # add term lists to the set of known words
        words.update(t1_terms)
        words.update(t2_terms)
        # build dict of indexes from the word list (so we can index into our matrix)
        word_lookup = dict([(w, n) for (n, w) in enumerate(words)])
        # build 2 row matrix with nbr-of-words columns
        matrix = np.zeros((2, len(words)))
        # for each word, update the relevant row with the new word count
        for nd, d in enumerate([t1_terms, t2_terms]):
            for w in d:
                assert w in word_lookup  # if not present - we've got a bug!
                matrix[nd][word_lookup[w]] += 1
        cos_sim = __cosine_similarity(matrix[0], matrix[1])
    if cos_sim > 1.0 and cos_sim < 1.0000000005:
        cos_sim = 1.0  # round down if fractionally over 1.0
    if cos_sim < 0 or cos_sim > 1.0:
        print "Error in distance_cosine_measure"
        print t1
        print t2
        print cos_sim
    assert cos_sim >= 0 and cos_sim <= 1.0
    return 1.0 - cos_sim
