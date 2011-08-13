# -*- coding: UTF-8
"""Unit tests for the string_distance_measures module"""
import string_distance_measures


def test_ngrams():
    """check our local ngrams routine (rather than nltk's) works"""
    l1 = [1,2,3,4]
    assert string_distance_measures.ngrams(l1, 2) == [(1,2), (2,3), (3,4)]

    l2 = []
    assert string_distance_measures.ngrams(l2, 2) == []

    l2 = [1]
    assert string_distance_measures.ngrams(l2, 2) == []

    l2 = [1, 2]
    assert string_distance_measures.ngrams(l2, 2) == [(1,2)]


def test_calls_work():
    """test the calls to check that the signature works as expected"""
    s1 = "string1"
    s2 = "string2"
    assert string_distance_measures.distance_levenshtein_distance(s1, s1) == 0
    assert string_distance_measures.distance_levenshtein_distance(s1, s2) > 0

    assert string_distance_measures.distance_levenshtein_jaro_winkler(s1, s1) == 0
    assert string_distance_measures.distance_levenshtein_jaro_winkler(s1, s2) > 0

    assert string_distance_measures.distance_levenshtein_ratio(s1, s1) == 0
    assert string_distance_measures.distance_levenshtein_ratio(s1, s2) > 0

    assert string_distance_measures.distance_title_len(s1, s1) == 0
    assert string_distance_measures.distance_title_len(s1, s2) == 0  # note same length strings!

    s1 = string_distance_measures.make_terms_from_string(s1)
    s2 = string_distance_measures.make_terms_from_string(s2)
    assert string_distance_measures.distance_nbr_title_terms(s1, s1) == 0
    assert string_distance_measures.distance_nbr_title_terms(s1, s2) == 0  # note same length strings!

    dist = string_distance_measures.distance_cosine_measure(s1, s1)
    print dist
    assert dist == 0
    dist = string_distance_measures.distance_cosine_measure(s1, s2)
    print dist
    assert dist == 1


def test_levenshtein_distance():
    """test Levenshtein Distance"""
    # test 1 character difference
    s1 = "string1"
    s2 = "string2"
    dist = string_distance_measures.distance_levenshtein_distance(s1, s2)
    print dist
    assert dist == 1

    # test 0 character difference
    assert string_distance_measures.distance_levenshtein_distance(s1, s1) == 0


def test_cosine_distance():
    """test the Vector Model's cosine similarity distance measurement"""
    s1 = "string1"
    s2 = "string2"
    dist = string_distance_measures.distance_cosine_measure(s1, s1)
    print dist
    assert dist == 0
    dist = string_distance_measures.distance_cosine_measure(s1, s2)
    print dist
    assert dist == 1

    s3 = "mary had a little lamb"
    s4 = "mary had another little lamb"
    dist = string_distance_measures.distance_cosine_measure(s3, s4)
    print "distance:", dist
    assert dist > 0.19 and dist < 0.21  # approx. 0.2

    s3 = "mary had a little lamb"
    s4 = "mary had little lamb"
    dist = string_distance_measures.distance_cosine_measure(s3, s4)
    print "distance:", dist
    assert dist > 0.1 and dist < 0.11  # approx. 0.105572809

    dist = string_distance_measures.distance_cosine_measure("", "")
    print "distance:", dist
    assert dist == 0 


def test_ngram_distances_one_word():
    """test ngram methods with a single word"""
    s1 = "string1"
    s2 = "string2"
    
    assert string_distance_measures.distance_unigrams_same(s1, s1) == 0
    assert string_distance_measures.distance_unigrams_same(s1, s2) == 1

    dist = string_distance_measures.distance_bigrams_same(s1, s1)
    print dist
    assert dist == 1  # no bigrams so regarded as being different!
    dist = string_distance_measures.distance_bigrams_same(s1, s2)
    print dist
    assert dist == 1

    dist = string_distance_measures.distance_trigrams_same(s1, s1)
    print dist
    assert dist == 1  # no trigrams so regarded as being different
    dist = string_distance_measures.distance_trigrams_same(s1, s2)
    print dist
    assert dist == 1


def test_ngram_distances_four_words():
    """test ngram methods with four word sentences"""
    s1 = "string1 some thing else"
    s2 = "string2 some thing else"
    
    assert string_distance_measures.distance_unigrams_same(s1, s1) == 0
    dist = string_distance_measures.distance_unigrams_same(s1, s2)
    print dist
    assert dist == 0.4

    dist = string_distance_measures.distance_bigrams_same(s1, s1)
    print dist
    assert dist == 0
    dist = string_distance_measures.distance_bigrams_same(s1, s2)
    print dist
    assert dist == 0.5

    dist = string_distance_measures.distance_trigrams_same(s1, s1)
    print dist
    assert dist == 0
    dist = string_distance_measures.distance_trigrams_same(s1, s2)
    print dist
    assert dist > 0.6 and dist < 0.7  # approx. 0.66666667
