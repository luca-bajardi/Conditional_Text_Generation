#IMPORTING AND SHOWCASING
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
from nltk import word_tokenize
from nltk import pos_tag
from transformers import pipeline

def bleu_score(references, model_output, k):
    '''
    references is the y_true, list of string
    model output, list of string
    k is the number of n-grams you wanto to consider, from 2 to 5 is acceptable

    the "right" sentence for model_output[i] is references[i].
    '''
    if len(references) != len(model_output):
        print("Error, references and model_output must be same length")
        return -1
    if k < 2 or k > 5:
        print("Error, k must be between 2 and 5")
        return -2

    N = len(references)
    sum_scores = 0
    chencherry = SmoothingFunction()

    if k == 2:
        weights = (1./2.,1./2.)
    elif k == 3:
        weights = (1./3.,1./3.,1./3.)
    elif k == 4:
        weights = (1./4.,1./4.,1./4.,1./4.)
    else:
        #k==5
        weights = (1./5.,1./5.,1./5.,1./5.,1./5.)


    for i in range(N):
        ref = references[i]
        out = model_output[i]

        ref_words = str.split(ref)
        out_words = str.split(out)

        score_i = sentence_bleu([ref_words],out_words,weights,smoothing_function = chencherry.method1)

        sum_scores += score_i

    return sum_scores/N


def self_bleu_score(model_output, k):
    '''
    model_output has list of strings
    k is the n-grams number n as before

    '''

    N = len(model_output)

    sum_scores = 0

    chencherry = SmoothingFunction()

    if k == 2:
        weights = (1./2.,1./2.)
    elif k == 3:
        weights = (1./3.,1./3.,1./3.)
    elif k == 4:
        weights = (1./4.,1./4.,1./4.,1./4.)
    else:
        #k==5
        weights = (1./5.,1./5.,1./5.,1./5.,1./5.)

    outputs = []
    for el in model_output:
        outputs.append(str.split(el))

    for i in range(N):
        out_words_copied = outputs.copy()
        #remove the i-esim string
        candidate = out_words_copied.pop(i)

        score_i = sentence_bleu(out_words_copied,candidate,weights,smoothing_function = chencherry.method1)

        sum_scores += score_i


    return sum_scores/N


def pos_bleu_score(references, model_output, k):
    '''
    references is the y_true, list of string
    model output, list of string
    k is the number of n-grams you wanto to consider, from 2 to 5 is acceptable

    the "right" sentence for model_output[i] is references[i].

    This is the same as bleu_score, with the only difference that now the senteces
    (both reference and model_outputs) are changed as Part of speech tags
    '''
    if len(references) != len(model_output):
        print("Error, references and model_output must be same length")
        return -1
    if k < 2 or k > 5:
        print("Error, k must be between 2 and 5")
        return -2

    N = len(references)
    sum_scores = 0
    chencherry = SmoothingFunction()

    if k == 2:
        weights = (1./2.,1./2.)
    elif k == 3:
        weights = (1./3.,1./3.,1./3.)
    elif k == 4:
        weights = (1./4.,1./4.,1./4.,1./4.)
    else:
        #k==5
        weights = (1./5.,1./5.,1./5.,1./5.,1./5.)


    for i in range(N):
        ref = references[i]
        out = model_output[i]
        ref_words = word_tokenize(ref)
        out_words = word_tokenize(out)

        ref_pos = pos_tag(ref_words)
        out_pos = pos_tag(out_words)

        ref_words = []
        for i in range(len(ref_pos)):
            ref_words.append(ref_pos[i][1])

        out_words = []
        for i in range(len(out_pos)):
            out_words.append(out_pos[i][1])

        score_i = sentence_bleu([ref_words],out_words,weights,smoothing_function = chencherry.method1)

        sum_scores += score_i

    return sum_scores/N


def bart_classifier(sequence_to_classify, candidate_labels):
    classifier = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli")
    candidate_labels = ['informal', 'formal']
    res = classifier(sequence_to_classify, candidate_labels)
    informality_level = res['scores'][0]
    formality_level = res['scores'][1] #1. - informality_level
    return res['scores']
