import numpy as np
import os
import json
import math
import csv
import collections
from collections import defaultdict

### CONSTANTS ###
O_XXX = 0
B_PER = 1
I_PER = 2
B_LOC = 3
I_LOC = 4
B_ORG = 5
I_ORG = 6
B_MISC = 7
I_MISC = 8
START = 9


def read_file(filename):
    stringInFile = ""
    with open(filename, "r", encoding="utf8") as fileObject:
        stringInFile = fileObject.read()
    return stringInFile


def read_all_lines(filename):
    with open(filename) as f:
        lines = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    lines = [line.strip() for line in lines]
    return lines


def read_tag_lines(filename):
    lines = read_all_lines(filename)
    desired_lines = lines[2::3]
    return desired_lines


def read_token_lines(filename):
    lines = read_all_lines(filename)
    desired_lines = lines[0::3]
    return desired_lines


def read_prediction_lines(filename):
    lines = read_all_lines(filename)
    desired_lines = lines[0::3]
    return desired_lines


def writeToCSVFile(rowList):
    with open('output.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id', 'Prediction'])
        for string in rowList:
            writer.writerow(string)


def tagNameToId(tag_string):
    if tag_string == "O": return 0
    if tag_string == "B-PER": return 1
    if tag_string == "I-PER": return 2
    if tag_string == "B-LOC": return 3
    if tag_string == "I-LOC": return 4
    if tag_string == "B-ORG": return 5
    if tag_string == "I-ORG": return 6
    if tag_string == "B-MISC": return 7
    if tag_string == "I-MISC": return 8


def idToTagName(tag_id):
    if tag_id == 0: return "O"
    if tag_id == 1: return "B-PER"
    if tag_id == 2: return "I-PER"
    if tag_id == 3: return "B-LOC"
    if tag_id == 4: return "I-LOC"
    if tag_id == 5: return "B-ORG"
    if tag_id == 6: return "I-ORG"
    if tag_id == 7: return "B-MISC"
    if tag_id == 8: return "I-MISC"


'''
def token_generation(data):
    tokenList = data.split(" ")
    token_to_id_map = {}

    k = 0
    for i in range(len(tokenList)):
        if tokenList[i] not in token_to_id_map:
            token_to_id_map[tokenList[i]] = k

            k += 1
        # id_to_token_map={v: k for k,v in token_to_id_map.items()}

    return token_to_id_map
'''


def token_generation(data):
    data = "\t".join(data)
    token_to_id_map = {}
    k = 0
    # for data in inputDataList:
    tokenList = data.split("\t")
    for i in range(len(tokenList)):
        if tokenList[i] not in token_to_id_map:
            token_to_id_map[tokenList[i]] = k
            k += 1
            # id_to_token_map={v: k for k,v in token_to_id_map.items()}

    return token_to_id_map


# returns matrix M such that M[i, j] = # of (t_i t_j) sequences
def createUnsmoothedTagBigramCounts(lines_of_tags):
    bigramCountMatrix = np.zeros([10, 10])
    bigramCountMatrix.fill(0.1)
    print(bigramCountMatrix)

    for l in range(len(lines_of_tags)):
        line = lines_of_tags[l].split('\t')
        #print (line)
        first_tag = tagNameToId(line[0])
        bigramCountMatrix[START, first_tag] += 1
        for i in range(1, len(line)):
            bigramCountMatrix[tagNameToId(line[i - 1]), tagNameToId(line[i])] += 1
            #print (bigramCountMatrix[tagNameToId(line[i - 1]), tagNameToId(line[i])])
    return bigramCountMatrix


def getBigramProbsFromCounts(bigramCountMatrix):
    bigramProbMatrix = bigramCountMatrix / bigramCountMatrix.sum(axis=1, keepdims=True)
    return bigramProbMatrix


'''
def getbaseline_matrix(tokenToIdMap,wordlines,taglines):
    words=wordlines.split(" ")
    tags=taglines.split(" ")
    word_types=len(tokenToIdMap)

    baseline_matrix_counts=np.zeros[(word_types,9)]
    for w,t in words,tags:
        tag_id=tagNameToId(t)
        token_id=tokenToIdMap[w]
        baseline_matrix_counts[token_id][tag_id]+=1
    return baseline_matrix_counts

def getBaselinePrediction(prediction_string,baseline_matrix_counts,tokenToIdMap):
    prediction_string=prediction_string.split(" ")
    result_tags=[]
    for token in prediction_string:
        token_id=tokenToIdMap[token]
        result=np.argmax(np.max(baseline_matrix_counts[token_id], axis=0))
        result_tags.append(idToTagName(result))
    return result_tags
'''


def getbaseline_matrix(tokenToIdMap, wordlines, taglines):
    wordlines = "\t".join(wordlines)
    taglines = "\t".join(taglines)
    words = wordlines.split("\t")
    tags = taglines.split("\t")
    word_types = len(tokenToIdMap)
    # print (word_types)
    # print (len(tags))

    baseline_matrix_counts = np.zeros((word_types, 9))
    # for w,t in words,tags:
    for i in range(len(words)):
        # print (tags[i])
        tag_id = tagNameToId(tags[i])
        # print (tag_id)
        token_id = tokenToIdMap[words[i]]
        baseline_matrix_counts[token_id][tag_id] += 1
        # if baseline_matrix_counts[token_id][tag_id] > 0:
        # print (baseline_matrix_counts[token_id][tag_id])
    # print (baseline_matrix_counts)

    return baseline_matrix_counts


def getBaselinePrediction(prediction_string, baseline_matrix_counts, tokenToIdMap):
    prediction_string = prediction_string.split("\t")
    # print (prediction_string)
    result_tags = []
    resultMap = {}
    index = 0
    tag = ""
    previousTag = None
    # currentTag = ""
    # startIndex = 0
    for token in prediction_string:
        if token in tokenToIdMap:
            token_id = tokenToIdMap[token]
            # print (baseline_matrix_counts[token_id])
            result = np.argmax((baseline_matrix_counts[token_id]))
            tag = (getTag(idToTagName(result)), index)
        else:
            # result_tags.append((getTag(idToTagName(0)), index))
            tag = (getTag(idToTagName(0)), index)

        result_tags.append(tag)
        index += 1
    # print (result_tags)
    return result_tags


def isPER(str):
    return str == "B-PER" or str == "I-PER" or str == "PER"


def isLOC(str):
    return str == "B-LOC" or str == "I-LOC" or str == "LOC"


def isORG(str):
    return str == "B-ORG" or str == "I-ORG" or str == "ORG"


def isMISC(str):
    return str == "B-MISC" or str == "I-MISC" or str == "MISC"


def getTag(tag):
    if (isPER(tag)):
        return "PER"
    if (isLOC(tag)):
        return "LOC"
    if (isORG(tag)):
        return "ORG"
    if (isMISC(tag)):
        return "MISC"
    return "O"


def getlexical_generation_probs(baseline_matrix, word, tag, tokenToIdMap):
    if word in tokenToIdMap:

        tag_id = tagNameToId(tag)
        token_id = tokenToIdMap[word]
        tag_column = baseline_matrix[:, tag_id]
        tag_count = np.sum(tag_column)
        word_count = baseline_matrix[token_id][tag_id]
        return (word_count + 0.1) / (tag_count + len(tokenmap))
    else:
        sum = np.sum(baseline_matrix)
        return 1.0 / sum


# returns list of most probable sequence of tag names for given sentence
# transitionProbs: ngram probs for tags
def runViterbi(transitionProbs, baseline_matrix, tokenToIdMap, sentence):
    sentenceList = sentence.split('\t')
    numrows = 9
    numcols = len(sentenceList)
    scoreMatrix = np.zeros([numrows, numcols])
    bptrMatrix = np.zeros([numrows, numcols])

    # calculate first column separately because previous column is start
    # for first column: score = P(t | <s>) * P(w | t), bptr = 0
    for i in range(numrows):
        lexprob = getlexical_generation_probs(baseline_matrix, sentenceList[0], idToTagName(i), tokenToIdMap)
        score = transitionProbs[START, i] * lexprob
        scoreMatrix[i, 0] = score
        # leave bptrMatrix[i, 0] as 0

    # calculate scores and backpointers for all other cols
    for j in range(1, numcols):
        prev_col = scoreMatrix[:, j - 1]
        for i in range(numrows):
            lex_gen_prob = getlexical_generation_probs(baseline_matrix, sentenceList[j], idToTagName(i), tokenToIdMap)
            score, bptr = calculateScoreAndBptr(prev_col, i, transitionProbs, lex_gen_prob)
            scoreMatrix[i, j] = score
            bptrMatrix[i, j] = bptr

    return getTagSequence(bptrMatrix, scoreMatrix)


# return tuple of (score, bptr) for a given point in the viterbi matrix ([tag, word])
# transition_probs = bigram prob matrix of tags
# lex_gen_prob = P(word | tag)
def calculateScoreAndBptr(prev_col_scores, tag_id, transition_probs, lex_gen_prob):
    # score = max_i (prev_col_scores[i] * P(tag | t_i)) * P(word | tag)
    # bptr = i that gives max score
    transition_probs_t = transition_probs[:, tag_id]  # size 9 array where [i] = P(tag | t_i)
    # score_times_transition = prev_col_scores * transition_probs_t[:START]  #  size 9 array, [i] = Score[i,j-1]*P(tag|t_i); have to cut off start value from tx probs
    score_times_transition = np.log(prev_col_scores) + np.log(transition_probs_t[:START])
    score_times_transition = np.exp(score_times_transition)
    best_prev_tag_id = np.argmax(score_times_transition)
    a = np.max(score_times_transition)
    b = lex_gen_prob
    # score = np.max(score_times_transition) * lex_gen_prob
    score = np.log(a) + np.log(b)
    score = np.exp(score)
    return score, best_prev_tag_id


# return list of tag names representing most probable tag sequence
def getTagSequence(bptr_matrix, score_matrix):
    # last tag = index of max value in last column of scoreMatrix
    # walk backwards through bptr_matrix starting at [last_tag, last_col], prepending each bptr to the sequence
    sentence_len = np.size(score_matrix[0])
    last_tag = np.argmax(score_matrix[:, sentence_len - 1])  # index of largest value in last col
    tag_sequence = []

    col = sentence_len - 1  # start at last col
    row = last_tag
    while col >= 0:
        tag_sequence.append(idToTagName(row))
        bptr_matrix = bptr_matrix.astype("int")
        row = bptr_matrix[row, col]
        col -= 1

    tag_sequence.reverse()

    return tag_sequence

def writeOutputToFile(fileName, strToWrite):
    with open(fileName, 'w') as the_file:
        the_file.write(strToWrite)


def convertToSubmissionOutput(predicted_tags):
    resultMap = defaultdict(list)
    i = 0
    while (i < len(predicted_tags)):
        # print (predicted_tags[i][0])
        if predicted_tags[i][0] == "O":
            i += 1
            continue
        tag = predicted_tags[i][0]
        startIndex = i
        while (i + 1 < len(predicted_tags) and predicted_tags[i + 1][0] == predicted_tags[i][0]):
            i += 1
        endIndex = i
        resultMap[tag].append((startIndex, endIndex))
        i += 1
    return resultMap


# return:
# r_LOC, r_PER, r_ORG, r_MISC, r_microaverage, r_macroaverage,
#   p_LOC, p_PER, p_ORG, p_MISC, p_microaverage, p_macroaverage,
#   f_measure
#
# prediction_list: list of (tag, index) tuples for each line
def calculateDevMetrics(prediction_list):
    gold_standard_lines = read_tag_lines("validation.txt")
    gs_labels_list = [tags for lines in gold_standard_lines for tags in lines.split()]
    # PM[i, j] = number of trials where system predicted i, gold standard = j
    # entities mapped to indices in following order: LOC, PER, ORG, MISC, O
    pm = np.zeros((5, 5))
    for (tag, index) in prediction_list:
        i = np.argmax(np.array([isLOC(tag), isPER(tag), isORG(tag), isMISC(tag), 0.5]))
        j = np.argmax(np.array([isLOC(gs_labels_list[index]), isPER(gs_labels_list[index]),
                                isORG(gs_labels_list[index]), isMISC(gs_labels_list[index]), 0.5]))
        pm[i, j] += 1

    r_LOC = pm[0, 0] / np.sum(pm[:, 0])
    r_PER = pm[1, 1] / np.sum(pm[:, 1])
    r_ORG = pm[2, 2] / np.sum(pm[:, 2])
    r_MISC = pm[3, 3] / np.sum(pm[:, 3])
    r_microaverage = (pm[0, 0] + pm[1, 1] + pm[2, 2] + pm[3, 3]) / np.sum(pm[:, :4])
    r_macroaverage = (r_LOC + r_PER + r_ORG + r_MISC) / 4

    p_LOC = pm[0, 0] / np.sum(pm[0, :])
    p_PER = pm[1, 1] / np.sum(pm[1, :])
    p_ORG = pm[2, 2] / np.sum(pm[2, :])
    p_MISC = pm[3, 3] / np.sum(pm[3, :])
    p_microaverage = (pm[0, 0] + pm[1, 1] + pm[2, 2] + pm[3, 3]) / np.sum(pm[:4, :])
    p_macroaverage = (p_LOC + p_PER + p_ORG + p_MISC) / 4

    f_measure = 2 * p_microaverage * r_microaverage / (p_microaverage + r_microaverage)

    return (r_LOC, r_PER, r_ORG, r_MISC, r_microaverage, r_macroaverage,
            p_LOC, p_PER, p_ORG, p_MISC, p_microaverage, p_macroaverage, f_measure)


if __name__ == "__main__":
    wordlines = read_token_lines("train90.txt")
    taglines = read_tag_lines("train90.txt")
    tokenmap = token_generation(wordlines)
    baseline_matrix = getbaseline_matrix(tokenmap, wordlines, taglines)
    #print (taglines)
    bigramCountMatrix = createUnsmoothedTagBigramCounts(taglines)
    bigramProbMatrix = getBigramProbsFromCounts(bigramCountMatrix)
    prediction = read_prediction_lines("validation.txt")
    #prediction = "\t".jlineoin(prediction)
    finalTags = []
    index = 0
    for line in prediction:
        #print (line)
        result = runViterbi(bigramProbMatrix, baseline_matrix, tokenmap, line)
        for tag in result:
            finalTags.append((getTag(tag), index))
            index += 1

    metrics = calculateDevMetrics(finalTags)
    print("Dev metrics: %s" % (metrics,))
    finalResult = convertToSubmissionOutput(finalTags)
    #print(finalResult)

    string = "Type,Prediction\n"
    for k, v in finalResult.items():
        # print (k)
        # print (v)
        string += k
        string += ","
        for tuple in v:
            string += str(tuple[0])
            string += "-"
            string += str(tuple[1])
            string += " "
        string += "\n"
    print (string)
    # writeToCSVFile(string)
    writeOutputToFile('hmm-output.csv', string)
