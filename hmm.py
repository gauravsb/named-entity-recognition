import numpy as np
import os
import json
import math
import csv
import collections

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


def writeToCSVFile(rowList):
    with open('output.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id','Prediction'])
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


# returns matrix M such that M[i, j] = # of (t_i t_j) sequences
def createUnsmoothedTagBigramCounts(lines_of_tags):
    bigramCountMatrix = np.zeros([10, 10])

    for l in range(len(lines_of_tags)):
        line = lines_of_tags[l].split(' ')
        first_tag = tagNameToId(line[0])
        bigramCountMatrix[START, first_tag] += 1
        for i in range(1, len(l)):
            bigramCountMatrix[tagNameToId(l[i - 1]), tagNameToId(l[i])] += 1

    return bigramCountMatrix


def getBigramProbsFromCounts(bigramCountMatrix):
    bigramProbMatrix = bigramCountMatrix / bigramCountMatrix.sum(axis=1, keepdims=True)
    return bigramProbMatrix

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

def getlexical_generation_probs(baseline_matrix,word,tag,tokenToIdMap):
    tag_id=tagNameToId(tag)
    token_id=tokenToIdMap[word]
    tag_column=baseline_matrix[:,tag_id]
    tag_count=np.sum(tag_column)
    word_count=baseline_matrix[token_id][tag_id]
    return word_count/tag_count


# returns list of most probable tag sequence for given sentence
# transitionProbs: ngram probs for tags
def runViterbi(transitionProbs, lexGenProbs, tokenToIdMap, sentence):
    sentenceList = sentence.split(' ')
    numrows = 9
    numcols = len(sentenceList)
    scoreMatrix = np.zeros([numrows, numcols])
    bptrMatrix = np.zeros([numrows, numcols])

    # calculate first column separately because previous column is start
    # for first column: score = P(t | <s>) * P(w | t), bptr = 0
    for i in range(numrows):
        score = transitionProbs[START, i]




if __name__ == "__main__":
    # read and parse the train file
    fileData = read_file("train.txt")
    print(fileData)

    # TODO: Add relevant preprocessing
    wordlines=read_token_lines("train.txt")
    taglines=read_token_lines("train.txt")
    tokenmap=token_generation(wordlines)
    baseline_matrix=getbaseline_matrix(tokenmap,wordlines,taglines)
