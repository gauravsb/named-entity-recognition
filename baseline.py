import numpy as np
import os
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


def read_lines(filename, start, end):
    lines = read_all_lines(filename)
    desired_lines = lines[start::end]
    return desired_lines


def read_prediction_lines(filename):
    lines = read_all_lines(filename)
    desired_lines = lines[0::3]
    return desired_lines


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
        #print (tags[i])
        tag_id = tagNameToId(tags[i])
        #print (tag_id)
        token_id = tokenToIdMap[words[i]]
        baseline_matrix_counts[token_id][tag_id] += 1
        #if baseline_matrix_counts[token_id][tag_id] > 0:
        #print (baseline_matrix_counts[token_id][tag_id])
    #print (baseline_matrix_counts)

    return baseline_matrix_counts


def getBaselinePrediction(prediction_string, baseline_matrix_counts, tokenToIdMap):
    prediction_string = prediction_string.split("\t")
    # print (prediction_string)
    result_tags = []
    resultMap = {}
    index = 0
    tag = ""
    previousTag = None
    #currentTag = ""
    #startIndex = 0
    for token in prediction_string:
        if token in tokenToIdMap:
            token_id = tokenToIdMap[token]
            #print (baseline_matrix_counts[token_id])
            result = np.argmax((baseline_matrix_counts[token_id]))
            #print(result)
            #result_tags.append(idToTagName(result))
            #print (idToTagName(result))
            #print (getTag(idToTagName(result)))
            #result_tags.append(getTag(idToTagName(result)))
            tag = (getTag(idToTagName(result)), index)
        else:
            #result_tags.append((getTag(idToTagName(0)), index))
            tag = (getTag(idToTagName(0)), index)

        '''
        if previousTag == None:
            previousTag = tag
        if tag != "O":
            if tag == previousTag:
                resultMap[tag] =
            else :
        
        previousTag = tag
        '''

        result_tags.append(tag)
        index += 1
    #print (result_tags)
    return result_tags


def getlexical_generation_probs(baseline_matrix, word, tag, tokenToIdMap):
    tag_id = tagNameToId(tag)
    token_id = tokenToIdMap[word]
    tag_column = baseline_matrix[:, tag_id]
    tag_count = np.sum(tag_column)
    word_count = baseline_matrix[token_id][tag_id]
    return word_count / tag_count


# returns list of most probable tag sequence for given sentence
# transitionProbs: ngram probs for tags
def runViterbi(transitionProbs, baseline_matrix, tokenToIdMap, sentence):
    sentenceList = sentence.split(' ')
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
            score, bptr = calculateScoreAndBptr(prev_col, tokenToIdMap[sentenceList[j]], i)
            scoreMatrix[i, j] = score
            bptrMatrix[i, j] = bptr

    return getTagSequence(bptrMatrix)


# TODO
# return tuple of (score, bptr) for a given point in the viterbi matrix ([tag, word])
def calculateScoreAndBptr(prev_col_scores, word_id, tag_id):
    # score = max_i (prev_col_scores[i] * P(tag | t_i)) * P(word | tag)
    # bptr = i that gives max score
    return 0, 0  # placeholder


# TODO
# return list of tags representing most probable tag sequence
def getTagSequence(bptrMatrix, scoreMatrix):
    # last tag = index of max value in last column of scoreMatrix
    # walk backwards through bptrMatrix starting at [last_tag, last_col], prepending each bptr to the sequence
    return  # placeholder


def isPER(str):
    return str == "B-PER" or str == "I-PER"

def isLOC(str):
    return str == "B-LOC" or str == "I-LOC"

def isORG(str):
    return str == "B-ORG" or str == "I-ORG"

def isMISC(str):
    return str == "B-MISC" or str == "I-MISC"

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

def convertToSubmissionOutput(predicted_tags):
    resultMap = defaultdict(list)
    i = 0
    while(i < len(predicted_tags)):
        #print (predicted_tags[i][0])
        if predicted_tags[i][0] == "O":
            i += 1
            continue
        tag = predicted_tags[i][0]
        startIndex = i
        while (i+1 < len(predicted_tags) and predicted_tags[i+1][0] == predicted_tags[i][0]):
            i += 1
        endIndex = i
        resultMap[tag].append((startIndex, endIndex))
        i += 1
    return resultMap


def writeToCSVFile(rowList):
    with open('baseline-output.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Type','Prediction'])
        rowList = rowList.split("\n")
        for str in rowList:
            print (str)
            writer.writerow(str)


def writeOutputToFile(fileName, strToWrite):
    with open(fileName, 'w') as the_file:
        the_file.write(strToWrite)

if __name__ == "__main__":
    # read and parse the train file
    # fileData = read_file("train.txt")
    # print(fileData)

    wordlines = read_token_lines("train.txt")
    # print (wordlines)
    taglines = read_tag_lines("train.txt")
    # print (taglines)
    tokenmap = token_generation(wordlines)
    # print (tokenmap)
    baseline_matrix = getbaseline_matrix(tokenmap, wordlines, taglines)

    prediction = read_prediction_lines("test.txt")
    # print (prediction)
    prediction = "\t".join(prediction)

    baseline_predicted_tags = getBaselinePrediction(prediction, baseline_matrix, tokenmap)
    #print (baseline_predicted_tags)

    result = convertToSubmissionOutput(baseline_predicted_tags)
    #print (dict(result))
    string = "Type,Prediction\n"
    for k, v in result.items():
        #print (k)
        #print (v)
        string += k
        string += ","
        for tuple in v:
            string += str(tuple[0])
            string += "-"
            string += str(tuple[1])
            string += " "
        string += "\n"
    print (string)
    #writeToCSVFile(string)
    writeOutputToFile('baseline-output.csv', string)

