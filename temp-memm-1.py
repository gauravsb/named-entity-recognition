import numpy as np
import os
import json
import math
import csv
import collections
from collections import defaultdict
import nltk
from nltk.classify import MaxentClassifier

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

def read_pos_lines(filename):
    lines = read_all_lines(filename)
    desired_lines = lines[1::3]
    return desired_lines


def read_token_lines(filename):
    lines = read_all_lines(filename)
    desired_lines = lines[0::3]
    return desired_lines


def read_prediction_lines(filename):
    lines = read_all_lines(filename)
    desired_lines = lines[0::3]
    return desired_lines

def read_pos_prediction_lines(filename):
    lines = read_all_lines(filename)
    desired_lines = lines[1::3]
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

def generateTrainingFeatures(wordSentences, posSentences, bioSentences):
    train = []
    #TODO: previousAndNextLength = 1
    # label of previous will also be a part of feature
    for i in range(len(wordSentences)):
        wordList = wordSentences[i].split("\t")
        posList = posSentences[i].split("\t")
        bioList = bioSentences[i].split("\t")
        #print(wordList)
        #print(posList)
        #print(bioList)
        features = {}
        for j in range(len(wordList)):
            features["c"] = wordList[j][0].isupper()
            features["w"] = wordList[j]
            features["p"] = posList[j]
            # TODO: add numeric feature
            if j == 0:
                features["w-1"] = "<s>"
                features["p-1"] = "<s>"
                features["b-1"] = "<s>"
            else:
                features["w-1"] = wordList[j - 1]
                features["p-1"] = posList[j - 1]
                features["b-1"] = bioList[j - 1]
            if j == len(wordList)-1:
                features["w+1"] = "<e>"
                features["p+1"] = "<e>"
            else:
                features["w+1"] = wordList[j + 1]
                features["p+1"] = posList[j + 1]
            #print(features, bioList[j])
            train.append((features,bioList[j]))
            features = {}
    return train

def classify(probabilityDistribution):
    bioList = ["O", "B-PER","I-PER","B-LOC","I-LOC","B-ORG","I-ORG","B-MISC","I-MISC"]
    #resultProbabilityList = []
    max = -1
    maxTag = "O"
    for bioTag in bioList:
        probabilityOfABioTag = probabilityDistribution.prob(bioTag)
        if probabilityOfABioTag > max:
            max = probabilityOfABioTag
            maxTag = bioTag
        #print(bioTag, probabilityOfABioTag)
        #resultProbabilityList.append(probabilityOfABioTag)
    return maxTag


def generatePredictionFeatures(predictionSentence, predictPosSentence, index, bioPredictionResult):
    features = {}
    features["c"] = predictionSentence[index][0].isupper()
    features["w"] = predictionSentence[index]
    features["p"] = predictPosSentence[index]
    # TODO: add numeric feature
    if index == 0:
        features["w-1"] = "<s>"
        features["p-1"] = "<s>"
        features["b-1"] = "<s>"
    else:
        features["w-1"] = predictionSentence[index - 1]
        features["p-1"] = predictPosSentence[index - 1]
        features["b-1"] = bioPredictionResult[index - 1]
    if index == len(predictionSentence) - 1:
        features["w+1"] = "<e>"
        features["p+1"] = "<e>"
    else:
        features["w+1"] = predictionSentence[index + 1]
        features["p+1"] = predictPosSentence[index + 1]
    #print(features)
    return features

def predict(predictionSentence, predictPosSentence):
    #probMatrix = np.zeros([1,1])
    #for i in range(len(predictionSentence)):
    bioPredictionResult = []
    #finalResult = []
    for i in range(len(predictionSentence)):
        features = generatePredictionFeatures(predictionSentence, predictPosSentence, i, bioPredictionResult)
        probabilityDistribution = maxent_classifier.prob_classify(features)
        tagClassification = classify(probabilityDistribution)
        #print(tagClassification)
        bioPredictionResult.append(tagClassification)
        #finalResult.append(getTag(tagClassification))
    return bioPredictionResult

if __name__ == "__main__":
    wordLines = read_token_lines("train.txt")
    tagLines = read_tag_lines("train.txt")
    posLines = read_pos_lines("train.txt")
    train = generateTrainingFeatures(wordLines, posLines, tagLines)
    #print(train)
    maxent_classifier = MaxentClassifier.train(train, max_iter = 10)
    '''
    test = []
    features={}
    features["w"] = "CRICKET"
    features["p"] = "NNP"
    features["w-1"] = "<s>"
    features["p-1"] = "<s>"
    test.append((features, "O"))
    #print(maxent_classifier.prob_classify(({"word" : "CRICKET","pos" : "NNP"},"O")))
    probabilityDistribution = maxent_classifier.prob_classify(features)
    #print(maxent_classifier.prob_classify(features))
    classify(probabilityDistribution)
    #maxent_classifier.show_most_informative_features(10)

    #tokenmap = token_generation(wordlines)

'''
    prediction = read_prediction_lines("test.txt")
    posPrediction = read_pos_prediction_lines("test.txt")
    #prediction = "\t".join(prediction)
    finalTags = []
    index = 0
    for i in range(len(prediction)):
        #print (line)
        predSentence = prediction[i].split('\t')
        posPredSentence = posPrediction[i].split('\t')
        result = predict(predSentence, posPredSentence)
        for tag in result:
            finalTags.append((getTag(tag), index))
            index += 1

    print(finalTags)
    finalResult = convertToSubmissionOutput(finalTags)
    print(finalResult)

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
    writeOutputToFile('memm-output.csv', string)
