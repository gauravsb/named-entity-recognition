import numpy as np
import os
import json
import math
import csv
import collections

def read_file(filename):
    stringInFile = ""
    with open(filename, "r", encoding="utf8") as fileObject:
        stringInFile = fileObject.read()
    return stringInFile


def writeToCSVFile(rowList):
    with open('output.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id','Prediction'])
        for string in rowList:
            writer.writerow(string)


def preprocess(some_text):
    some_text = "<s> " + some_text
    some_text = some_text.replace(" ’", "’")
    some_text = some_text.replace("’ ", "’")
    some_text = some_text.replace("\n", " ")
    some_text = some_text.replace(" . ", " . </s> <s> ")
    some_text = some_text[:-5]
    return some_text


if __name__ == "__main__":
    # read and parse the train file
    fileData = read_file("train.txt")
    print(fileData)

    # TODO: Add relevant preprocessing
    preprocess(fileData)
