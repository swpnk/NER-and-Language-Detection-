import spotlight
import re
import json
from spotlight import annotate
import csv
import pytesseract
import cv2
import numpy as np
import os
from pytesseract import Output
import nltk
from nltk import word_tokenize
from nltk.util import ngrams

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"  # Path to the tesseract-OCR
config = 'tessdata-dir "tessdata" -l eng --oem 2 --psm 3'
words = []
entities = []
list_files = "../Data/Images/TickBox_1/Tickbox_1_1.png"
im = cv2.imread(list_files, 0)
# th3 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,3)
# kernel = np.ones((3,3), np.uint8)
# closing = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
strings = pytesseract.image_to_string(im, config=config)
token = nltk.word_tokenize(strings)

for j in range(len(token)):
    try:
        only_place_filter = {'policy':"whitelist",'types':"DBpedia:Location, DBpedia:Organization",'coreferenceResolution':False}
        annotations = spotlight.annotate('http://15.206.75.50/rest/annotate','Microsoft',confidence=0.0, support=0)
        split_annotations = annotations[0]['types'].split(",")
        print(split_annotations)
        patterns = ['DBpedia']
        for x in split_annotations:
            for pattern in patterns:
                if re.search(pattern, x):
                    # print(x.split(":")[1])
                    entities.append((x.split(":")[1]))
    except:
        pass

print(entities)
