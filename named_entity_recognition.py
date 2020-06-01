'''
Named Entity Recognition is the task of naming entities in a document. The following script handles the task of recognizing the entities in a document and tagging the entities present in a pdf document. 
The input to the script is a pdf file
The output on execution of the script, generates html files for every page with tagged names to entities in a document
'''

##### Run the following commands to install the necessary packages
# Install poppler from the link: 'https://blog.alivate.com.au/poppler-windows/' and specify the path to poppler/bin folder in the PATH variable in system variables
# python -m spacy download en_core_web_sm #Download the english vocabulary for spacy operations ##Run the command on command prompt or the conda prompt

# pip install pillow
# pip install pytesseract
# pip install pdf2image
# pip install PyPDF2
# conda install -c conda-forge poppler
# pip install opencv-python

#Install tesseract-OCR for 32 or 64 bit edition from : https://digi.bib.uni-mannheim.de/tesseract/

#pip install spacy
#python -m spacy download en_core_web_sm

#Import the necessary packages
from PIL import Image #Pillow package for image processing
import pytesseract #Free source OCR library for python
import cv2 #Package for image processing and segmentation
import os #System operations package in python
from pdf2image import convert_from_path #Convert pdf files to images 
from PyPDF2 import PdfFileReader #Read a pdf file and get the number of pages in a pdf
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe" #Provide the path to the installed tesseract-OCR 
import spacy #Spacy package in python for text analysis and processing
from spacy import displacy #Dispacy module from spacy helps to visualize named entities
# import nltk
# from nltk.tokenize import word_tokenize


def read_pdf_to_image(filepath):
    #Converts the pdf file to a array of images for all the pages
    reader = PdfFileReader(open(filepath, "rb")) #open file in read mode
    num_of_pages = reader.getNumPages() #get number of pages in the file
    print(f"Number of pages = {num_of_pages}")
    images_from_path = convert_from_path(filepath, last_page=num_of_pages, first_page =0) #Convert the pdf to a image array for all the pages
    return images_from_path

def get_text(page):
    #Uses tesseractOCR to convert image contours to text and detect the language using LangDetect Module
    text = pytesseract.image_to_string(page) #use the py-tesseract OCR to get the text from images
    text = text.lower() #Lowercase all the text for better language prediction
    return text


def named_entity_recognition(input_path):
    nlp = spacy.load("en_core_web_sm") #Loads the corpus of entities linked to the tokens in a sentence
    file_path = input_path #source path for the pdf
    csv_name = file_path.split("/")[-1][:-4]
    out_path = 'C:/Users/advai/PycharmProjects/Output/NER_Detection/' #destination path for output html files
    if out_path.split()[-1]=='/':
                out_path = out_path+"/{}/".format(csv_name)
    else:
                out_path = out_path+"/{}/".format(csv_name)
    images_from_path = read_pdf_to_image(file_path) #Call the read_pdf_to_image function to return an array of images for all the pages
    for i, page in enumerate(images_from_path): #Run the user defined functions over all the pages and save into separate jpg files
        text = get_text(page) #Call the contour_detection function to detect contours in the image and return the modified array
        text = nlp(text) #Use the text enclosed in every page in the nlp dict object type
        print(text)
        html_file = displacy.render(text, jupyter=False, style='ent') #Use displacy to save the named entities for tokens in an image and visualize it
        if not os.path.exists(out_path): #Check if the directory exists
            os.makedirs(out_path) #Create directory with the destination path

        f = open(out_path+"page_{}_named_entity_recognition.html".format(i),'w+') #Save the file to an html page for every page in the pdf
        f.write(html_file)
        f.close()
        for ent in text.ents:
            if out_path.split()[-1]=='/':
                    dest_path = out_path+'page_{}/{}/'.format(i,ent.label_) #Destination path for the sentences based on the output label that they are mapped to. The 2 variables appended to the path are, 1. The csv_file name to create a folder for every csv_file ,and 2. The output label that a sentence is classified into.
            else:
                    dest_path = out_path+'/page_{}/{}/'.format(i,ent.label_)

            if not os.path.exists(dest_path): #Check if the directory exists
                    os.makedirs(dest_path) #Create directory with the destination path

            try:
                    with open(dest_path+"sentences.txt", "a+") as file: #If already a sentences.txt file exists, replace it by the new predictions
                            file.write(ent.text) #Write the output sentences into a txt file in the directories that they belong to
                            file.write("\n") #Separate every sentence with a new line
            except FileNotFoundError: #If there is no file named sentences.txt in the folder, it creates a new txt file and appends the sentences to the file
                    with open(dest_path+"sentences.txt", "a+") as file: #Create a new file called sentences.txt in the folder
                            file.write(ent.text) #Write the output sentences into a txt file in the directories that they belong to
                            file.write("\n") #Separate every sentence with a new line


if __name__ == "__main__":  #main function only called when the script is running and not while importing modules
    named_entity_recognition("C:/Users/advai/Downloads/TickBox_1.pdf")
