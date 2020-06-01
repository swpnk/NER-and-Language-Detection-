'''
Language detection is the task of identifying and detecting the language of text in a document.
The input to the script is a pdf file
The output on execution of the script, generates images per page with text contours detected in a bounding box and the language in the text annotated to it
'''

##### Run the following commands to install the necessary packages
# Install poppler from the link: 'https://blog.alivate.com.au/poppler-windows/' and specify the path to poppler bin folder in the PATH variable in system variables

# pip install pillow
# pip install pytesseract
# pip install pdf2image
# pip install PyPDF2
# conda install -c conda-forge poppler
# pip install langdetect 
# pip install opencv-python

#Install tesseract-OCR for 32 or 64 bit edition from : https://digi.bib.uni-mannheim.de/tesseract/
#TODO: Higlight the languages which are sparse.


from PIL import Image #Pillow package for image processing
import pytesseract #Free source OCR library for python
import cv2 
import os
from pdf2image import convert_from_path #Convert pdf files to images 
from PyPDF2 import PdfFileReader #Read a pdf file and get the number of pages in a pdf
import langdetect #Language detection for python
import numpy as np
from collections import defaultdict
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe" #Provide the path to the installed tesseract-OCR 
# from pytesseract import Output
import gc
import multiprocessing
from functools import partial
from azure_api import read_results

def read_pdf_to_image(filepath):
    #Converts the pdf file to a array of images for all the pages
    reader = PdfFileReader(open(filepath, "rb")) #open file in read mode
    num_of_pages = reader.getNumPages() #get number of pages in the file
    print(f"Number of pages = {num_of_pages}")
    images_from_path = convert_from_path(filepath, last_page=num_of_pages, first_page =0) #Convert the pdf to a image array for all the pages 
    return images_from_path

def language_detect(contours,mask,open_cv_image,page):
    #Uses tesseractOCR to convert image contours to text and detect the language using LangDetect Module
    text_and_language = defaultdict(list) #Dictionary containing contours of text and corresponding language of the text
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx]) #Get the bbox coordinates of the contour
        mask[y:y+h, x:x+w] = 0 #Mask out the values containing the contour in it
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1) #Draw contours on the original image
        # gray_contour = cv2.cvtColor(open_cv_image[y:y+h-1,x+5:x+w-5], cv2.COLOR_BGR2GRAY) #Convert the array to grayscale
        # edges = cv2.Canny(open_cv_image,threshold1 = 100,threshold2 = 200,apertureSize = 3)
        # lines = cv2.HoughLinesP(edges,rho=1,theta = np.pi,threshold=100,minLineLength=100,maxLineGap=10)
        # if lines is not None:
        #     for line in get_lines(lines):
        #         leftx, boty, rightx, topy = line
        #         cv2.line(open_cv_image[y:y+h-1,x:x+w-1], (leftx, boty), (rightx,topy), (0,0, 255), 2)

        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h) 

        # if r > 0.45 and w > 8 and h > 8:
        #     cv2.rectangle(open_cv_image, (x, y), (x+w-1, y+h-1), (0, 255, 0), 3) #Draw a rectangle on the text using cv2

        text = pytesseract.image_to_string(page.crop((x-20, y-20, x+w+20, y+h+20))) #use the py-tesseract OCR to get the text from images
        text = text.lower() #Lowercase all the text for better language prediction
        
        try:
            language = langdetect.detect(text) #Detect the best suitable text language in the pdf file
            # cv2.putText(open_cv_image,str(language), (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2) #Print on the screen, the most suitable language of the text in the contour
        except:
            language = "error" #If the langdetect module does not return a language for the identified text. For eg: '.','222', etc
        text_and_language[language].append(text)
    # print(text_and_language)
    return open_cv_image,text_and_language

# def get_lines(lines_in):
#     #Get lines in a document file to apply hough transform and separate the contours and predict language on the text
#     if cv2.__version__ < '3.0':
#         return lines_in[0]
#     return [l[0] for l in lines_in] 

# def split(contour):
#     #Splits tables and column separated text contours in a image document
#     new_contour = []
#     toAdd = []
#     sCurrent = 0.0
#     sPrevious = 0.0
#     v = 0.0
#     dPrevious = 1

#     for points in range(0,len(contour)):
#         #
#         currentX = float(contour[points][0])
#         currentY = float(contour[points][1])
#         if points != 0:
#             v = math.sqrt(math.pow(float(toAdd[0][0])-currentX,2) + math.pow(float(toAdd[0][1])-currentY,2)) 
#             dPrevious = len(toAdd)
#         sCurrent = v/dPrevious
#         if sCurrent < 0.85*sPrevious:
#             new.append(toAdd)
#             toAdd = []
#         toAdd.append(contour[points])
#         sPrevious = sCurrent
#     new_contour.append(toAdd)
#     return new_contour

def contour_detection(page):
    #Detect contours in a image file and draw bounding boxes on them
    open_cv_image = np.array(page) #Get the PIL image file as a numpy array
    open_cv_image = open_cv_image[:, :, ::-1].copy() #Convert RGB values to BGR
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY) #Convert the array to grayscale
    kernel = np.ones((10,10), np.uint8) #Define a kernel for morph function by cv2
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel) #Morph the grayscale image
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #bw gives the gradient filtered image converted to black and white
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))#Define a rectangular kernel for morph function by cv2
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)#Morph the black and white image
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Find contours in the black and white image
    mask = np.zeros(bw.shape, dtype=np.uint8)
    open_cv_image,text_and_language = language_detect(contours,mask,open_cv_image,page) #Call the language detect function to detect the language for the page
    return open_cv_image,text_and_language

def language_detection(input_path):
    file_path = input_path #source path for the pdf
    csv_name = file_path.split("/")[-1][:-4]
    out_path = 'C:/Users/advai/PycharmProjects/Output/Language_Detection/'  #destination path for output html files
    if out_path.split()[-1]=='/':
                out_path = out_path+"/{}/".format(csv_name)
    else:
                out_path = out_path+"/{}/".format(csv_name)

    if not os.path.exists(out_path): #Check if the directory exists
        os.makedirs(out_path) #Create directory with the output path

    images_from_path = read_pdf_to_image(file_path) #Call the read_pdf_to_image function to return an array of images for all the pages
    print(len(images_from_path))
    page_list = [] #Run the user defined functions over all the pages and save into separate jpg files
    with multiprocessing.Pool(processes=4) as pool:
        func = partial(get_contours, input_path, images_from_path, page_list)
        pool.map(func, list(range(len(images_from_path))))
        pool.close()
        pool.join()

def get_contours(input_path, page, page_list, number):
    file_path = input_path  # source path for the pdf
    csv_name = file_path.split("/")[-1][:-4]
    out_path = 'C:/Users/advai/PycharmProjects/Output/Language_Detection/'
    out_path = out_path + "/{}/".format(csv_name)
    open_cv_image,text_and_language = contour_detection(page[number]) #Call the contour_detection function to detect contours in the image and return the modified array
    page_list_temp = []
    for key, value in text_and_language.items():
        page_list_temp.append((key, len([item for item in value if item])))
    page_list.append(sorted(page_list_temp, key=lambda x: x[1], reverse=True))
    cv2.imwrite(out_path+"/page_{}_language_detection.png".format(number),open_cv_image)
    for language in text_and_language.keys():
        if out_path.split()[-1]=='/':
            dest_path = out_path+'page_{}/{}/'.format(number,language) #Destination path for the sentences based on the output label that they are mapped to. The 2 variables appended to the path are, 1. The csv_file name to create a folder for every csv_file ,and 2. The output label that a sentence is classified into.
        else:
            dest_path = out_path+'/page_{}/{}/'.format(number,language)

        if not os.path.exists(dest_path): #Check if the directory exists
            os.makedirs(dest_path) #Create directory with the destination path
        for paragraph in text_and_language[language]:
            try:
                with open(dest_path+"sentences.txt", "a+") as file: #If already a sentences.txt file exists, replace it by the new predictions
                    file.write(paragraph) #Write the output sentences into a txt file in the directories that they belong to
                    file.write("\n") #Separate every sentence with a new line
            except FileNotFoundError: #If there is no file named sentences.txt in the folder, it creates a new txt file and appends the sentences to the file
                with open(dest_path+"sentences.txt", "a+") as file: #Create a new file called sentences.txt in the folder
                    file.write(paragraph) #Write the output sentences into a txt file in the directories that they belong to
                    file.write("\n") #Separate every sentence with a new line
    gc.collect()
    return page_list
    
            
        
        
if __name__ == "__main__":  #main function only called when the script is running and not while importing modules
    page_list = language_detection("C:/Users/advai/PyCharmProjects/Data/Pdfs/TickBox_1.pdf")
    pdf_name = "C:/Users/advai/PyCharmProjects/Data/Pdfs/TickBox_1.pdf".split("/")[-1].split(".pdf")[0]
    words,sentences = read_results(pdf_name)
    print(len(sentences))


"""
Languages supported by pytesseract

afr (Afrikaans), amh (Amharic), ara (Arabic), asm (Assamese), 
aze (Azerbaijani), aze_cyrl (Azerbaijani - Cyrilic), bel (Belarusian), ben (Bengali), 
bod (Tibetan), bos (Bosnian), bre (Breton), bul (Bulgarian), cat (Catalan; Valencian), 
ceb (Cebuano), ces (Czech), chi_sim (Chinese simplified), chi_tra (Chinese traditional), 
chr (Cherokee), cym (Welsh), dan (Danish), deu (German), 
dzo (Dzongkha), ell (Greek, Modern, 1453-), eng (English), enm (English, Middle, 1100-1500), 
epo (Esperanto), equ (Math / equation detection module), est (Estonian), eus (Basque), 
fas (Persian), fin (Finnish), fra (French), frk (Frankish), 
frm (French, Middle, ca.1400-1600), gle (Irish), glg (Galician), 
grc (Greek, Ancient, to 1453), guj (Gujarati), hat (Haitian; Haitian Creole), heb (Hebrew), 
hin (Hindi), hrv (Croatian), hun (Hungarian), iku (Inuktitut), ind (Indonesian), 
isl (Icelandic), ita (Italian), ita_old (Italian - Old), jav (Javanese), jpn (Japanese), 
kan (Kannada), kat (Georgian), kat_old (Georgian - Old), kaz (Kazakh), khm (Central Khmer), 
kir (Kirghiz; Kyrgyz), kmr (Kurdish Kurmanji), kor (Korean), kor_vert (Korean vertical), 
kur (Kurdish), lao (Lao), lat (Latin), lav (Latvian), lit (Lithuanian), ltz (Luxembourgish), 
mal (Malayalam), mar (Marathi), mkd (Macedonian), mlt (Maltese), 
mon (Mongolian), mri (Maori), msa (Malay), mya (Burmese), 
nep (Nepali), nld (Dutch; Flemish), nor (Norwegian), oci (Occitan post 1500), 
ori (Oriya), osd (Orientation and script detection module), pan (Panjabi; Punjabi), pol (Polish), 
por (Portuguese), pus (Pushto; Pashto), que (Quechua), ron (Romanian; Moldavian; Moldovan), 
rus (Russian), san (Sanskrit), sin (Sinhala; Sinhalese), slk (Slovak), 
slv (Slovenian), snd (Sindhi), spa (Spanish; Castilian), spa_old (Spanish; Castilian - Old), 
sqi (Albanian), srp (Serbian), srp_latn (Serbian - Latin), sun (Sundanese), 
swa (Swahili), swe (Swedish), syr (Syriac), tam (Tamil), tat (Tatar), 
tel (Telugu), tgk (Tajik), tgl (Tagalog), tha (Thai), tir (Tigrinya), 
ton (Tonga), tur (Turkish), uig (Uighur; Uyghur), ukr (Ukrainian), 
urd (Urdu), uzb (Uzbek), uzb_cyrl (Uzbek - Cyrilic), vie (Vietnamese), yid (Yiddish), yor (Yoruba)
"""