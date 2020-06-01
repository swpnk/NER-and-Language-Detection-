import os.path
import time
import requests
import json
import pandas as pd
from PyPDF2 import PdfFileReader, PdfFileWriter, PdfFileMerger
import csv
import cv2
import re
from polyglot.text import Text, Word
import spotlight
from math import ceil, floor
import shutil
import gc
import multiprocessing


def get_create_json_pdf(pdf_file_path):
    pdf_name = pdf_file_path.split("/")[-1].split(".pdf")[0]
    pdfs_folder = "C:/Users/advai/PycharmProjects/Data/Split_Pdfs/"
    data_folder = "C:/Users/advai/PycharmProjects/Data/"
    get_jsons(pdf_file_path)
    pdf_paths = sorted(os.listdir(pdfs_folder + pdf_name + "/"), key=len)
    print(pdf_paths)
    if not os.path.exists(os.path.join(data_folder + "Jsons/",
                                       pdf_name + "/")):  # Check if the directory exists
        os.makedirs(os.path.join(data_folder + "Jsons/", pdf_name + "/"))
    text_recognition_url = os.environ['COMPUTER_VISION_ENDPOINT_1'] + \
                           "vision/v2.1/read/core/asyncBatchAnalyze"
    api_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY_1']

    headers = {'Ocp-Apim-Subscription-Key': api_key,
               'Content-Type': 'application/octet-stream'}
    params = {'language': 'unk', 'detectOrientation': 'true'}
    if len(pdf_paths) == 1:
        with open(os.path.join(pdfs_folder, pdf_name + "/" + pdf_paths[0]), 'rb') as pdf_file:
            response = requests.post(text_recognition_url, headers=headers, data=pdf_file)
        analysis = {}
        poll = True
        while (poll):
            response_final = requests.get(
                response.headers["Operation-Location"], headers=headers)
            analysis = response_final.json()
            # print(analysis)
            with open(os.path.join(data_folder + "Jsons/", pdf_name + "/" + pdf_name + ".json"), 'w',
                      encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=4)
            time.sleep(1)
            if "recognitionResults" in analysis:
                poll = False
            if "status" in analysis and analysis['status'] == 'Failed':
                poll = False

        json_file = pd.read_json(os.path.join(data_folder + "Jsons/", pdf_name + "/" + pdf_name + ".json"))
        for i in range(len(json_file['recognitionResults'])):
            with open(os.path.join(data_folder + "Jsons/", pdf_name + "/" + pdf_name + "_" + str(i + 1) + ".json"), 'w',
                      encoding='utf-8') as f:
                json.dump(json_file['recognitionResults'][i], f, ensure_ascii=False, indent=4)
        os.remove(os.path.join(data_folder + "Jsons/", pdf_name + "/" + pdf_name + ".json"))
    else:
        for i in range(len(pdf_paths)):
            with open(os.path.join(pdfs_folder, pdf_name + "/" + pdf_paths[i]), 'rb') as pdf_file:
                response = requests.post(text_recognition_url, headers=headers, data=pdf_file)
            analysis = {}
            poll = True
            while (poll):
                response_final = requests.get(
                    response.headers["Operation-Location"], headers=headers)
                analysis = response_final.json()
                # print(analysis)
                with open(os.path.join(data_folder + "Jsons/", pdf_name + "/" + pdf_name + ".json"), 'w',
                          encoding='utf-8') as f:
                    json.dump(analysis, f, ensure_ascii=False, indent=4)
                time.sleep(1)
                if "recognitionResults" in analysis:
                    poll = False
                if "status" in analysis and analysis['status'] == 'Failed':
                    poll = False

            json_file = pd.read_json(os.path.join(data_folder + "Jsons/", pdf_name + "/" + pdf_name + ".json"))
            for j in range(len(json_file['recognitionResults'])):
                with open(os.path.join(data_folder + "Jsons/",
                                       pdf_name + "/" + pdf_name + "_" + str((j + 1) + (8 * i)) + ".json"),
                          'w',
                          encoding='utf-8') as f:
                    json.dump(json_file['recognitionResults'][j], f, ensure_ascii=False, indent=4)
            os.remove(os.path.join(data_folder + "Jsons/", pdf_name + "/" + pdf_name + ".json"))


def get_jsons(pdf_file_path):
    pdf_name = pdf_file_path.split("/")[-1].split(".pdf")[0]
    print(pdf_name)
    pdfs_folder = "C:/Users/advai/PycharmProjects/Data/Split_Pdfs/"
    data_folder = "C:/Users/advai/PycharmProjects/Data/"

    if not os.path.exists(os.path.join(pdfs_folder, pdf_name + "/")):  # Check if the directory exists
        os.makedirs(os.path.join(pdfs_folder, pdf_name + "/"))

    if ceil(os.stat(pdf_file_path).st_size / (1024 * 1024)) < 23:  # if the total size of the pdf is less than 23MB
        shutil.copyfile(pdf_file_path, os.path.join(pdfs_folder + pdf_name + "/" + '{}_1.pdf'.format(pdf_name)))
        return

    pdf = PdfFileReader(pdf_file_path)
    total_pages = pdf.getNumPages()
    for pdfs in range(ceil(pdf.getNumPages() / 8)):
        pdf_writer = PdfFileWriter()
        try:
            for page in range(8):
                # print(page)
                pdf_writer.addPage(pdf.getPage((page + (8 * pdfs))))
        except:
            output_filename = os.path.join(pdfs_folder + pdf_name + "/" + '{}_{}.pdf'.format(pdf_name, pdfs + 1))
            with open(output_filename, 'wb') as out:
                pdf_writer.write(out)
                # print('Created: {}'.format(output_filename))
        if page == 7:
            output_filename = os.path.join(pdfs_folder + pdf_name + "/" + '{}_{}.pdf'.format(pdf_name, pdfs + 1))
            with open(output_filename, 'wb') as out:
                pdf_writer.write(out)
                # print('Created: {}'.format(output_filename))


def get_create_json_images(pdf_file_name):
    pdf_name = pdf_file_name
    images_folder = "C:/Users/advai/PycharmProjects/Data/Images/"
    data_folder = "C:/Users/advai/PycharmProjects/Data/"
    image_path = sorted(os.listdir(images_folder + pdf_name + "/"), key=len)
    if not os.path.exists(os.path.join(data_folder + "Jsons/",
                                       pdf_name + "/")):  # Check if the directory exists
        os.makedirs(os.path.join(data_folder + "Jsons/", pdf_name + "/"))
    text_recognition_url = os.environ['COMPUTER_VISION_ENDPOINT'] + \
                           "vision/v2.1/read/core/asyncBatchAnalyze"
    api_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']

    headers = {'Ocp-Apim-Subscription-Key': api_key,
               'Content-Type': 'application/octet-stream'}
    params = {'language': 'unk', 'detectOrientation': 'true'}

    for i in range(len(image_path)):
        with open(images_folder + pdf_name + "/" + image_path[i], "rb") as pdf_file:
            response = requests.post(text_recognition_url, headers=headers, params=params, data=pdf_file)

        analysis = {}
        poll = True
        while (poll):
            response_final = requests.get(
                response.headers["Operation-Location"], headers=headers)
            analysis = response_final.json()
            # print(analysis)
            with open(os.path.join(data_folder + "Jsons/", pdf_name + "/" + pdf_name + "_" + str(i + 1) + ".json"), 'w',
                      encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=4)
            time.sleep(1)
            if "recognitionResults" in analysis:
                poll = False
            if "status" in analysis and analysis['status'] == 'Failed':
                poll = False


def read_results(pdf_file_name):
    word_coordinates = []
    sentence_coordinates = []
    data_folder = "C:/Users/advai/PycharmProjects/Data/"
    json_path = os.path.join(data_folder + "Jsons/", pdf_file_name + "/")
    Jsons = sorted(os.listdir(json_path), key=len)
    # print(Jsons)
    for i in range(len(Jsons)):
        results = pd.read_json(os.path.join(data_folder + "Jsons/", pdf_file_name + "/" + Jsons[i]))
        # print(results)
        # Sentences
        page_result = results['lines']
        words = []
        sentences = []
        for j in range(len(page_result)):
            sentences.append([page_result[j]['text'], page_result[j]['boundingBox']])
            # Words
            for k in range(len(page_result[j]['words'])):
                # print(len(page_result[j]['words']))
                words.append([page_result[j]['words'][k]['text'], page_result[j]['words'][k]['boundingBox']])
        word_coordinates.append([i + 1, words])
        sentence_coordinates.append([i + 1, sentences])
    return word_coordinates, sentence_coordinates


def bigram_coordinates(word_coordinates):
    word_coordinates_bi = []
    for i in range(len(word_coordinates)):
        word_coordinates_page = []
        # print(word_coordinates[i][1])
        for j in range(1, len(word_coordinates[i][1])):
            # print(word_coordinates[i][1][j][0])
            if abs(word_coordinates[i][1][j - 1][1][1] - word_coordinates[i][1][j][1][1]) <= 0.02:
                # print(word_coordinates[i][1][j - 1][0])
                # print(word_coordinates[i][1][j][0])
                word_coordinates_page.append([word_coordinates[i][1][j - 1][0] + " " + word_coordinates[i][1][j][0],word_coordinates[i][1][j - 1][1][0:4]+word_coordinates[i][1][j][1][4:8]])
        word_coordinates_bi.append([i+1, word_coordinates_page])
    print(len(word_coordinates_bi))
    return word_coordinates_bi


# def highlight_images(pdf_file_name, database_name):
#     data_folder = "C:/Users/advai/PycharmProjects/Data/"
#     json_path = os.path.join(data_folder + "Jsons/", pdf_file_name + "/")
#     if not len(os.listdir(json_path)):
#         get_create_json_pdf(pdf_file_name)
#     else:
#         with open("../Data/Database/{}.csv".format(database_name), newline="") as f:
#             reader = csv.reader(f)
#             data = list(reader)
#             # print(len(data))
#     word_coordinates, sentence_coordinates = read_results(pdf_file_name)
#     unigram_cord = word_coordinates
#     bigram_cord = bigram_coordinates(word_coordinates)
#     total_dict = {}
#     for i in range(len(word_coordinates)):
#         total_dict[i+1] = unigram_cord[i][1] + bigram_cord[i][1]
#     # print(total_dict)
#     # print(len(total_dict))
#     # for i in range(len(data)):
#     #     found_terms = []
#     #     for j in range(len(word_coordinates)):
#     #         for k in range(len(word_coordinates[j][1])):
#     #             # print(word_coordinates[j][1][k][1])
#     #             if data[i][0].lower() in word_coordinates[j][1][k][0].lower():
#     #                 print(word_coordinates[j][1][k][0])
#     words_found_page = []
#     for i in range(1, len(total_dict)):
#         words_found = []
#         for j in range(len(data)):
#             for k in range(len(total_dict[i])):
#                 substring = total_dict[i][k][0].lower()
#                 if data[j][0].lower() + "." == substring or \
#                         data[j][0].lower() + " " == substring or \
#                         data[j][0].lower() + ";" == substring or \
#                         data[j][0].lower() + "-" == substring or \
#                         data[j][0].lower() + "," == substring or \
#                         data[j][0].lower() + "(" == substring or \
#                         data[j][0].lower() + "" == substring or \
#                         data[j][0].lower() + "/" == substring:
#                     words_found.append([total_dict[i][k][0], total_dict[i][k][1]])
#         words_found_page.append([i, words_found])
#     print(words_found_page)
#
#     for i in range(len(words_found_page)):
#         print(len(words_found_page[i][1]))
#         if len(words_found_page[i][1]) != 0:
#             print(words_found_page[i][0])
#             img = cv2.imread("../Data/Images/{}/{}_{}.png".format(pdf_file_name, pdf_file_name, str(i+1)))
#             for j in range(len(words_found_page[i][1])):
#                 # print(i, words_found_page[i][1][j][1])
#                 bbox = (min(words_found_page[i][1][j][1][0] * 200, words_found_page[i][1][j][1][6] * 200),
#                         min(words_found_page[i][1][j][1][1] * 200, words_found_page[i][1][j][1][3] * 200),
#                         max(words_found_page[i][1][j][1][2] * 200, words_found_page[i][1][j][1][4] * 200), max(
#                     words_found_page[i][1][j][1][5] * 200, words_found_page[i][1][j][1][7] * 200))
#                 # print(bbox)
#                 print(bbox)
#                 cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
#
#                 if not os.path.exists(os.path.join(data_folder + "Highlighted_Pdfs/",
#                                                    pdf_file_name + "/" + database_name + "/")):  # Check if the directory exists
#                     os.makedirs(
#                         os.path.join(data_folder + "Highlighted_Pdfs/", pdf_file_name + "/" + database_name + "/"))
#             cv2.imwrite(os.path.join(data_folder + "Highlighted_Pdfs/",
#                                      pdf_file_name + "/" + database_name + "/" + "{}_{}.png".format(pdf_file_name,
#                                                                                                     str(i + 1))),
#                         img=img)
#         else:
#             continue

def highlight_images(pdf_file_name, database_name):
    data_folder = "C:/Users/advai/PycharmProjects/Data/"
    json_path = os.path.join(data_folder + "Jsons/", pdf_file_name + "/")
    if not len(os.listdir(json_path)):
        get_create_json_pdf(pdf_file_name)
    else:
        word_coordinates, sentence_coordinates = read_results(pdf_file_name)
        for database in range(len(database_name)):
            with open("../Data/Database/{}.csv".format(database_name[database]), newline="") as f:
                reader = csv.reader(f)
                data = list(reader)
                print(len(data))
        # for i in range(len(data)):
        #     found_terms = []
        #     for j in range(len(word_coordinates)):
        #         for k in range(len(word_coordinates[j][1])):
        #             # print(word_coordinates[j][1][k][1])
        #             if data[i][0].lower() in word_coordinates[j][1][k][0].lower():
        #                 print(word_coordinates[j][1][k][0])
            words_found_page = []
            for i in range(len(word_coordinates)):
                words_found = []
                for j in range(len(data)):
                    for k in range(len(word_coordinates[i][1])):
                        substring = word_coordinates[i][1][k][0].lower()
                        if data[j][0].lower() + "." == substring or \
                                data[j][0].lower() + " " == substring or \
                                data[j][0].lower() + ";" == substring or \
                                data[j][0].lower() + "-" == substring or \
                                data[j][0].lower() + "," == substring or \
                                data[j][0].lower() + "(" == substring or \
                                data[j][0].lower() + "" == substring or \
                                data[j][0].lower() + "/" == substring:
                            words_found.append([word_coordinates[i][1][k][0], word_coordinates[i][1][k][1]])
                words_found_page.append([i + 1, words_found])
            # print(words_found_page)
            if database == 0:
                for i in range(len(words_found_page)):
                    img = cv2.imread("../Data/Images/{}/{}_{}.png".format(pdf_file_name, pdf_file_name, str(i + 1)))
                    for j in range(len(words_found_page[i][1])):
                        print(i, words_found_page[i][1][j][1])
                        bbox = (min(words_found_page[i][1][j][1][0]*200, words_found_page[i][1][j][1][6]*200),
                                min(words_found_page[i][1][j][1][1]*200, words_found_page[i][1][j][1][3]*200),
                                max(words_found_page[i][1][j][1][2]*200, words_found_page[i][1][j][1][4]*200), max(
                            words_found_page[i][1][j][1][5]*200, words_found_page[i][1][j][1][7]*200))
                        # print(bbox)
                        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255*(i+1), 0), 2)

                        if not os.path.exists(os.path.join(data_folder + "Highlighted_Pdfs/",
                                                           pdf_file_name + "/")):  # Check if the directory exists
                            os.makedirs(os.path.join(data_folder + "Highlighted_Pdfs/", pdf_file_name + "/"))
                    cv2.imwrite(os.path.join(data_folder + "Highlighted_Pdfs/",
                                             pdf_file_name + "/" + "{}_{}.png".format(pdf_file_name, str(i + 1))),
                                img=img)
            elif database == 1:
                for i in range(len(words_found_page)):
                    img = cv2.imread("../Data/Highlighted_Pdfs/{}/{}_{}.png".format(pdf_file_name,pdf_file_name, str(i + 1)))
                    for j in range(len(words_found_page[i][1])):
                        print(i, words_found_page[i][1][j][1])
                        bbox = (min(words_found_page[i][1][j][1][0] * 200, words_found_page[i][1][j][1][6] * 200),
                                min(words_found_page[i][1][j][1][1] * 200, words_found_page[i][1][j][1][3] * 200),
                                max(words_found_page[i][1][j][1][2] * 200, words_found_page[i][1][j][1][4] * 200), max(
                            words_found_page[i][1][j][1][5] * 200, words_found_page[i][1][j][1][7] * 200))
                        # print(bbox)
                        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (255,0, 0), 2)

                        if not os.path.exists(os.path.join(data_folder + "Highlighted_Pdfs/",
                                                           pdf_file_name + "/" + "/")):  # Check if the directory exists
                            os.makedirs(
                                os.path.join(data_folder + "Highlighted_Pdfs/", pdf_file_name + "/"))
                    cv2.imwrite(os.path.join(data_folder + "Highlighted_Pdfs/",
                                             pdf_file_name +"/" + "{}_{}.png".format(pdf_file_name,str(i + 1))),img=img)


def named_entity(pdf_file_name):
    output_folder = "C:/Users/advai/PycharmProjects/output/NER_Detection"
    data_folder = "C:/Users/advai/PycharmProjects/Data/"
    if not os.path.exists(os.path.join(data_folder + "Jsons/",
                                       pdf_file_name + "/")):  # Check if the directory exists
        os.makedirs(os.path.join(data_folder + "Jsons/", pdf_file_name + "/"))
    json_path = os.path.join(data_folder + "Jsons/", pdf_file_name + "/")
    if not len(os.listdir(json_path)):
        get_create_json_images(pdf_file_name)
    words, sentences = read_results(pdf_file_name)
    for i in range(len(sentences)):
        person_names = []
        location_names = []
        negate_list = ["acupuncturist", "sugar", "beer", "pregnecy", "inhaler", "wheezing"]
        for j in range(len(sentences[i][1])):
            # print(sentences[i][1][j][0], sentences[i][1][j][1])
            text = Text(sentences[i][1][j][0])
            try:
                for entity in text.entities:
                    tag = entity.tag
                    name = ' '.join(entity._collection)
                    if len(name) < 2:
                        continue
                    if tag == 'I-PER':
                        if str(name).lower() in negate_list:
                            continue
                        person_names.append((name, sentences[i][1][j][1]))
                    if tag == 'I-LOC':
                        location_names.append((name, sentences[i][1][j][1]))
                        # print(location_names)
                # print(person_names)

            except:
                pass
        gc.collect()
        img = cv2.imread("../Data/Images/{}/{}_{}.png".format(pdf_file_name, pdf_file_name, str(i + 1)))

        for j in range(len(person_names)):
            print(person_names)
            bbox_names = (min(person_names[j][1][0] * 200, person_names[j][1][6] * 200),
                          min(person_names[j][1][1] * 200, person_names[j][1][3] * 200),
                          max(person_names[j][1][2] * 200, person_names[j][1][4] * 200), max(
                person_names[j][1][5] * 200, person_names[j][1][7] * 200))
            cv2.rectangle(img, (int(bbox_names[0]), int(bbox_names[1])), (int(bbox_names[2]), int(bbox_names[3])),
                          (255, 0, 0), 2)
            if not os.path.exists(os.path.join(output_folder,
                                               pdf_file_name + "/" + "Name_Files" + "/")):  # Check if the directory exists
                os.makedirs(os.path.join(output_folder, pdf_file_name + "/" + "Name_Files" + "/"))
            cv2.imwrite(os.path.join(output_folder,
                                     pdf_file_name + "/" + "Name_Files" + "/" + "{}_{}.png".format(pdf_file_name,
                                                                                                   str(i + 1))),
                        img=img)
            gc.collect()

        img = cv2.imread("../Data/Images/{}/{}_{}.png".format(pdf_file_name, pdf_file_name, str(i + 1)))

        for k in range(len(location_names)):
            bbox_locations = (min(location_names[k][1][0] * 200, location_names[k][1][6] * 200),
                              min(location_names[k][1][1] * 200, location_names[k][1][3] * 200),
                              max(location_names[k][1][2] * 200, location_names[k][1][4] * 200), max(
                location_names[k][1][5] * 200, location_names[k][1][7] * 200))
            print(location_names[k][0], bbox_locations)
            cv2.rectangle(img, (int(bbox_locations[0]), int(bbox_locations[1])),
                          (int(bbox_locations[2]), int(bbox_locations[3])), (0, 0, 255), 2)
            if not os.path.exists(os.path.join(output_folder,
                                               pdf_file_name + "/" + "Location_Files" + "/")):  # Check if the directory exists
                os.makedirs(os.path.join(output_folder, pdf_file_name + "/" + "Location_Files" + "/"))
            cv2.imwrite(os.path.join(output_folder,
                                     pdf_file_name + "/" + "Location_Files" + "/" + "{}_{}.png".format(pdf_file_name,
                                                                                                       str(i + 1))),
                        img=img)
            gc.collect()


def dbpedia_get(pdf_file_name):
    output_folder = "C:/Users/advai/PycharmProjects/output/NER_Detection"
    data_folder = "C:/Users/advai/PycharmProjects/Data/"
    json_path = os.path.join(data_folder + "Jsons/", pdf_file_name + "/")
    if not len(os.listdir(json_path)):
        get_create_json_images(pdf_file_name)
    words, sentences = read_results(pdf_file_name)
    only_place_filter = {'policy': "whitelist", 'types': "DBpedia:Location, DBpedia:Organization",
                         'coreferenceResolution': False}
    for i in range(len(sentences)):
        for j in range(len(sentences[i][1])):
            print(sentences[i][1][j][0])
            if len(sentences[i][1][j][0]) <= 2 or "no" in sentences[i][1][j][0].lower():
                continue
            else:
                try:
                    annotations = spotlight.annotate('http://15.206.75.50/rest/annotate',
                                                     '{}'.format(sentences[i][1][j][0]), confidence=0.0,
                                                     support=0, filters=only_place_filter)
                    split_annotations = annotations[0]['types'].split(",")
                    print(sentences[i][1][j][0], split_annotations)
                except:
                    pass


if __name__ == "__main__":
    highlight_images("Covid_Article_16", ["Diseases","Drugs"])
