import os
from azure_api import read_results
from tk_multi_pdf import get_jsons
import cv2
from polyglot.text import Text
import gc
import multiprocessing
from functools import partial
import re


def named_entity(pdf_file_name):
    pdf_name = pdf_file_name.split("/")[-1].split(".pdf")[0]
    print(pdf_name)
    data_folder = "C:/Users/advai/PycharmProjects/Data/"
    if not os.path.exists(os.path.join(data_folder + "Jsons/",
                                       pdf_name + "/")):  # Check if the directory exists
        os.makedirs(os.path.join(data_folder + "Jsons/", pdf_name + "/"))
    json_path = os.path.join(data_folder + "Jsons/", pdf_name + "/")
    # print(len(os.listdir(json_path)))
    if not len(os.listdir(json_path)):
        print("here")
        get_jsons(pdf_file_name)
    words, sentences = read_results(pdf_name)
    print(len(sentences))
    with multiprocessing.Pool(processes=4) as pool:
        pdf_name_file = pdf_name
        func = partial(named_entity_page, pdf_name_file, sentences)
        pool.map(func, list(range(len(sentences))))
        pool.close()
        pool.join()


def named_entity_page(pdf_file_name, sentences, number):
    print(number)
    output_folder = "C:/Users/advai/PycharmProjects/output/NER_Detection"

    if not os.path.exists(os.path.join(output_folder,
                                       pdf_file_name + "/" + "Location_Files" + "/")):  # Check if the directory exists
        os.makedirs(os.path.join(output_folder, pdf_file_name + "/" + "Location_Files" + "/"))

    if not os.path.exists(os.path.join(output_folder,
                                       pdf_file_name + "/" + "Name_Files" + "/")):  # Check if the directory exists
        os.makedirs(os.path.join(output_folder, pdf_file_name + "/" + "Name_Files" + "/"))
    person_names = []
    location_names = []
    negate_list = ["acupuncturist", "sugar", "beer", "pregnecy", "inhaler", "wheezing", "$", 'lymphocytepenia', 'nonsevere']
    for j in range(len(sentences[number][1])):
        print(sentences[number][1][j][0])
        if re.match('[a-zA-Z]', sentences[number][1][j][0]):
            text = Text(sentences[number][1][j][0])
            try:
                for entity in text.entities:
                    # print("here")
                    tag = entity.tag
                    name = ' '.join(entity._collection)
                    if len(name) < 2:
                        continue
                    if tag == 'I-PER':
                        if str(name).lower() in negate_list:
                            continue
                        person_names.append((name, sentences[number][1][j][1]))
                    if tag == 'I-LOC':
                        location_names.append((name, sentences[number][1][j][1]))
                        # print(location_names)
                print(person_names)

            except:
                continue
        else:
            continue

    img_name = cv2.imread("../Data/Images/{}/{}_{}.png".format(pdf_file_name, pdf_file_name, str(number + 1)))

    for j in range(len(person_names)):
        print("{}_names".format(number + 1))
        bbox_names = (min(person_names[j][1][0] * 200, person_names[j][1][6] * 200),
                      min(person_names[j][1][1] * 200, person_names[j][1][3] * 200),
                      max(person_names[j][1][2] * 200, person_names[j][1][4] * 200), max(
            person_names[j][1][5] * 200, person_names[j][1][7] * 200))
        cv2.rectangle(img_name, (int(bbox_names[0]), int(bbox_names[1])), (int(bbox_names[2]), int(bbox_names[3])),
                      (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_folder,
                                 pdf_file_name + "/" + "Name_Files" + "/" + "{}_{}.png".format(pdf_file_name,
                                                                                               str(number + 1))),
                    img=img_name)
        gc.collect()

    img = cv2.imread("../Data/Images/{}/{}_{}.png".format(pdf_file_name, pdf_file_name, str(number + 1)))

    for k in range(len(location_names)):
        print("{}_locations".format(number + 1))
        bbox_locations = (min(location_names[k][1][0] * 200, location_names[k][1][6] * 200),
                          min(location_names[k][1][1] * 200, location_names[k][1][3] * 200),
                          max(location_names[k][1][2] * 200, location_names[k][1][4] * 200), max(
            location_names[k][1][5] * 200, location_names[k][1][7] * 200))
        cv2.rectangle(img, (int(bbox_locations[0]), int(bbox_locations[1])),
                      (int(bbox_locations[2]), int(bbox_locations[3])), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(output_folder,
                                 pdf_file_name + "/" + "Location_Files" + "/" + "{}_{}.png".format(pdf_file_name,
                                                                                                   str(number + 1))),
                    img=img)
        gc.collect()

def named_entity_collection(input_path, folder_name, output_path):
    num_files = os.listdir(input_path)
    print(num_files)
    text_path = "../Output/Document_Summary/" + folder_name
    person_names = []
    location_names = []
    for i in range(len(num_files)):
        print(i)
        num_file = num_files[i].split(".pdf")[0]
        text_file = os.path.join(text_path, num_file + ".txt")
        pdf = open(text_file, 'rb')
        lines = pdf.readlines()
        for j in range(len(lines)):
            text = Text(str(lines[j]))
            print(text)
            try:
                for entity in text.entities:
                    print("here")
                    tag = entity.tag
                    name = ' '.join(entity._collection)
                    if len(name) < 2:
                        continue
                    if tag == 'I-PER':
                        if str(name).lower() in negate_list:
                            continue
                        person_names.append(name)
                    if tag == 'I-LOC':
                        location_names.append(name)
                        # print(location_names)
            except:
                continue

if __name__ == "__main__":
    named_entity_collection("../Data/Pdfs/Covid_medicine","covid_medicine","../Output/")
