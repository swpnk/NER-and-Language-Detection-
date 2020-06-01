import requests
from pprint import pprint
import os
import json

subscription_key = "e2b8a179d6f74bd2b60239e04fa45d89"
endpoint = "https://canadacentral.api.cognitive.microsoft.com"

entities_url = endpoint + "/text/analytics/v3.0-preview.1/entities/recognition/general"

print(entities_url)

documents = {"documents": [
    {"id": "2", "text": "corona virus"}
]}

headers = {"Ocp-Apim-Subscription-Key": subscription_key}
response = requests.post(entities_url, headers=headers, json=documents)
entities = dict(response.json())
entry_1 = entities['documents'][0]['entities']
for i in range(len(entry_1)):
    print((entry_1[i]['text'],entry_1[i]['type']))