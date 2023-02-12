import json

import pandas as pd
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from functools import reduce
from collections import Counter
import numpy as np
import pickle as pkl
from preprocess_funcs import run_preprocess
from replacements import replacement_sokak,replacement_cadde,replacement_apartman,replacement_mahalle,replacement

eliminate = ['mahallesi', 'mah.', 'mah']

def text_edit(x):
    value = x.lower()
    value = value.replace("ğ", "g")
    value = value.replace("ı", "i")
    value = value.replace("ç", "c")
    value = value.replace("ö", "o")
    value = value.replace("ü", "u")
    value = value.replace("ş", "s")
    split = value.split(" ")
    # print(" ".join(list(dict.fromkeys(split))).strip())
    return " ".join(list(dict.fromkeys(split))).strip()

class DataFrameHandler():

    def read_data(self, path):
        dff = pd.read_csv(path)



def similarity_score(rows):
    rows = pd.DataFrame(rows)
    print(2)
    rows = run_preprocess(rows)
    isimler = rows["Ad-Soyad"].values
    # train and store TF-IDF

    #print("Inferring using TF-IDF names model")
    tfidff_isim = pkl.load(open("models/tfidff_isim.pkl",'rb'))
    isim_model, old_isimler_vectors = tfidff_isim[0], tfidff_isim[1]
    isimler_vectors = isim_model.transform(isimler)

    #Get maximum cosine similarity
    isim_similarities = list(((old_isimler_vectors * isimler_vectors.T).T).toarray().max(axis=1))

    #print("Inferring using TF-IDF text model")
    rows['models_names'] = list(zip(rows['İl'], rows["İlçe"], rows["Mahalle"]))
    rows["text"] = rows['Bina Adı'] + " " + rows['Dış Kapı/ Blok/Apartman No'] + " " + rows[
            "Bulvar/Cadde/Sokak/Yol/Yanyol"] + " " + rows["Adres"]

    text_similarities = []
    for index, row in rows.iterrows():

        try:
            #print("Getting TF-IDF text model for " + str(row["models_names"]))
            tfidff_text = pkl.load(open("models/tfidff_text.pkl", 'rb'))[str(row["models_names"])]
            isim_model, old_text_vectors = tfidff_text[0], tfidff_text[1]
            text_vectors = isim_model.transform([row.text])
            # Get maximum cosine similarity
            text_similarities.append(((old_text_vectors * text_vectors.T).T).max(axis=1).data[0])

        except KeyError:
            print("No model for "+ str(row["models_names"]))
            text_similarities.append(0)
            pass

    df = pd.DataFrame(list(zip(text_similarities, isim_similarities)),
              columns=['text_score','name_score']).reset_index()
    df.rename(columns={"index": "id"}, inplace=True)
    df.loc[df['text_score']>0.8,'is_similar'] = "True"
    df.is_similar.fillna("False",inplace=True)
    result = df.to_json(orient="records")
    parsed = json.loads(result)
    return parsed

def train_tfidf_from_csv(data_path="data/merged_v1.csv"):
    #dff = pd.read_csv('data/merged_v1_5.csv')
    dff = run_preprocess(dff)
    isimler = dff["Ad-Soyad"].values
    #train and store TF-IDF

    #print("Training TF-IDF names model")
    isim_vectorizer = TfidfVectorizer()
    tfidff_isim = isim_vectorizer.fit(isimler)
    tfidff_isim_vectors = isim_vectorizer.transform(isimler)
    #print('tf idf isim done')
    #print("Saving TF-IDF names model")
    pkl.dump((tfidff_isim, tfidff_isim_vectors), open("models/tfidff_isim.pkl", 'wb'))
    replacement = {
        'sk': 'sokak',
        'sok': 'sokak',
        'sokağı': 'sokak',
        'apartmani': 'apartman',
        'apartmanı': 'apartman',
        'apt.': 'apartman',
        'apt': 'apartman',
        'caddesi': 'cadde',
        'cad.': 'cadde',
        'cad': 'cadde',
    }

    dff1 = dff  # [(dff["oran_isim"] >0.90) ] #& (dfff["benzer_id_isim"] != dfff["id"])
    dff1 = dff1.fillna("")
    dff1["text"] = dff1['Bina Adı'] + " " + dff1['Dış Kapı/ Blok/Apartman No'] \
                   + " " + dff1["Bulvar/Cadde/Sokak/Yol/Yanyol"] + " " + dff1["Adres"]
    text_tfidf_models = dict()
    for i in dff1.groupby(['İl', 'İlçe', 'Mahalle']):
        dff_i = i[1]

        dff_i["text"] = dff_i['Bina Adı'] + " " + dff_i['Dış Kapı/ Blok/Apartman No'] + " " + dff_i[
            "Bulvar/Cadde/Sokak/Yol/Yanyol"] + " " + dff_i["Adres"]  # + " " + dff_i["Ad-Soyad"]
        dff_i["text"] = dff_i["text"].replace(replacement)
        dff_i["text"] = dff_i["text"].apply(text_edit)
        text = dff_i["text"]
        try:
            #print("Training TF-IDF text model for " + str(i[0]))
            text_vectorizer = TfidfVectorizer()
            tfidff = text_vectorizer.fit(text)
            vectors = text_vectorizer.transform(text)
            text_tfidf_models[str(i[0])] = (tfidff, vectors)
            #print('tf idf text done')
        except ValueError:
            print("Passed model for " + str(i[0]))

    #print("Saving TF-IDF text indexed models")
    pkl.dump(text_tfidf_models, open("models/tfidff_text.pkl", 'wb'))