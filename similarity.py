
import pandas as pd
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from functools import reduce
from collections import Counter
import numpy as np
import pickle as pkl
from preprocess_funcs import run_preprocess, text_edit, do_replacements

class DataFrameHandler():

    def read_data(self, path):
        dff = pd.read_csv(path)


def train_tfidf_from_csv(data_path="data/merged_v1_4.csv"):
    dff = pd.read_csv('data/merged_v1_4.csv')
    dff1 = run_preprocess(dff)
    # isimler = dff["Ad-Soyad"].values
    # # train and store TF-IDF

    # # print("Training TF-IDF names model")
    # isim_vectorizer = TfidfVectorizer()
    # tfidff_isim = isim_vectorizer.fit(isimler)
    # tfidff_isim_vectors = isim_vectorizer.transform(isimler)
    # # print('tf idf isim done')
    # # print("Saving TF-IDF names model")
    # pkl.dump((tfidff_isim, tfidff_isim_vectors),
    #          open("models/tfidff_isim.pkl", 'wb'))

    
    
    # [(dff["oran_isim"] >0.90) ] #& (dfff["benzer_id_isim"] != dfff["id"])

    dff1 = dff1.fillna("")
    dff1["text"] = dff1['Bina Adı'] + " " + dff1['Dış Kapı/ Blok/Apartman No'] \
        + " " + dff1["Bulvar/Cadde/Sokak/Yol/Yanyol"] + " " + dff1["new_adres"]
    text_tfidf_models = dict()
    dff_concat = []
    for i in dff1.groupby(['İl', 'İlçe', 'Mahalle']):
        dff_i = i[1]

        dff_i["text"] = dff_i['Bina Adı'] + " " + dff_i['Dış Kapı/ Blok/Apartman No'] + " " + dff_i[
            "Bulvar/Cadde/Sokak/Yol/Yanyol"] + " " + dff_i["new_adres"]  # + " " + dff_i["Ad-Soyad"]
        
        do_replacements(dff_i)
        
        text = dff_i["text"]
        isimler = dff_i["Ad-Soyad"]
        try:
            # print("Training TF-IDF text model for " + str(i[0]))
            text_vectorizer = TfidfVectorizer()
            tfidff = text_vectorizer.fit(text)
            vectors = text_vectorizer.transform(text)
            text_tfidf_models[str(i[0])] = (tfidff, vectors)
            # print('tf idf text done')

            tfidff = TfidfVectorizer().fit_transform(text)

            id_dff = dff_i["id"].values
            pairwise_similarity = tfidff * tfidff.T 
            idler = []
            oran = []
            for idx in pairwise_similarity.toarray():
                id_bulma = [ids for ids in idx]
                try:
                    idler.append(id_dff[id_bulma.index(
                        sorted(id_bulma, reverse=True)[1])])
                    oran.append(sorted(id_bulma, reverse=True)[1])

                except:
                    idler.append(None)
                    oran.append(None)

            dff_i["benzer_id"] = idler

            dff_i["oran"] = oran
            dff_i["İl"] = i[0][0]
            dff_i["İlçe"] = i[0][1]
            dff_i["Mahalle"] = i[0][2]

            # Ad-Soyad TFIDF
            id_dff_isim = id_dff
            tfidff_isim = TfidfVectorizer().fit_transform(isimler)
            pairwise_similarity_isim = tfidff_isim * tfidff_isim.T
            idler = []
            oran = []
            for idx in pairwise_similarity_isim.toarray():
                id_bulma = [ids for ids in idx]
                try:
                    idler.append(id_dff_isim[id_bulma.index(
                        sorted(id_bulma, reverse=True)[1])])
                    oran.append(sorted(id_bulma, reverse=True)[1])
                except:
                    idler.append(None)
                    oran.append(None)

            dff_i["benzer_id_isim"] = idler
            dff_i["oran_isim"] = oran

            # [(dff_i["oran"] > 0.40) ] #& (dff_i["benzer_id"] != dff_i["id"])
            dff_filtre = dff_i
            if len(dff_filtre) > 0:
                dff_concat.append(dff_filtre)
        except ValueError:
            print("Passed model for " + str(i[0]))

    # print("Saving TF-IDF text indexed models")
    dffc = pd.concat(dff_concat, ignore_index=True)
    dffc.to_excel("deneme.xlsx", index=False)  # csv yapılacak
    pkl.dump(text_tfidf_models, open("models/tfidff_text.pkl", 'wb'))


def process_csv(csvfile):
    # TODO: implement csv in-csv out process. Take a filename and return a dataframe
    raise NotImplementedError("This needs to be implemented")


if __name__ == "__main__":
    train_tfidf_from_csv()
