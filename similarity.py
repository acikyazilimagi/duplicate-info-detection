
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from functools import reduce
from collections import Counter
import numpy as np
import pickle as pkl
from preprocess_funcs import run_preprocess, text_edit, do_replacements
import time

class DataFrameHandler():

    def read_data(self, path):
        dff = pd.read_csv(path)


def train_tfidf_from_csv(data_path="data/merged_v1_4.csv"):
    df_main = pd.read_csv(r'C:\Users\savo\Desktop\deprem_yardım\acikkaynak\merged_v1_4.csv')
    df_main = run_preprocess(df_main)
    
    df_main = df_main.fillna("")
    df_main["text"] = df_main['Bina Adı'] + " " + df_main['Dış Kapı/ Blok/Apartman No'] \
        + " " + df_main["Bulvar/Cadde/Sokak/Yol/Yanyol"] + " " + df_main["new_adres"]
    text_tfidf_models = dict()
    df_concat = []
    toplam = []
    for i in df_main.groupby(['İl', 'İlçe', 'Mahalle']):
        df_groupby = i[1]

        df_groupby["text"] = df_groupby['Bina Adı'] + " " + df_groupby['Dış Kapı/ Blok/Apartman No'] + " " + df_groupby[
            "Bulvar/Cadde/Sokak/Yol/Yanyol"] + " " + df_groupby["new_adres"]  # + " " + dff_i["Ad-Soyad"]
        
        do_replacements(df_groupby)
        
        
        
        try:
            # Ad-Soyad TFIDF
            isimler = df_groupby["Ad-Soyad"]
            id_df_isim = df_groupby["id"].values
            tfidf_isim = TfidfVectorizer().fit_transform(isimler)
            start = time.time()
            pairwise_similarity_isim = tfidf_isim * tfidf_isim.T
            total = time.time()-start
            toplam.append(total)
            print("pairwise_similarity_isim seconds: ",total)
            idler = []
            oran = []
            
            for idx in pairwise_similarity_isim.toarray():
                id_bulma = [ids for ids in idx]
                try:
                    idler.append(id_df_isim[id_bulma.index(
                        sorted(id_bulma, reverse=True)[1])])
                    oran.append(sorted(id_bulma, reverse=True)[1])
                except:
                    idler.append(None)
                    oran.append(None)

            df_groupby["benzer_id_isim"] = idler
            df_groupby["oran_isim"] = oran
            df_groupby = df_groupby[df_groupby["oran_isim"]>0.5].copy()


            # text TFIDF
            text = df_groupby["text"]
            text_vectorizer = TfidfVectorizer()
            tfidf = text_vectorizer.fit_transform(text)
            vectors = text_vectorizer.transform(text)
            text_tfidf_models[str(i[0])] = (tfidf, vectors)
            id_dff = df_groupby["id"].values
            pairwise_similarity = tfidf * tfidf.T 
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
            
            df_groupby["benzer_id"] = idler
            df_groupby["oran"] = oran
            df_groupby["İl"] = i[0][0]
            df_groupby["İlçe"] = i[0][1]
            df_groupby["Mahalle"] = i[0][2]
            if len(df_groupby) > 0:
                df_concat.append(df_groupby)
        except ValueError:
            # print("Passed model for " + str(i[0]))
            pass

    print("toplam: ",sum(toplam))
    df_concat = pd.concat(df_concat, ignore_index=True)
    df_concat.to_excel("deneme.xlsx", index=False)  # csv yapılacak
    pkl.dump(text_tfidf_models, open("models/tfidff_text.pkl", 'wb'))
    return df_concat


def process_csv(csvfile):
    # TODO: implement csv in-csv out process. Take a filename and return a dataframe
    raise NotImplementedError("This needs to be implemented")


def similarity_filter(df_concat:pd.DataFrame):
    referance_rows = df_concat[(df_concat["id"] == df_concat["benzer_id_isim"]) & (df_concat["id"] == df_concat["benzer_id"])]["id"].to_list()
    # HER BİR SCORE İÇİN TMP DF
    main_index_80_40_df = df_concat[df_concat.id.isin(referance_rows)]
    oran_isim = 0.80
    oran_text = 0.40

    filter_df = []
    similarity_dict = {}
    for i in main_index_80_40_df[["id",'benzer_id_isim', 'benzer_id','oran_isim',   'oran']].values:
        row_id = i[0]
        tmp_df_filter = df_concat[(df_concat["id"] != row_id) & (df_concat["benzer_id"] == row_id) & (df_concat["benzer_id_isim"] == row_id) & (df_concat["oran"] >oran_text) & (df_concat["oran_isim"] > oran_isim)]
        if len(tmp_df_filter)>0:
            similarity_dict[row_id] = tmp_df_filter["id"].to_list()
            filter_df.append(tmp_df_filter)

        
    filter_df = pd.concat(filter_df,ignore_index=True)
    df_concat_left = main_index_80_40_df.merge(filter_df[["id","text","benzer_id_isim" ,'Adres', 'Bulvar/Cadde/Sokak/Yol/Yanyol',
        'Bina Adı', 'Dış Kapı/ Blok/Apartman No']],on="benzer_id_isim",how="left")
    df_concat_left.to_excel("filter_df_80-40_prosesyok.xlsx",index=False)

if __name__ == "__main__":
    train = train_tfidf_from_csv()
    similarity_filter(df_concat=train)
