import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from functools import reduce
from collections import Counter
import numpy as np
import joblib
import jellyfish
from replacements import replacement_sokak,replacement_cadde,replacement_apartman,replacement_mahalle

def extract_integers_from_string(string):
    return [int(item) for item in re.findall(r'\b\d+\b', string)]

def dis( s1: str, s2: str ) -> str:
    '''
    Does: 
    '''
    s1 = list(s1)
    s2 = list(s2)
    s1 = dict(Counter(s1))
    s2 = dict(Counter(s2))

    s1 = list(s1.keys())
    s2 = list(s2.keys())

    intersection = list(set(s1).intersection(set(s2)))
    s1 = [i for i in s1 if i not in intersection]
    s2 = [i for i in s2 if i not in intersection]

    s1 = s1 + s2
    s1 = "".join(s1)

    cost = 0

    for s in s1:
        if s in ['.', ',', ';', ':', '/', '-', ' ']:
            cost = cost + 0
        else:
            cost += 1

    return cost

def levenshteinDistance(str1, str2):
    m = len(str1)
    n = len(str2)
    d = [[i] for i in range(1, m + 1)]   # d matrix rows
    d.insert(0, list(range(0, n + 1)))   # d matrix columns
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:   # Python (string) is 0-based
                substitutionCost = 0
            else:
                substitutionCost = 1
            d[i].insert(j, min(d[i - 1][j] + 1,
                               d[i][j - 1] + 1,
                               d[i - 1][j - 1] + substitutionCost))
    return d[-1][-1]

def getLower( input: str ) -> str:
	input= str(input)
	#: Map
	d = {
	"Ş":"ş", "I":"ı", "Ü":"ü", "Ç":"ç", "Ö":"ö", "Ğ":"ğ", 
	"İ":"i", "Â":"â", "Î":"î", "Û":"û"
	}
	#: Replace
	input = reduce(lambda x, y: x.replace(y, d[y]), d, input)
	input = input.lower()
	#: Return
	return input


def adres(row):
    il = getLower(row['İl'])
    ilce = getLower(row['İlçe'])
    mah = getLower(row['Mahalle'])
    adres = getLower(row['Adres'])
    adres = " " + getLower(str(adres)) + " "
    
    adres = adres.replace(" " + il + " ", " ")
    adres = adres.replace(" " + ilce + " ", " ")
    adres = adres.replace(" " + mah + " ", " ")
    adres = re.sub(r"([^a-zA-Z0-9\s])", r" \1 ", adres)
    adres = re.sub(r" +", " ", adres)

    adres = adres.replace(" ı ", "ı")
    adres = adres.replace(" ç ", "ç")
    adres = adres.replace(" ö ", "ö")
    adres = adres.replace(" ü ", "ü")
    adres = adres.replace(" ğ ", "ğ")
    adres = adres.replace("apartman ı", "apartman")
    adres = adres.replace(" ğ", "ğ")

    for r in eliminate:
        adres = adres.replace(" " + r + " ", " ")

    for r in replacement:
        adres = re.sub(r" +", " ", adres)
        adres = " " + adres.replace(" " + r + " ", " " + replacement[r] + " ").strip() + " "

    adres = re.sub(r" +", " ", adres)
    return adres.strip()

def clean(value):
    '''
    # 
    '''
    value = " " + str(value) + " "
    value = getLower(value)

    for r in replacement:
        value = re.sub(r" +", " ", value)
        value = " " + value.replace(" " + r + " ", " " + replacement[r] + " ").strip() + " "

    value = value.replace("ğ", "g")
    value = value.replace("ı", "i")
    value = value.replace("ç", "c")
    value = value.replace("ö", "o")
    value = value.replace("ü", "u")
    value = value.replace("ş", "s")

    for c in [',', ';', ':', '-', '.', '/']:
        value = value.replace(c, " ") 
    value = re.sub(r" +", " ", value).strip()
    return value

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

def remove_block(text):
    return re.sub(r'\bblok\b', '', text, flags=re.IGNORECASE)

def process_apart_no(row):
    '''
    Does: uses regex for cases below (IN ORDER)
        Case 1:
            If "blok" and "No" in same particular cell, donot touch it
        Case 2:
            If ONLY "No" in particular cell, format cell as No: {İf list[integer] == 1 // 
                                                             else No: {İf remainin integers}
        Case 3
            If ONLY "blok" in particular cell, format cell as Blok: {remaining String}
        Case 4:
            If ONLY integer, format cells as No: {İf list[integer] == 1 // 
                                                 else No: {İf remainin integers}
        Case 5:
            If just strings in particular cell than live it be.
    '''
    # Get particular row and start ops
    row_string = row["Dış Kapı/ Blok/Apartman No"].lower()
    # Case 1
    if "blok" and "no" in row_string:
        return row
    if "no" in row_string:
        detected_no_list = [int(item) for item in re.findall(r'\b\d+\b', row_string)]
        if len(detected_no_list) == 1:
            row["Dış Kapı/ Blok/Apartman No"] =  f'No: {detected_no_list[0]}'
            return row
        else:
            return row
    if "blok" in row_string:
        # Check if integer exists
        detected_int_list = [int(item) for item in re.findall(r'\b\d+\b', row_string)]
        if len(detected_int_list) == 0:
            remaining_string = remove_block(row_string)
            if remaining_string != '':
                row["Dış Kapı/ Blok/Apartman No"] =  f'Blok: {remaining_string}'
                return row
            else:
                return row
        else:
            return row
    if  str.isdigit(row_string) == True:
        detected_int_list = [int(item) for item in re.findall(r'\b\d+\b', row_string)]
        row["Dış Kapı/ Blok/Apartman No"] =  f'No: {detected_int_list[0]}'
        return row
    return row

# prepares the word to be corrected
def clean_words(word=str):
    
    # lower case
    if 'I' in word:
        word = word.replace('I','ı')
        word = word.lower()
    else:
        word = word.lower()
    word = word.strip()
    # find \n and remove it
    word = word.replace('\n','')
    # delete comma
    word = word.replace(',','')
    # delete dot
    word = word.replace('.','')
    # delete slash
    word = word.replace('/','')

    return word


def il_ilce_mah_corrector(df):
    '''
    Finds the correct il/ilce/mahalle from the data received from icisleri.gov.tr
    and replaces the similar words exist in the current dataframe
    '''

    df['İl'].fillna('undefined_il', inplace=True)
    df['İlçe'].fillna('undefined_ilçe', inplace=True)
    df['Mahalle'].fillna('undefined_mahalle', inplace=True)
    # Delete 'Mahallesi' from Mahalle column
    df['Mahalle'] = df['Mahalle'].str.replace('Mahallesi', '')
    df['Mahalle'] = df['Mahalle'].str.replace('MAHALLESİ', '')

    # read correct csv file to check to correct the data
    df_correct = pd.read_csv('reference_data/Mahalle_Koy_joined.csv',header=0, on_bad_lines='skip')
    for i in range(0, len(df_correct['ILCE'])):
        if 'MERKEZİ' in df_correct['ILCE'][i]:
            df_correct.at[i, 'ILCE'] = 'MERKEZ'

    # get correct il list
    correct_il = df_correct['IL'].unique().tolist()
    # iter rows of merged data
    for index, row in df.iterrows():
        if row['İl'] not in correct_il:
            il_score_list = []
            for il in correct_il:
                score = jellyfish.jaro_winkler_similarity(clean_words(row['İl']), clean_words(il))
                il_score_list.append(score)
            if max(il_score_list) > 0.85:
                il = correct_il[il_score_list.index(max(il_score_list))]
                df.at[index, 'İl'] = il

                ilce_list = df_correct[df_correct['IL'] == il]['ILCE'].unique().tolist()
                if row['İlçe'] not in ilce_list:
                    ilce_score_list = []
                    for ilce in ilce_list:
                        score = jellyfish.jaro_winkler_similarity(clean_words(row['İlçe']), clean_words(ilce))
                        ilce_score_list.append(score)
                    if max(ilce_score_list) > 0.85:
                        ilce = ilce_list[ilce_score_list.index(max(ilce_score_list))]
                        df.at[index, 'İlçe'] = ilce

                        mahalle_list = df_correct[df_correct['IL'] == il][df_correct['ILCE'] == ilce]['MAHALLE'].unique().tolist()
                        if row['Mahalle'] not in mahalle_list:
                            mahalle_score_list = []
                            for mahalle in mahalle_list:
                                if 'MAHALLE' in mahalle:
                                    mahalle = mahalle.replace('MAHALLE', '')
                                score = jellyfish.jaro_winkler_similarity(clean_words(row['Mahalle']), clean_words(mahalle))

                                mahalle_score_list.append(score)
                            if max(mahalle_score_list) > 0.85:
                                mahalle = mahalle_list[mahalle_score_list.index(max(mahalle_score_list))]
                                df.at[index, 'Mahalle'] = mahalle
    return df

def find_adres_value(row):
    '''
    Finds the value of 'Adres' column 
    '''
    # if 'Adres' column is exist
    if 'Adres' in row:
        row = row['Adres']
        valueble_words = ['il','ilçe','mahalle','sokak','cadde','bulvar','apartman','no']
        score=0
        for word in valueble_words:
            if word in row:
                score += 1
        return score
    else:
        # return empty string
        return 0

def add_mah_str(row):
    '''
    Adds "Mahallesi" to 'Mahalle' column
    '''
    row['Mahalle'] = row['Mahalle'] + ' Mahallesi'
    return row

def replace_nan_with_0(row):
    '''
    Replace NaN with 0
    '''
    if np.isnan(row['oran']) == True:
        row['oran'] = 0
    return row
# If this script is called
if __name__ == '__main__':
    # Import df
    dff = pd.read_csv("data/merged_v1_5.csv")
    # CALL IL/ILCE/MAHALLE HANDLER
    ################
    dff = il_ilce_mah_corrector(dff)
    dff['Mahalle'] = dff['Mahalle'].apply(lambda value: str(value).replace('undefined_mahalle',""))
    dff = dff.apply(lambda row: add_mah_str(row),axis=1)
    ################
    # Fill na
    dff = dff.fillna("")
    # Rule based prep
    dff['Mahalle'] = dff['Mahalle'].apply(lambda value: str(value).replace(" Mahallesi", ""))
    dff['Mahalle'] = dff['Mahalle'].apply(lambda value: str(value).replace(" MAHALLESI", ""))
    dff['Mahalle'] = dff['Mahalle'].apply(lambda value: str(value).replace(" MAHALLESİ", ""))
    # Define rule base string ops
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
    eliminate = ['mahallesi', 'mah.', 'mah']
    # Clean specific columns
    for c in ['İl','İlçe','Mahalle','Adres','Ad-Soyad']:
        dff[c] = dff[c].apply(lambda value: clean(value))
    # Call address adress func
    dff['new_adres'] = dff.apply(lambda row: adres(row), axis = 1)
    dff = dff.apply(lambda row:process_apart_no(row),axis=1)
    # Sort by columns below
    # dff = dff.sort_values(['İl','İlçe','Mahalle','A'])
    print(len(dff),'len dff before drop')
    dff = dff.drop_duplicates(subset=["İl","İlçe","Mahalle","Bina Adı","Bulvar/Cadde/Sokak/Yol/Yanyol",
            "Ad-Soyad","İç Kapı","Adres","Telefon","Dış Kapı/ Blok/Apartman No"])  
    # print dropped len
    print(len(dff),'len dff after drop')
    # TF - IDF
    isimler = dff["Ad-Soyad"].values
    id_dff_isim = dff["id"].values
    tfidff_isim = TfidfVectorizer().fit_transform(isimler)
    pairwise_similarity_isim = tfidff_isim * tfidff_isim.T
    idler = []
    oran = []
    for idx in pairwise_similarity_isim.toarray():
        id_bulma = [ids for ids in idx]
        try:
            idler.append(id_dff_isim[id_bulma.index(sorted(id_bulma,reverse=True)[1])])
            oran.append(sorted(id_bulma,reverse=True)[1])
        except:
            idler.append(None)
            oran.append(None)
    dff["oran_isim"] = oran
    dff["benzer_id_isim"] = idler
    dff_concat = []
#dfff.dropna(thresh=2, inplace=True)

    dff1 = dff #[(dff["oran_isim"] >0.90) ] #& (dfff["benzer_id_isim"] != dfff["id"])
    dff1 = dff1.fillna("")
    for i in dff1.groupby(['İl', 'İlçe', 'Mahalle']):
        dff_i = i[1]
        dff_i["text"] = dff_i['Bina Adı'] +" " + dff_i['Dış Kapı/ Blok/Apartman No'] + " " +dff_i["Bulvar/Cadde/Sokak/Yol/Yanyol"]+" " + dff_i["new_adres"] #+ " " + dff_i["Ad-Soyad"]
        dff_i["text"] = dff_i["text"].replace(replacement_apartman)
        dff_i["text"] = dff_i["text"].replace(replacement_cadde)
        dff_i["text"] = dff_i["text"].replace(replacement_mahalle)
        dff_i["text"] = dff_i["text"].replace(replacement_sokak)
        dff_i["text"] = dff_i["text"].apply(text_edit)
        text = dff_i["text"].values  
        id_dff = dff_i["id"].values
        
        try:
            tfidff = TfidfVectorizer().fit_transform(text)
            pairwise_similarity = tfidff * tfidff.T
            idler = []
            oran = []
            for idx in pairwise_similarity.toarray():
                id_bulma = [ids for ids in idx]
                try:
                    idler.append(id_dff[id_bulma.index(sorted(id_bulma,reverse=True)[1])])
                    oran.append(sorted(id_bulma,reverse=True)[1])
                    
                except:
                    idler.append(None)
                    oran.append(None)
        
            dff_i["benzer_id"] = idler
            
            dff_i["oran"] = oran
            dff_i["İl"] = i[0][0]
            dff_i["İlçe"] = i[0][1]
            dff_i["Mahalle"] = i[0][2]

            dff_filtre = dff_i #[(dff_i["oran"] > 0.40) ] #& (dff_i["benzer_id"] != dff_i["id"])
            if len(dff_filtre)>0:
                dff_concat.append(dff_filtre)
        except:
            pass
        dffc = pd.concat(dff_concat)
        # dffc.to_excel("sonuc_isim_id2_90-40.xlsx",index=False)
    # Replace NaN with 0
    dffc = dffc.apply(lambda row: replace_nan_with_0(row),axis=1)
    # DUMP TF/IDF model
    # Save the Tf-Idf model to disk
    joblib.dump(tfidff_isim, 'tfidf_model.joblib')
    # FIND MVP INDEX (SELF MAX SCORE MATCHED BY CHECKİNG "oran" and "oran_isim") 
    MVP_indexes = dffc[(dffc["id"] == dffc["benzer_id_isim"]) & (dffc["id"] == dffc["benzer_id"])]["id"].to_list()
    mvp_index_df = dffc[dffc.id.isin(MVP_indexes)]
    # Declare threshold ratios
    o_isim = 0.80
    o_text = 0.40
    # Declare filter_df for presantation (ITS NOT FOR similart elimination!!!) 
    filter_df = []
    # Declare similarity dict, its for elimating similar rows
    similarity_dict = {}
    # Iterate mvp_index_df by grouping columns below
    for i in mvp_index_df[["id",'benzer_id_isim', 'benzer_id','oran_isim', 'oran']].values:
        row_id = i[0]
        main_rowid = mvp_index_df[mvp_index_df["id"] == row_id]
        # Detect similar row by checking their "oran" and "oran_isim" ratios
        tmp_df_filter = dffc[(dffc["id"] != row_id) & (dffc["benzer_id"] == row_id) & (dffc["benzer_id_isim"] == row_id) & (dffc["oran"] >o_text) & (dffc["oran_isim"] > o_isim)]
        # If there is a similar row, than append it to list
        if len(tmp_df_filter)>0:
            similarity_dict[row_id] = tmp_df_filter["id"].to_list()
            filter_df.append(tmp_df_filter)

    #f_df = pd.concat(filter_df)
    #dffc_left = mvp_index_df.merge(f_df[["id","text","benzer_id_isim" ,'Adres', 'Bulvar/Cadde/Sokak/Yol/Yanyol',
    #    'Bina Adı', 'Dış Kapı/ Blok/Apartman No']],on="benzer_id_isim",how="left")
    # print presentation Dataframe
    #dffc_left.to_excel("filter_df_80-40.xlsx",index=False)
    # Append all indexes to list called index_value
    index_value = []
    for key,value in zip(list(similarity_dict.keys()),list(similarity_dict.values())):
        for v in value:
            index_value.append(int(v))
    # Remove mvp_indexes mapped similar indexes
    print(len(dffc), 'len ddfc before sim drop')
    dropped_similars_df = dffc[~dffc.id.isin(index_value)]
    print(len(dropped_similars_df), 'after ddfc sim drop')
    #print(len(dropped_similars_df),'len of dropped_similars_df')
    #print(dropped_similars_df.head(5))
    #print(len(dropped_similars_df), 'before 0.85 drop')
    #dropped_similars_df_final = dropped_similars_df[dropped_similars_df.oran_isim > 0.85].sort_values(['İl','İlçe','Mahalle','Dış Kapı/ Blok/Apartman No'])
    #print(len(dropped_similars_df_final), 'after 0.85 drop')
    print(len(dropped_similars_df), 'before drop eksik bilgi')
    dropped_similars_df_final = dropped_similars_df[~dropped_similars_df['text'].str.contains("eksik bilgi")]
    print(len(dropped_similars_df_final), 'after eksik bilgi drop')
    dropped_similars_df_final.to_excel("ihttimaller.xlsx", index=False)



    