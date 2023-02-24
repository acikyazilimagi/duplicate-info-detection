import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from functools import reduce
from collections import Counter
import numpy as np
import joblib
import jellyfish
from replacements import replacement_sokak, replacement_cadde, replacement_apartman, replacement_mahalle, replacement_site 
from expressions import ifadeler, yer_yon_belirten


def extract_integers_from_string(string):
    return [int(item) for item in re.findall(r'\b\d+\b', string)]


def dis(s1: str, s2: str) -> str:
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


def getLower(input: str) -> str:
    input = str(input)
    #: Map
    d = {
        "Ş": "ş", "I": "ı", "Ü": "ü", "Ç": "ç", "Ö": "ö", "Ğ": "ğ",
        "İ": "i", "Â": "â", "Î": "î", "Û": "û"
    }
    #: Replace
    input = reduce(lambda x, y: x.replace(y, d[y]), d, input)
    input = input.lower()
    #: Return
    return input

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

def adres(row):
    
    il = getLower(row['İl'])
    ilce = getLower(row['İlçe'])
    mah = getLower(row['Mahalle'])
    
    adres = getLower(row['Adres'])
    adres = " " + getLower(str(adres)) + " "

    adres = adres.replace(" " + il + " ", " ")
    adres = adres.replace(" " + ilce + " ", " ")
    adres = adres.replace(" " + "mahallesi" + " ", " ")
    adres = adres.replace(" " + "mah." + " ", " ")
    adres = adres.replace(" " + "mah" + " ", " ")
    adres = re.sub(r"([^a-zA-Z0-9\s])", r" \1 ", adres)
    adres = re.sub(r" +", " ", adres)
    
    adres = adres.replace(" ı ", "ı")
    adres = adres.replace(" ç ", "ç")
    adres = adres.replace(" ö ", "ö")
    adres = adres.replace(" ü ", "ü")
    adres = adres.replace(" ğ ", "ğ")
    adres = adres.replace("apartman ı", "apartman")
    adres = adres.replace(" ğ", "ğ")
    
                                                 
    # replacement handle edilecek
    for r in replacement:
        adres = re.sub(r" +", " ", adres)
        adres = " " + adres.replace(" " + r + " ",
                                    " " + replacement[r] + " ").strip() + " "

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
        value = " " + value.replace(" " + r + " ",
                                    " " + replacement[r] + " ").strip() + " "

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
        detected_no_list = [int(item)
                            for item in re.findall(r'\b\d+\b', row_string)]
        if len(detected_no_list) == 1:
            row["Dış Kapı/ Blok/Apartman No"] = f'No: {detected_no_list[0]}'
            return row
        else:
            return row
    if "blok" in row_string:
        # Check if integer exists
        detected_int_list = [int(item)
                             for item in re.findall(r'\b\d+\b', row_string)]
        if len(detected_int_list) == 0:
            remaining_string = remove_block(row_string)
            if remaining_string != '':
                row["Dış Kapı/ Blok/Apartman No"] = f'{remaining_string} Blok'
                return row
            else:
                return row
        else:
            return row
    if str.isdigit(row_string) == True:
        detected_int_list = [int(item)
                             for item in re.findall(r'\b\d+\b', row_string)]
        row["Dış Kapı/ Blok/Apartman No"] = f'No: {detected_int_list[0]}'
        return row
    return row

# prepares the word to be corrected


def clean_words(word=str):

    # lower case
    if 'I' in word:
        word = word.replace('I', 'ı')
        word = word.lower()
    else:
        word = word.lower()
    word = word.strip()
    # find \n and remove it
    word = word.replace('\n', '')
    # delete comma
    word = word.replace(',', '')
    # delete dot
    word = word.replace('.', '')
    # delete slash
    word = word.replace('/', '')

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
    df_correct = pd.read_csv(
        'reference_data/Mahalle_Koy_joined.csv', header=0, on_bad_lines='skip')
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
                score = jellyfish.jaro_winkler_similarity(
                    clean_words(row['İl']), clean_words(il))
                il_score_list.append(score)
            if max(il_score_list) > 0.85:
                il = correct_il[il_score_list.index(max(il_score_list))]
                df.at[index, 'İl'] = il

                ilce_list = df_correct[df_correct['IL']
                                       == il]['ILCE'].unique().tolist()
                if row['İlçe'] not in ilce_list:
                    ilce_score_list = []
                    for ilce in ilce_list:
                        score = jellyfish.jaro_winkler_similarity(
                            clean_words(row['İlçe']), clean_words(ilce))
                        ilce_score_list.append(score)
                    if max(ilce_score_list) > 0.85:
                        ilce = ilce_list[ilce_score_list.index(
                            max(ilce_score_list))]
                        df.at[index, 'İlçe'] = ilce

                        mahalle_list = df_correct[df_correct['IL'] ==
                                                  il][df_correct['ILCE'] == ilce]['MAHALLE'].unique().tolist()
                        if row['Mahalle'] not in mahalle_list:
                            mahalle_score_list = []
                            for mahalle in mahalle_list:
                                if 'MAHALLE' in mahalle:
                                    mahalle = mahalle.replace('MAHALLE', '')
                                score = jellyfish.jaro_winkler_similarity(
                                    clean_words(row['Mahalle']), clean_words(mahalle))

                                mahalle_score_list.append(score)
                            if max(mahalle_score_list) > 0.85:
                                mahalle = mahalle_list[mahalle_score_list.index(
                                    max(mahalle_score_list))]
                                df.at[index, 'Mahalle'] = mahalle
    return df


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


def detect_non_adress(row):
    '''
    If no adress strings exists, it updates row['new_adress'] as ""
    '''
    adres_string = row['new_adres']
    if adres_string == "":
        return row
    # Counter is matched string list calculator lists, at the end of the for if its 0 then assign adress row as ""
    counter = list()
    for adres in yer_yon_belirten:
        if adres.lower() in adres_string:
            counter.append(adres)
    if len(counter) == 0:
        row['new_adres'] = ""
        return row
    else:
        return row


def replace_help_call_strings(row):
    '''
    Removes help strings if exists in row strings
    '''
    adres_string = row['new_adres']
    if adres_string == "":
        return row
    for help_string in ifadeler:
        if help_string.lower() in row['new_adres'].lower():
            adres_string = adres_string.replace(help_string, "").strip()
    row['new_adres'] = adres_string
    return row

def do_replacements(rows, col="text"):
    rows[col] = rows[col].replace(replacement_site)
    rows[col] = rows[col].replace(replacement_apartman)
    rows[col] = rows[col].replace(replacement_cadde)
    rows[col] = rows[col].replace(replacement_mahalle)
    rows[col] = rows[col].replace(replacement_sokak)
    rows[col] = rows[col].apply(text_edit)
    return rows
    
def run_preprocess(df: pd.DataFrame):
    df = il_ilce_mah_corrector(df)
    df['Mahalle'] = df['Mahalle'].apply(
        lambda value: str(value).replace('undefined_mahalle', ""))
    df = df.apply(lambda row: add_mah_str(row), axis=1)
    df = df.fillna("")
    df["group"] = df["İl"] +"_" +df["İlçe"] +"_" +df["Mahalle"] +"_" +df["Bina Adı"] +"_" +df["Bulvar/Cadde/Sokak/Yol/Yanyol"] +"_" +df["Ad-Soyad"] +"_" +df["İç Kapı"]+"_" +df["Adres"]+"_" +df["Telefon"]+"_" +df["Dış Kapı/ Blok/Apartman No"]
    df = df.drop_duplicates(["group"])
    df["Adres"] = df["Adres"].str.lower()
    df['Adres'] = df['Adres'].str.replace('\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '',regex=True) # remove url
    df['Adres'] = df['Adres'].str.replace('@[A-Za-z0-9_]+', '',regex=True) # remove tag
    df['Adres'] = df['Adres'].str.replace('#[A-Za-z0-9_]+', '',regex=True) # remove hashtag
    df['Adres'] = df['Adres'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE) #emoji
    df['Adres'] = df['Adres'].str.replace('\n', '')
    df['Adres'] = df['Adres'].str.replace('\t', '')
    
    # Rule based prep
    df['Mahalle'] = df['Mahalle'].apply(
        lambda value: str(value).replace(" Mahallesi", ""))
    df['Mahalle'] = df['Mahalle'].apply(
        lambda value: str(value).replace(" MAHALLESI", ""))
    df['Mahalle'] = df['Mahalle'].apply(
        lambda value: str(value).replace(" MAHALLESİ", ""))
   
    # Define rule base string ops
    # Clean specific columns
    for c in ['İl', 'İlçe', 'Mahalle', 'Adres', 'Ad-Soyad']:
        df[c] = df[c].apply(lambda value: clean(value))
        
    # Call address adress func
    df['new_adres'] = df.apply(lambda row: adres(row), axis=1)
    df = df.apply(lambda row: replace_help_call_strings(row), axis=1)
    df = df.apply(lambda row: detect_non_adress(row), axis=1)
    df = df.apply(lambda row: process_apart_no(row), axis=1)

    return df
