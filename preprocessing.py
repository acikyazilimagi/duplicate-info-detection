import re
import pandas as pd

from replacements import replacement_sokak, replacement_cadde, replacement_apartman, replacement_mahalle, replacement_site 
from expressions import ifadeler, yer_yon_belirten
yer_yon_belirten_non_turkish = [replace_turkish_letters(expr) for expr in yer_yon_belirten]

"""
1. preprocessing yapilacak:
    - temel string normalizasyonu butun columnlar icin yapilacak:
        - birden fazla whitespace tek space'e collapse edilecek [OK]
        - turkce harf degistirmeler [OK]
        - kucuk harfe alma [OK]
        - regex replacementlar (url vb) [OK]
        - ozel semboller [OK]
        - noktalama isaretleri [Emirhan'a sor]
    - column ozeli filterelemeler yapilacak:
        - Bina No'da sadece numara varsa basina no eklenecek [OK]
        - il/ilce/mahalle exact matchleri adres column'undan silinecek [OK]
    - sok mah vb replace edilecek
    - adres column'undaki duplicate kelimeler cikarilacak 
    - alakasiz ifade varsa silinecek
    - yer yon ifadesi hic yoksa boslukla degisiyo
2. adres merge edilecek [OK]
    - bina sokak vb columnlar dikkate alinmayacak merge'den sonra [OK]
4. mahalle manipulation'lar yapilacak:
    - sonundaki "mahallesi", "mahalle", "mah" vb silinecek
    - bos olanlar "undefined" olarak guncellenicek
5. ilce manipulation'lar yapilacak:
    - bos olanlar "undefined" olarak guncellenicek
6. il manipulation'lar yapilacak:
    - bos olanlar "undefined" olarak guncellenicek
7. il/ilce/mahalle icisileri dokumanina gore normalize edilecek
8. exact match'ler process edilecek:
    - exact match olanlar arasinda, en dusuk index'li olan original secilecek
    - exact match olanlar arasinda digerlerine original index eklenecek
    - sonra bunlarin clustering'e girmemesi saglanacak
"""

"""
Removes lowercase turkish letters from a string
"""
def replace_turkish_letters(input: str) -> str:
    return input \
        .replace("ğ", "g") \
        .replace("ı", "i") \
        .replace("ç", "c") \
        .replace("ö", "o") \
        .replace("ü", "u") \
        .replace("ş", "s") \
        .replace("â", "a") \
        .replace("î", "i") \
        .replace("û", "u")

"""
Removes a set of regex patterns from a string, to remove:
- URLs
- Tags
- Hashtags
- Substrings wrapped in ()
- Substrings wrapped in !!<string>!!
"""
def replace_regex_patterns(input: str) -> str:
    regexes = [
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', # URLs
        r'[^\w\s#@/:%.,_-]', # Emoji
        r'@[A-Za-z0-9_]+', # Tags
        r'#[A-Za-z0-9_]+', # Hashtags
        r'(\([^)]+\)+)', # Substrings wrapped in ()
        r'(\!+[^!]+\!+)' # Substrings wrapped in !!
    ]
    for regex in regexes:
        input = re.sub(regex, '', input)
    return input

"""
Replaces all tabs, newlines, leading/trailing, consecuutive spaces with a single space

python string.split() with no arguments splits on all whitespace.
Searching online suggests splitting and joining 
"""
def clean_whitespace(input: str) -> str:
    return input.split().join(" ").strip()

"""
Applies general purpose preprocessing to a string, including:
- lowercasing
- Turkish letter replacement
- whitespace cleaning
- regex pattern replacement
- help call string removal
"""
def process_column_string(input: str) -> str:
    lower = input.lower()
    whitespace_cleaned = clean_whitespace(lower)
    un_turkish = replace_turkish_letters(whitespace_cleaned)
    regex_cleaned = replace_regex_patterns(un_turkish)
    help_calls_removed = replace_help_call_strings(regex_cleaned) 
    return help_calls_removed

def replace_address_abbreviations(input: str) -> str:
    replacement_dicts = [
        replacement_site,
        replacement_apartman,
        replacement_cadde,
        replacement_mahalle,
        replacement_sokak
    ]
    for replacement_dict in replacement_dicts:
        for frm, to in replacement_dict.items():
            input = input.replace(frm, to)
    return input

def replace_help_call_strings(input: str) -> str:
    for help_string in ifadeler:
        input = input.replace(help_string, "").strip()
    return input

"""
Returns true if the input string contains any of the address phrases
"""
def is_non_address_string(input: str) -> bool:
    return any(phrase in input for phrase in yer_yon_belirten_non_turkish)
    
"""
Applies address specific preprocessing to a string, including:
- replace abbreviations such as sok mah cad etc.
- remove strings with no known address phrases
"""
def process_address_column_string(input: str) -> str:
    abbreviations_replaced = replace_address_abbreviations(input)
    if is_non_address_string(abbreviations_replaced):
        return ""
    return abbreviations_replaced

"""
Processes street/door number and "block" phrases in an address string
"""
def process_building_number(input: str) -> str:
    no_regex = r"\bno\b(\s+|:|\.) ?(\d+)"
    nos_processed = re.sub(no_regex, r"no\2 ", input)
    blok_regex = r"( )?blok\b(-|\.| )?"
    blok_processed = re.sub(blok_regex, "blok ", nos_processed)
    blok_pre_regex = r"\bblok (\w)"
    blok_processed = re.sub(blok_pre_regex, r"\1blok ", blok_processed)
    return blok_processed

"""
Applies the following transformations to the input dataframe:
- applies the following string normalization to all columns that will be used in clustering:
    - lowercasing
    - Turkish letter replacement
    - whitespace cleaning
    - regex pattern replacement
- replaces address-specific abbreviations in all address columns
- removes any words in the Addres column that are found in the other address-specific columns
- adds the "no" suffix to the building number column if it's just a number
- merges all address columns into a single "processed address" column
- removes any duplicate words from the "processed address" column
- normalizes building number/block phrases
"""
def process_pre_merge_columns(row: pd.Series) -> pd.Series:
    name_column = ["Ad-Soyad"]
    address_columns = [
        "İl",
        "İlçe",
        "Mahalle",
        "Bina Adı", 
        "Dış Kapı/ Blok/Apartman No", 
        "Bulvar/Cadde/Sokak/Yol/Yanyol",
        "Adres",
    ]

    # do standard string normalization for all the columns
    for column in address_columns + name_column:
        row[column] = process_column_string(row[column])

    # replace address-specific abbreviations in address columns
    for column in address_columns:
        row[column] = process_address_column_string(row[column])

    # remove any duplicate words in the Adres columg
    row['Adres'] = row['Adres'] \
        .replace(row["İl"], "") \
        .replace(row["İlçe"], "") \
        .replace(row["Mahalle"], "") \
        .replace(row["Bina Adı"], "") \
        .replace(row['Dış Kapı/ Blok/Apartman No'], "") \
        .replace(row["Bulvar/Cadde/Sokak/Yol/Yanyol"], "")

    # add the suffix "no" to the door number if it is just a number
    existing_no = row['Dış Kapı/ Blok/Apartman No']
    new_no = re.sub(r"^(\d+)$", r"no\1", existing_no)
    row['Dış Kapı/ Blok/Apartman No'] = new_no

    # merge the address columns into one
    merged_address = " ".join(
        row['Bina Adı'],
        row['Dış Kapı/ Blok/Apartman No'],
        row["Bulvar/Cadde/Sokak/Yol/Yanyol"],
        row['Adres'],
    )

    # again remove the city, district, and neighborhood from the merged address
    merged_address = merged_address \
        .replace(row["İl"], "") \
        .replace(row["İlçe"], "") \
        .replace(row["Mahalle"], "")

    # normalize building number/block phrases
    merged_address = process_building_number(merged_address)

    row['processed_address'] = merged_address
    return row


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.apply(process_pre_merge_columns, axis=1)
    df.drop(columns=['Adres', 'Bina Adı', 'Dış Kapı/ Blok/Apartman No', 'Bulvar/Cadde/Sokak/Yol/Yanyol'], inplace=True)
    # TODO: il/ilce/mahalle icisleri dokumanina normalize edilecek
    # TODO: post-normalize il/ilce/mahalle adres column'unda bulunursa silinecek
    # TODO: normalize sonrasi exac match duplicate'lar silinecek
    # TODO: silinen duplicate'lar aslinda silinmeden tabloda mark edilecek