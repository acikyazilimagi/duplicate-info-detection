import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import numpy as np
import typer

from preprocess_funcs import run_preprocess

def load_data(data_path: str, preprocess_save_path: str = "data/df_preprocessed.pkl") -> pd.DataFrame:
    if data_path.endswith(".pkl"):
        df = pd.read_pickle(data_path)
        print(f"Loaded processed data from {data_path}, with {df.shape[0]} rows and {df.shape[1]} columns.")
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        df = df.fillna("")
        print(f"Loaded unprocessed data from {data_path}, with {df.shape[0]} rows and {df.shape[1]} columns.")
        df = run_preprocess(df)
        print(f"Processed the data, ended up with {df.shape[0]} rows and {df.shape[1]} columns.")
        pd.to_pickle(df, preprocess_save_path)
    return df

def merge_address_columns(
        df: pd.DataFrame,
        new_column_name: str = "merged_address"
    ) -> pd.DataFrame:
    df[new_column_name] = df['Bina Adı'] + " " + df['Dış Kapı/ Blok/Apartman No'] \
        + " " + df["Bulvar/Cadde/Sokak/Yol/Yanyol"] + " " + df["new_adres"]
    return df

def cluster_by_column(
        df: pd.DataFrame,
        key_column_name: str,
        duplicate_max_distance_threshold: float,
        tfidf_ngram_range: tuple,
        tfidf_min_df: int,
        tfidf_use_char_ngrams: bool,
        df_mask,
) -> pd.DataFrame:
    # index the rows that will be clustered
    df.loc[df_mask, "clustering_index"] = list(range(df.loc[df_mask].shape[0]))

    # derive names for cluster information columns
    cluster_column_name = f"{key_column_name}-cluster"
    duplicate_info_column_name = f"{key_column_name}-duplicate"
    similarity_column_name = f"{key_column_name}-duplicate-similarity"
    duplicate_original_column_name = f"{key_column_name}-duplicate-original-id"

    analyzer = "char_wb" if tfidf_use_char_ngrams else "word"
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=tfidf_ngram_range, min_df=tfidf_min_df)
    vectors = vectorizer.fit_transform(df.loc[df_mask, key_column_name])
    # compute pairwise cosine distances between all rows
    distance_matrix = pairwise_distances(vectors, vectors, metric="cosine")
    # run the DBSCAN clustering algorithm using the pairwise distances
    dbscan = DBSCAN(eps=duplicate_max_distance_threshold, min_samples=2, metric="precomputed") \
        .fit(distance_matrix)
    # annotate each row with the id of the cluster it belongs to
    df.loc[df_mask, cluster_column_name] = dbscan.labels_

    # process each cluster for marking reference points and similarity scores to the reference point
    for cluster in np.unique(dbscan.labels_):
        if cluster == "-1":
            # -1 is the no-cluster cluster label, no need to do anything else
            continue

        # create a mask for the rows in the cluster
        cluster_mask = df_mask & (df[cluster_column_name] == cluster)

        # the first entry in the cluster is picked as the "original"
        original_row_mask = df.index[cluster_mask][0]
        duplicate_row_mask = df.index[cluster_mask][1:]
        original_row = df.loc[original_row_mask]

        # fetch the similarities of each row to the original row
        original_row_similarities = 1.0 - distance_matrix[original_row["clustering_index"], :]

        # mark the original row with "O"
        df.loc[original_row_mask, duplicate_info_column_name] = "O"
        # mark the other rows with "D"
        df.loc[duplicate_row_mask, duplicate_info_column_name] = "D"
        # mark every row with the original row's id
        df.loc[cluster_mask, duplicate_original_column_name] = original_row["id"]
        # mark every row with the similarity score to the original row
        df.loc[cluster_mask, similarity_column_name] = original_row_similarities[df.loc[cluster_mask, "clustering_index"]]

    return df

def cluster_data(
        df: pd.DataFrame,
        name_duplicate_max_distance_threshold: float,
        address_duplicate_max_distance_threshold: float,
        tfidf_ngram_range: tuple,
        tfidf_min_df: int,
        tfidf_use_char_ngrams: bool,
) -> pd.DataFrame:
    df.loc[:, "clustering_index"] = -1
    def cluster_group(group_df):
        try:
            # only cluster names that are defined
            name_defined_mask = (group_df["Ad-Soyad"] != "")
            group_df = cluster_by_column(
                df=group_df, 
                key_column_name="Ad-Soyad", 
                duplicate_max_distance_threshold=name_duplicate_max_distance_threshold, 
                tfidf_ngram_range=tfidf_ngram_range, 
                tfidf_min_df=tfidf_min_df, 
                tfidf_use_char_ngrams=tfidf_use_char_ngrams, 
                df_mask = name_defined_mask)
            # if name is not defined, assign -1 to cluster (means no cluster)
            group_df.loc[group_df["Ad-Soyad"] == "", "Ad-Soyad-cluster"] = -1
        except ValueError as e:
            # in case of any errors when clustering names, assign -1 to every row (means no cluster)
            group_df["Ad-Soyad-cluster"] = -1
        name_clusters = group_df["Ad-Soyad-cluster"].unique()
        for name_cluster in name_clusters:
            cluster_df_mask = (group_df["Ad-Soyad-cluster"] == name_cluster)
            if name_cluster == -1:
                # If name is not in any cluster, no need to cluster addresses
                group_df.loc[cluster_df_mask, 'merged_address-cluster'] = -1
                continue
            try:
                group_df = cluster_by_column(
                    df=group_df, 
                    key_column_name="merged_address", 
                    duplicate_max_distance_threshold=address_duplicate_max_distance_threshold, 
                    tfidf_ngram_range=tfidf_ngram_range, 
                    tfidf_min_df=tfidf_min_df, 
                    tfidf_use_char_ngrams=tfidf_use_char_ngrams, 
                    df_mask=cluster_df_mask)
            except ValueError as e:
                # in case of any errors when clustering addresses, assign -1 to every row (means no cluster)
                group_df.loc[cluster_df_mask, 'merged_address-cluster'] = -1
        return group_df
    df = df \
        .groupby(["İl", "İlçe", "Mahalle"], group_keys=False) \
        .apply(cluster_group)
    df.drop("clustering_index", axis=1, inplace=True)
    return df

def main(
        data_path: str = "data/merged_v1_4.csv",
        name_duplicate_max_distance_threshold: float = 0.2,
        address_duplicate_max_distance_threshold: float = 0.3,
        tfidf_ngram_min: int = 2,
        tfidf_ngram_max: int = 4,
        tfidf_use_char_ngrams: bool = True,
        output_data_path: str = "data/clustered_v_1_4.csv",
        save_clustered_csv: bool = False,
    ):
    df_main = load_data(data_path)
    df_main = merge_address_columns(df_main)
    df_main = cluster_data(
        df=df_main, 
        name_duplicate_max_distance_threshold=name_duplicate_max_distance_threshold, 
        address_duplicate_max_distance_threshold=address_duplicate_max_distance_threshold, 
        tfidf_ngram_range=(tfidf_ngram_min, tfidf_ngram_max),
        tfidf_use_char_ngrams=tfidf_use_char_ngrams,
        tfidf_min_df=1,
    )
    if save_clustered_csv:
        df_main.to_csv(output_data_path, index=False)

if __name__ == "__main__":
    typer.run(main)