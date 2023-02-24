import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
import typer

from preprocess_funcs import run_preprocess

def load_data(data_path: str, preprocess_save_path: str = "data/df_preprocessed.pkl") -> pd.DataFrame:
    if data_path.endswith(".pkl"):
        df = pd.read_pickle(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        df = df.fillna("")
        df = run_preprocess(df)
        pd.to_pickle(df,preprocess_save_path)
    return df

def merge_address_columns(
        df: pd.DataFrame,
        columns_to_merge: list = 
            ["Bina Adı", 
             "Dış Kapı/ Blok/Apartman No", 
             "Bulvar/Cadde/Sokak/Yol/Yanyol", 
             "new_adres"],
        new_column_name: str = "merged_address"
    ) -> pd.DataFrame:
    df[new_column_name] = " ".join(df[column] for column in columns_to_merge)
    return df

def cluster_by_column(
        df: pd.DataFrame,
        key_column_name: str,
        cluster_column_name: str,
        similarity_threshold: float = 0.1,
        tfidf_ngram_range: tuple = (1, 1),
        df_mask = None,
) -> pd.DataFrame:
    name_vectorizer = TfidfVectorizer(ngram_range=tfidf_ngram_range)
    name_tfidf_vectors = name_vectorizer.fit_transform(df.loc[df_mask, key_column_name])
    name_dbscan = DBSCAN(eps=similarity_threshold, min_samples=2, metric="cosine").fit(name_tfidf_vectors)
    df.loc[df_mask, cluster_column_name] = name_dbscan.labels_
    return df

def cluster_data(
        df: pd.DataFrame,
        name_similarity_threshold: float = 0.1,
        address_similarity_threshold: float = 0.1,
        tfidf_ngram_range: tuple = (1, 1),
) -> pd.DataFrame:
    def cluster_group(group_df):
        try:
            # only cluster names that are defined
            name_defined_mask = (group_df["Ad-Soyad"] != "")
            group_df = cluster_by_column(
                group_df, 
                "Ad-Soyad", 
                "Ad-Soyad-cluster", 
                name_similarity_threshold, 
                tfidf_ngram_range, 
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
                    group_df, 
                    "merged_address", 
                    "merged_address-cluster", 
                    address_similarity_threshold, 
                    tfidf_ngram_range, 
                    df_mask=cluster_df_mask)
            except ValueError as e:
                # in case of any errors when clustering addresses, assign -1 to every row (means no cluster)
                group_df.loc[cluster_df_mask, 'merged_address-cluster'] = -1
        return group_df
    return df.groupby(["İl", "İlçe", "Mahalle"]).apply(cluster_group)

def main(
        data_path: str = "data/merged_v1_4.csv",
        name_similarity_threshold: float = 0.1,
        address_similarity_threshold: float = 0.1,
        tfidf_ngram_max: int = 1,
        output_data_path: str = "data/clustered_v_1_4.csv",
        save_clustered_csv: bool = False,
    ):
    df_main = load_data(data_path)
    df_main = merge_address_columns(df_main)
    df_main = cluster_data(df_main, name_similarity_threshold, address_similarity_threshold, (1, tfidf_ngram_max))
    if save_clustered_csv:
        df_main.to_csv(output_data_path, index=False)

if __name__ == "__main__":
    typer.run(main)