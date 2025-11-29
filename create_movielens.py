import os
import io
import requests
import zipfile
import pandas as pd
import shutil
import json  # <--- Import necessario per i JSON

# --- CONFIGURAZIONE ---
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
OUTPUT_FOLDER = os.path.join("data", "movielens_small")
TEMP_FOLDER = "temp_ml_download"


def create_directory_structure():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"[OK] Cartella creata: {OUTPUT_FOLDER}")
    else:
        print(f"[INFO] La cartella {OUTPUT_FOLDER} esiste già.")


def download_and_extract():
    print("[INFO] Download del dataset in corso...")
    try:
        r = requests.get(DATASET_URL)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(TEMP_FOLDER)
        print("[OK] Estrazione completata.")
    except Exception as e:
        print(f"[ERROR] Errore durante il download: {e}")
        exit()


def process_data_remapped():
    print("[INFO] Avvio elaborazione e rimappatura (0-based indexing)...")

    # Percorsi sorgente
    ratings_source = os.path.join(TEMP_FOLDER, "ml-latest-small", "ratings.csv")
    movies_source = os.path.join(TEMP_FOLDER, "ml-latest-small", "movies.csv")

    # Percorsi destinazione
    dataset_dest = os.path.join(OUTPUT_FOLDER, "dataset.tsv")
    features_dest = os.path.join(OUTPUT_FOLDER, "item_features.tsv")

    # --- 1. CARICAMENTO DATI ---
    df_ratings = pd.read_csv(ratings_source)
    df_movies = pd.read_csv(movies_source)

    # --- 2. CREAZIONE MAPPE (REMAPPING) ---
    # Identifichiamo tutti gli utenti e gli oggetti unici presenti nelle INTERAZIONI
    unique_users = sorted(df_ratings['userId'].unique())
    unique_items = sorted(df_ratings['movieId'].unique())

    # Creiamo i dizionari: ID Originale (int) -> Nuovo Indice (int)
    # Nota: JSON converte le chiavi in stringhe automaticamente quando salva
    user_map = {int(original_id): index for index, original_id in enumerate(unique_users)}
    item_map = {int(original_id): index for index, original_id in enumerate(unique_items)}

    print(f"      - Utenti mappati: {len(user_map)}")
    print(f"      - Item mappati: {len(item_map)}")

    # --- SALVATAGGIO JSON MAPPING ---
    print("[INFO] Salvataggio file JSON di mapping...")

    # 1. User Mapping
    with open(os.path.join(OUTPUT_FOLDER, "user_map.json"), "w") as f:
        json.dump(user_map, f, indent=4)

    # 2. Item Mapping
    with open(os.path.join(OUTPUT_FOLDER, "item_map.json"), "w") as f:
        json.dump(item_map, f, indent=4)

    # --- 3. ELABORAZIONE DATASET (dataset.tsv) ---
    print("[INFO] Scrittura dataset.tsv rimappato...")

    # Applichiamo la mappa
    df_ratings['userId'] = df_ratings['userId'].map(user_map)
    df_ratings['movieId'] = df_ratings['movieId'].map(item_map)

    # Salviamo solo User, Item, Rating, Timestamp
    # df_ratings[['userId', 'movieId', 'rating', 'timestamp']].to_csv(
    #     dataset_dest, sep='\t', header=False, index=False
    # )
    df_ratings[['userId', 'movieId']].to_csv(
        dataset_dest, sep='\t', header=False, index=False
    )

    # --- 4. ELABORAZIONE FEATURES (item_features.tsv) ---
    print("[INFO] Scrittura item_features.tsv (One-Hot) rimappato...")

    # Filtriamo i film per tenere solo quelli che esistono nei ratings
    df_movies = df_movies[df_movies['movieId'].isin(item_map.keys())].copy()

    # Applichiamo la mappa agli item
    df_movies['mapped_id'] = df_movies['movieId'].map(item_map)

    # Generazione One-Hot Encoding
    dummies = df_movies['genres'].str.get_dummies(sep='|')

    if '(no genres listed)' in dummies.columns:
        dummies = dummies.drop(columns=['(no genres listed)'])

    # --- SALVATAGGIO FEATURES MAPPING (Column Index -> Genre Name) ---
    # È utile sapere che la colonna 0 è "Action", la 1 è "Adventure", ecc.
    features_map = {i: col for i, col in enumerate(dummies.columns)}
    with open(os.path.join(OUTPUT_FOLDER, "features_map.json"), "w") as f:
        json.dump(features_map, f, indent=4)

    # Creazione DataFrame finale
    final_features = pd.concat([df_movies['mapped_id'], dummies], axis=1)

    # Ordiniamo per ID crescente (assicura che la riga 0 sia l'item 0)
    final_features = final_features.sort_values('mapped_id')

    # Salvataggio
    final_features.to_csv(features_dest, sep='\t', header=False, index=False)

    print(f"[OK] Tutti i file generati in: {OUTPUT_FOLDER}")
    print(f"      - dataset.tsv")
    print(f"      - item_features.tsv")
    print(f"      - user_mapping.json")
    print(f"      - item_mapping.json")
    print(f"      - features_mapping.json")


def cleanup():
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)
    print("[OK] Pulizia file temporanei completata.")


if __name__ == "__main__":
    try:
        create_directory_structure()
        download_and_extract()
        process_data_remapped()
    finally:
        cleanup()
        print("\nDataset pronto per Elliot.")