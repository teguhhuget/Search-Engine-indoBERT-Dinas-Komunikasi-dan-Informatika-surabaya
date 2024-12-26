import re
import pandas as pd
import numpy as np
import string
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from symspellpy import SymSpell, Verbosity
import psycopg2

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load typo correction dictionary
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "C:/Users/sutih/Downloads/kamusss.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_typos(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    if suggestions:
        return suggestions[0].term, text if suggestions[0].term != text else text
    return text, text

# AMBIL DATA DARI DATABASE
import mysql.connector
from mysql.connector import Error

# FUNCTION TO LOAD DATA FROM DATABASE
def load_dataset_from_db():
    # Connect to MySQL Database
    conn = mysql.connector.connect(
        host="localhost",       # XAMPP MySQL runs on localhost
        user="root",            # Default username for XAMPP
        password="",            # Default password is blank
        database="test_db"      # Replace with your database name
        )

    # Create cursor object
    cursor = conn.cursor()

    # SQL query to select data from the solution table
    cursor.execute("""
        SELECT 
            id,
            kategori,
            subkategori,
            title,
            contain,
            deleted
        FROM solution
        WHERE deleted IS NULL OR deleted = ''
    """)

    # Fetch data and store it in a list of dictionaries
    data = []
    for row in cursor.fetchall():
        if row[3] and row[4]:  # Ensure 'title' and 'contain' are not NULL
            data.append({
                "id": row[0],
                "kategori": row[1],
                "subkategori": row[2],
                "title": row[3],
                "content": row[4],
                "deleted": row[5]
            })

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return data

# Load dataset from the database
data = load_dataset_from_db()
print(f"Number of dataset loaded after filtering by 'deleted' column: {len(data)}")

# Convert to DataFrame for processing
df = pd.DataFrame(data)

# Load model for semantic search
model = SentenceTransformer('indobenchmark/indobert-large-p2')
document_embeddings = []
for article in data:
    title_embedding = model.encode(preprocess_text(article['title']), convert_to_numpy=True)
    content_embedding = model.encode(preprocess_text(article['content']), convert_to_numpy=True)
    combined_embedding = 0.7 * title_embedding + 0.3 * content_embedding
    document_embeddings.append(combined_embedding)

# Manual search function with sorting by `id`
def search_dataset(keyword, dataset, max_results=30):
    keyword_pattern = rf'\b{re.escape(keyword)}\b'
    results = []
    for entry in dataset:
        if (re.search(keyword_pattern, entry['kategori'], re.IGNORECASE) or
            re.search(keyword_pattern, entry['subkategori'], re.IGNORECASE) or
            re.search(keyword_pattern, entry['title'], re.IGNORECASE) or
            re.search(keyword_pattern, entry['content'], re.IGNORECASE)):
            results.append(entry)
            if len(results) == max_results:
                break

    # Sort results by `id` in ascending order
    sorted_results = sorted(results, key=lambda x: x['id'])

    return sorted_results


def get_combined_ids(query):
    dataset=None
    if dataset is None:
        dataset = data

    # Run manual search
    manual_results = search_dataset(query, dataset)

    # Run semantic search
    semantic_results = search(query)

    combined_ids = []
    seen_ids = set()

    # Collect IDs from manual results, preserving order
    for entry in manual_results:
        if entry['id'] not in seen_ids:
            seen_ids.add(entry['id'])
            combined_ids.append(entry['id'])

    # Collect IDs from semantic results, preserving order and avoiding duplicates
    if semantic_results != 0:
        for result in semantic_results:
            result_id = result['data']['id']
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                combined_ids.append(result_id)

    # Return combined IDs as a list in the original order of discovery
    return combined_ids if combined_ids else 0


# The modified `search()` function with filtering for scores below 0.5
def search(query, top_n=10, top_k=20):
    corrected_query, original_query = correct_typos(query)
    if corrected_query != original_query:
        print(f"Kata yang dikoreksi: '{original_query}' menjadi '{corrected_query}'")

    corrected_query = preprocess_text(corrected_query)
    query_embedding = model.encode(corrected_query, convert_to_numpy=True)
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    top_k_candidates = [{'data': data[i], 'score': similarities[i]} for i in top_k_indices]
    top_k_candidates = sorted(top_k_candidates, key=lambda x: x['score'], reverse=True)

    # Filter out results with scores below 0.5
    filtered_candidates = [candidate for candidate in top_k_candidates if candidate['score'] >= 0.5]
    results = filtered_candidates[:top_n]

    # Return "0" if the top result's score is below 0.7
    if results and results[0]['score'] < 0.7:
        return 0

    first_kategori = results[0]['data']['kategori'].lower() if results else None
    filtered_results = [result for result in results if result['data']['kategori'].lower() == first_kategori]
    return filtered_results


# Main logic remains the same for terminal usage
if __name__ == "__main__":
    while True:
        user_query = input("Masukkan kueri pencarian (ketik 'exit' atau 'quit' untuk keluar): ")
        if user_query.lower() in ['exit', 'quit']:
            print("Terima kasih telah menggunakan program pencarian!")
            break

        # Run manual search first
        manual_results = search_dataset(user_query, data)
        
        # Run semantic search next
        semantic_results = search(user_query)

        combined_ids = set()
        all_results = []

        # Collect manual results first
        for entry in manual_results:
            if entry['id'] not in combined_ids:
                combined_ids.add(entry['id'])
                all_results.append(entry)

        # Collect semantic results, avoiding duplicates
        if semantic_results != 0:
            for result in semantic_results:
                result_id = result['data']['id']
                if result_id not in combined_ids:
                    combined_ids.add(result_id)
                    all_results.append(result['data'])

        # Display results
        print(f"Total matches found: {len(all_results)}")
        if all_results:
            for entry in all_results:
                print(f"Number: {entry['id']}, Kategori: {entry['kategori']}, Subkategori: {entry['subkategori']}")
                print(f"Title: {entry['title']}")
                print(f"Contain: {entry['content'][:100]}...")
                print("-" * 50)
        else:
            print("0")
