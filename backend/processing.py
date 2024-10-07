# backend/processing.py

import pandas as pd
import numpy as np
import os
import re
import spacy
import PyPDF2
from PyPDF2.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import time
from elsapy.elsclient import ElsClient
from elsapy.elssearch import ElsSearch
from requests.exceptions import RequestException

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

additional_stop_words = {
        'https', 'doi', 'figure', 'table', 'et', 'al',
        'ref', 're', 'ie', 'eg', 'pdf', 'url', 'tables',
        'bibliography', 'references', 'cited', 'literature',
        'et_al', 'and', 'et_al.', 'ref.', 'etc', 'vs', 'via'
    }

def extract_seed(files):
    """
    Extract text from uploaded PDF files and compile into a single string.
    """
    seed_text = ""
    for file in files:
        text = extract_text_from_pdf(file)
        seed_text += text + " "
    return seed_text

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass

def extract_text_from_pdf(file):
    """
    Extract text from a single PDF file and preprocess it.
    """
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + " "

        # Preprocess the extracted text
        text = preprocess_text(text)
        return text

    except PdfReadError as e:
        # Raise a custom error and stop processing
        raise PDFProcessingError(f"Error processing PDF file '{file}': {e}")
    except Exception as e:
        # Raise a custom error for any other general issue
        raise PDFProcessingError(f"An unexpected error occurred while processing '{file}': {e}")

import spacy

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    """
    Preprocess the text by removing URLs, numbers, and references.
    Additionally, use spaCy to filter out certain parts of speech and named entities.
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove numbers
    text = re.sub(r'\b\d+\b', '', text)


    # Use spaCy to process text and remove stop words, punctuation, etc.
    doc = nlp(text)
    cleaned_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.is_alpha:
            # Exclude certain named entities or tokens
            if token.text in additional_stop_words:
                continue
            if token.ent_type_ in {'ORG', 'PERSON', 'GPE', 'DATE'}:
                continue
            if token.pos_ in {'PROPN', 'NUM'}:
                continue
            cleaned_tokens.append(token.lemma_)

    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text


def get_keywords(seed_text, num_keywords):
    """
    Extract top 'num_keywords' keywords using TF-IDF with enhanced preprocessing.
    Returns a list of dictionaries with 'word' and 'weight'.
    """

    # Combine with English stop words from TfidfVectorizer
    combined_stop_words = list(set(ENGLISH_STOP_WORDS).union(additional_stop_words))

    # Initialize TfidfVectorizer with extended stop words and improved tokenization
    vectorizer = TfidfVectorizer(
        stop_words=combined_stop_words,  # Now a list
        max_features=num_keywords,
        token_pattern=r'\b[a-zA-Z]{2,}\b',  # Tokens with at least two letters
        ngram_range=(1, 2),  # Include unigrams and bigrams
        smooth_idf=True,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform([seed_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    # Exclude any bigram containing stop words
    filtered_keywords = []
    for word, weight in keywords:
        words = word.split()
        if any(w in combined_stop_words for w in words):
            continue
        if re.search(r'\d', word):
            continue
        filtered_keywords.append({'word': word, 'weight': round(weight, 2)})
        if len(filtered_keywords) == num_keywords:
            break

    # If not enough keywords after filtering, include more
    if len(filtered_keywords) < num_keywords:
        for word, weight in keywords[len(filtered_keywords):]:
            if re.search(r'\d', word):
                continue
            filtered_keywords.append({'word': word, 'weight': round(weight, 2)})
            if len(filtered_keywords) == num_keywords:
                break

    return filtered_keywords


def load_scopus_data(filepath='data/scopus.csv'):
    """
    Load Scopus article metadata from a CSV file into a Pandas DataFrame.
    """
    try:
        # Build absolute path relative to this script
        base_dir = os.path.dirname(__file__)
        full_path = os.path.join(base_dir, '..', filepath)
        df = pd.read_csv(full_path)
        # Convert 'Title' and 'Abstract' to lowercase for case-insensitive search
        df['Title'] = df['Title'].str.lower()
        df['Abstract'] = df['Abstract'].str.lower()
        print(f"Loaded Scopus data with {df.shape[0]} entries.", flush=True)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'. Please ensure the CSV file exists.", flush=True)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading CSV: {e}", flush=True)
        return pd.DataFrame()


_scopus_df = None


def get_scopus_df():
    """
    Retrieve the cached Scopus DataFrame, loading it if necessary.
    """
    global _scopus_df
    if _scopus_df is None or _scopus_df.empty:
        _scopus_df = load_scopus_data('data/scopus.csv')
    return _scopus_df


def weighted_random_selection(keywords, weights):
    """
    Select a keyword based on weights.

    Parameters:
    - keywords: List of keywords.
    - weights: Corresponding list of weights.

    Returns:
    - Selected keyword or None if no selection possible.
    """
    total_weight = sum(weights)
    if total_weight == 0:
        print("Total weight is zero. No keyword can be selected.", flush=True)
        return None
    probabilities = [w / total_weight for w in weights]
    selected_keyword = np.random.choice(keywords, p=probabilities)

    try:
        selected_index = keywords.index(selected_keyword)
        weight = weights[selected_index]
        print(f"Selected keyword: '{selected_keyword}' with weight {weight}", flush=True)
    except ValueError:
        print(f"Selected keyword '{selected_keyword}' not found in keywords list.", flush=True)
        weight = 0.0

    return selected_keyword


def construct_search_query(selected_keywords):
    """
    Construct a search query string using logical AND.
    """
    return ' AND '.join(selected_keywords)


def execute_search(df, query):
    """
    Execute the search query on the DataFrame.

    Parameters:
    - df: Pandas DataFrame containing Scopus data.
    - query: Search query string.

    Returns:
    - match_count: Number of matching articles.
    - matched_papers: Set of unique paper identifiers (e.g., Links).
    """
    keywords = [kw.strip().lower() for kw in query.split(' AND ')]

    # Apply the filter: articles where all keywords are in Title or Abstract
    mask = df.apply(lambda row: all(
        (kw in row['Title']) or (kw in row['Abstract']) for kw in keywords
    ), axis=1)

    matched_df = df[mask]
    match_count = matched_df.shape[0]
    matched_papers = set(matched_df['Link'])  # Using 'Link' as unique identifier

    print(f"Query '{query}' matched {match_count} articles.", flush=True)
    return match_count, matched_papers


def execute_search_scopus(query, scopus_api_key, threshold=1000):
    """
    Execute the search query using the Scopus API via elsapy.

    Parameters:
    - query: Search query string.
    - scopus_api_key: Dict containing 'apikey' and 'insttoken'.

    Returns:
    - match_count: Number of matching articles.
    - matched_papers: Set of unique paper identifiers (e.g., Links).
    """
    headers = {
        'X-ELS-APIKey': scopus_api_key['apikey'],
        'X-ELS-Insttoken': scopus_api_key['insttoken']
    }

    client = ElsClient(scopus_api_key['apikey'])
    client.inst_token = scopus_api_key['insttoken']

    search_query = f'TITLE-ABS-KEY({query})'
    doc_srch = ElsSearch(search_query, 'scopus')

    try:
        doc_srch.execute(client, get_all=False)
        num_results = doc_srch.tot_num_res
        print(f"Scopus API Query '{query}' matched {num_results} articles.")

        if num_results > 0 and num_results <= threshold:
            doc_srch.execute(client, get_all=True)
            data = doc_srch.results
            matched_papers = []
            for entry in data:
                paper_info = {
                    'scopus_id': entry.get('dc:identifier', '').replace('SCOPUS_ID:', ''),
                    'first_author': entry.get('dc:creator', '-'),
                    'year': entry.get('prism:coverDate', '-')[:4],
                    'title': entry.get('dc:title', '-'),
                    'journal': entry.get('prism:publicationName', '-'),
                    'citations': entry.get('citedby-count', '0'),
                    'open_access': entry.get('openaccess', '-'),
                    'link': entry.get('link', [{}])[2].get('@href', '#')  # Usually the third link is the scopus link
                }
                matched_papers.append(paper_info)
            match_count = len(matched_papers)
            return match_count, matched_papers
        else:
            return num_results, set()

    except RequestException as req_err:
        # Handle issues with network or API request
        print(f"Network or API request error during Scopus API call: {req_err}")
        return 0, set()

    except KeyError as key_err:
        # Handle missing data in the response
        print(f"Missing expected data in Scopus API response: {key_err}")
        return 0, set()

    except Exception as e:
        # General catch-all for other unforeseen errors
        print(f"Unexpected error during Scopus API call: {e}")
        return 0, set()


def mock_sampling_process(weight_dict, threshold, outer_iterations=5, progress_callback=None):
    """
    Perform the mock sampling process with outer and inner iterations.

    Parameters:
    - weight_dict: Dict of keywords and their weights.
    - threshold: The match count threshold.
    - outer_iterations: Number of separate sampling runs.
    - progress_callback: Function to call with progress updates.

    Returns:
    - ranked_papers: List of tuples (paper_link, count), sorted by count descending.
    """
    scopus_df = get_scopus_df()
    if scopus_df.empty:
        print("Scopus DataFrame is empty. Exiting sampling process.")
        return []

    keywords = list(weight_dict.keys())
    weights = list(weight_dict.values())

    paper_rank_counts = {}

    for outer in range(1, outer_iterations + 1):
        print(f"\n--- Outer Iteration {outer} ---")
        search_keywords = []
        while True:
            selected_keyword = weighted_random_selection(keywords, weights)
            if not selected_keyword:
                print("No keyword selected. Ending inner iterations.")
                break
            search_keywords.append(selected_keyword)
            query = construct_search_query(search_keywords)
            match_count, matched_papers = execute_search(scopus_df, query)
            print(f"Added '{selected_keyword}' | Query: '{query}' | Matches: {match_count}")

            # Update progress
            if progress_callback:
                progress_callback(outer, query, match_count)

            if match_count < threshold:
                print(f"Match count {match_count} below threshold {threshold}. Ending inner iterations.")
                break

            # Add a small delay to simulate processing time
            time.sleep(0.1)

        # Record matched papers from the final inner iteration
        for paper in matched_papers:
            paper_rank_counts[paper] = paper_rank_counts.get(paper, 0) + 1
            print(f"Recorded paper: {paper} | Current count: {paper_rank_counts[paper]}")

        # Add a small delay after each outer iteration
        time.sleep(0.1)

    # Create a ranked list sorted by count descending
    ranked_papers = sorted(paper_rank_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Sampling Completed ---")
    return ranked_papers


def scopus_sampling_process(weight_dict, threshold, outer_iterations=5, progress_callback=None, scopus_api_key=None):
    """
    Perform the sampling process using Scopus API with outer and inner iterations.

    Parameters:
    - weight_dict: Dict of keywords and their weights.
    - threshold: The match count threshold.
    - outer_iterations: Number of separate sampling runs.
    - progress_callback: Function to call with progress updates.
    - scopus_api_key: Dict containing 'apikey' and 'insttoken'.

    Returns:
    - ranked_papers: List of dictionaries containing paper information, sorted by occurrences.
    """
    if not scopus_api_key:
        print("No Scopus API Key provided. Cannot perform real sampling.")
        return []

    keywords = list(weight_dict.keys())
    weights = list(weight_dict.values())

    paper_rank_counts = {}

    for outer in range(1, outer_iterations + 1):
        print(f"\n--- Outer Iteration {outer} ---")
        search_keywords = []
        while True:
            selected_keyword = weighted_random_selection(keywords, weights)
            if not selected_keyword:
                print("No keyword selected. Ending inner iterations.")
                break
            # Prevent adding duplicate keywords
            if selected_keyword in search_keywords:
                print(f"Keyword '{selected_keyword}' already in query. Selecting a different keyword.")
                continue
            search_keywords.append(selected_keyword)
            query = construct_search_query(search_keywords)
            match_count, matched_papers = execute_search_scopus(query, scopus_api_key, threshold)
            print(f"Added '{selected_keyword}' | Query: '{query}' | Matches: {match_count}")

            # Update progress
            if progress_callback:
                progress_callback(outer, query, match_count)

            if match_count < threshold:
                print(f"Match count {match_count} below threshold {threshold}. Ending inner iterations.")
                break

            # Add a small delay to simulate processing time
            time.sleep(0.1)

        # Record matched papers from the final inner iteration
        for paper in matched_papers:
            scopus_id = paper['scopus_id']
            if scopus_id in paper_rank_counts:
                paper_rank_counts[scopus_id]['occurrences'] += 1
            else:
                paper['occurrences'] = 1
                paper_rank_counts[scopus_id] = paper

            print(f"Recorded paper: {scopus_id} | Current occurrences: {paper_rank_counts[scopus_id]['occurrences']}")

        # Add a small delay after each outer iteration
        time.sleep(0.1)

    # Create a ranked list sorted by occurrences descending
    ranked_papers = sorted(paper_rank_counts.values(), key=lambda x: x['occurrences'], reverse=True)
    print("\n--- Sampling Completed ---")
    return ranked_papers



# Main block for standalone testing
if __name__ == "__main__":
    # Sample weight_dict based on your example
    weight_dict = {
        'Process': 0.6,
        'Game': 0.4,
        'Coalition': 0.2,
        'Money': 0.1,
        'Power': 0.3,
        'Energy': 0.5,
        'Heat': 0.2,
        'Cold': 0.1,
        'Climate': 0.4,
    }

    # Define a threshold
    threshold = 40  # Adjust as needed

    # Define the number of outer iterations
    outer_iterations = 10  # As per your requirement

    # Execute the sampling process
    print("Starting mock sampling process...\n", flush=True)
    ranked_papers = mock_sampling_process(weight_dict, threshold, outer_iterations)

    # Print the ranked list
    print("\n--- Ranked List of Papers ---", flush=True)
    for paper, count in ranked_papers:
        print(f"Paper: {paper} | Rank Count: {count}", flush=True)
