# backend/processing.py

import numpy as np
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
import logging
import en_core_web_sm

# Get the logger instance
logger = logging.getLogger('app')

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
    Extract text from a list of uploaded PDF files and return a list of preprocessed strings.

    Each element in the returned list corresponds to the text content of one PDF document.
    The text is preprocessed (e.g., cleaned of stopwords and punctuation) for further use.

    Parameters:
    - files (list): A list of PDF files uploaded by the user.

    Returns:
    - list: A list of preprocessed text strings, where each string represents the content of one PDF file.
    """
    seed_texts = []
    for file in files:
        text = extract_text_from_pdf(file)
        seed_texts.append(text)
    return seed_texts


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass

def extract_text_from_pdf(file):
    """
    Extract and preprocess text from a single PDF file.

    This function reads through all the pages of the provided PDF file, extracts the text, and performs preprocessing
    (such as cleaning, lowercasing, and tokenization) on the extracted text for further use in natural language processing tasks.

    Parameters:
    - file: A PDF file object from which text needs to be extracted.

    Returns:
    - str: A single preprocessed string containing the combined text from all pages of the PDF, ready for further analysis.
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
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    # Download the model if it's not available
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    """
    Preprocess the input text by applying various cleaning and filtering steps for NLP tasks.

    This function performs the following operations:
    - Converts the text to lowercase.
    - Removes URLs and numeric values.
    - Utilizes spaCy for lemmatization, and filters out stop words, punctuation, and non-alphabetic tokens.
    - Excludes tokens related to specific named entities (organizations, people, geopolitical entities, dates) and certain parts of speech (proper nouns, numbers).
    - Custom stop words are also removed.

    Parameters:
    - text (str): The raw text to be processed.

    Returns:
    - str: A cleaned and preprocessed string where tokens have been lemmatized and unnecessary elements have been removed.
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


def get_keywords(seed_texts, num_keywords):
    """
    Extract the top 'num_keywords' keywords from a list of documents using TF-IDF (Term Frequency-Inverse Document Frequency).

    This function performs the following steps:
    - Combines custom stop words with standard English stop words.
    - Initializes a TfidfVectorizer to convert the text data into a matrix of TF-IDF features, considering unigrams only.
    - Fits the TF-IDF model to the provided documents (seed_texts).
    - Sums the TF-IDF scores across all documents to rank the importance of each keyword.
    - Filters out numbers and stop words from the resulting keywords.

    Parameters:
    - seed_texts (list of str): A list where each element represents the (filtered) text of a document.
    - num_keywords (int): The number of top keywords to extract.

    Returns:
    - list of dict: A list of dictionaries containing the top keywords and their corresponding TF-IDF scores.
      Each dictionary has two keys:
        - 'word': The keyword.
        - 'weight': The TF-IDF score, rounded to two decimal places.
    """

    # Combine with English stop words from TfidfVectorizer
    combined_stop_words = list(set(ENGLISH_STOP_WORDS).union(additional_stop_words))

    # Initialize TfidfVectorizer with extended stop words and improved tokenization
    vectorizer = TfidfVectorizer(
        stop_words=combined_stop_words,  # Now a list
        max_features=num_keywords,
        token_pattern=r'\b[a-zA-Z]{2,}\b',  # Tokens with at least two letters
        ngram_range=(1, 1),  # Include unigrams and bigrams
        smooth_idf=True,
        sublinear_tf=True
    )

    # Fit and transform the list of documents
    tfidf_matrix = vectorizer.fit_transform(seed_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Sum TF-IDF scores across all documents
    scores = tfidf_matrix.sum(axis=0).A1  # Convert to 1D array
    keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    # Filter and select top keywords
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

    return filtered_keywords


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
        logger.info("Total weight is zero. No keyword can be selected.")
        return None
    probabilities = [w / total_weight for w in weights]
    selected_keyword = np.random.choice(keywords, p=probabilities)

    try:
        selected_index = keywords.index(selected_keyword)
        weight = weights[selected_index]
        print(f"Selected keyword: '{selected_keyword}' with weight {weight}", flush=True)
    except ValueError:
        logger.error(f"Selected keyword '{selected_keyword}' not found in keywords list.", flush=True)

    return selected_keyword


def construct_search_query(selected_keywords):
    """
    Construct a search query string using logical AND.
    """
    return ' AND '.join(selected_keywords)


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
        logger.error(f"Network or API request error during Scopus API call: {req_err}")
        return 0, set()

    except KeyError as key_err:
        # Handle missing data in the response
        logger.error(f"Missing expected data in Scopus API response: {key_err}")
        return 0, set()

    except Exception as e:
        # General catch-all for other unforeseen errors
        logger.error(f"Unexpected error during Scopus API call: {e}")
        return 0, set()


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
        logger.warning("No Scopus API Key provided. Cannot perform real sampling.")
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
                logger.warning("No keyword selected. Ending inner iterations.")
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

            #print(f"Recorded paper: {scopus_id} | Current occurrences: {paper_rank_counts[scopus_id]['occurrences']}")

        # Add a small delay after each outer iteration
        time.sleep(0.1)

    # Create a ranked list sorted by occurrences descending
    ranked_papers = sorted(paper_rank_counts.values(), key=lambda x: x['occurrences'], reverse=True)
    print("\n--- Sampling Completed ---")
    return ranked_papers



# Main block for standalone testing
if __name__ == "__main__":
    1+1
