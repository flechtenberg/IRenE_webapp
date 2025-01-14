# app.py

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, flash
from flask_session import Session
from flask_talisman import Talisman
import os
import threading
import uuid
import json
from backend.processing import (extract_seed, get_keywords, scopus_sampling_process)
import logging
from logging.handlers import TimedRotatingFileHandler

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.mkdir('logs')

app = Flask(__name__)

def setup_logging():
    """
    Configure logging for the Flask application.
    """
    # Create a file handler that logs debug and higher level messages
    log_formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )

    log_file = os.path.join('logs', 'app.log')

    # Create a rotating file handler that creates a new log file every day
    file_handler = TimedRotatingFileHandler(
        log_file, when='midnight', interval=1, backupCount=7
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Add the handler to the app's logger
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

    # Log that logging is set up
    app.logger.info('Logging setup complete.')

csp = {
    'default-src': [
        "'self'",
        'https://stackpath.bootstrapcdn.com',  # Bootstrap CDN
        'https://cdn.jsdelivr.net',           # Any other CDN you're using
    ],
    'script-src': [
        "'self'",
        "'unsafe-inline'",  # Allows inline scripts (use with caution)
        'https://code.jquery.com',  # jQuery CDN
    ],
    'style-src': [
        "'self'",
        "'unsafe-inline'",  # Allows inline styles
        'https://stackpath.bootstrapcdn.com',  # Bootstrap CSS
    ],
    'img-src': [
        "'self'",
        'data:',  # Allows data URIs for images
    ],
    'font-src': [
        "'self'",
        'https://fonts.gstatic.com',  # Google Fonts
    ],
}

# Apply Content Security Policy using Talisman
Talisman(app, content_security_policy=csp)

app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')
app.config['MAX_CONTENT_LENGTH'] = 2 * 16 * 1024 * 1024  # 32 MB
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(app.instance_path, 'sessions')
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = True   # Set to True in production
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
Session(app)

# Call the logging setup function
setup_logging()

# Global dictionaries to store progress and results
progress_info = {}
ranked_results = {}

# Allowed extensions for API key upload
ALLOWED_EXTENSIONS = {'json'}

def allowed_file(filename):
    """
    Check if a filename has an allowed extension.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    Render the home page of the application.

    This route serves the landing page where users can upload their seed corpus
    and Scopus API key. It also clears any existing 'scopus_api_key' from the session
    to ensure a fresh start for each new session.

    Returns:
    - Response: Renders the 'index.html' template.
    """
    session.pop('scopus_api_key', None)
    return render_template('index.html')

def load_scopus_api_key(request):
    """
    Load and validate the Scopus API key file from the request.

    Parameters:
    - request (Request): The Flask request object containing the uploaded files.

    Returns:
    - None: If the API key is successfully loaded and validated.
    - Response: Redirects to the index page with an error message if validation fails.

    This function performs the following steps:
    - Checks if the 'scopusApiKey' file is present in the request and has a valid extension.
    - Attempts to parse the file as JSON.
    - Validates that the JSON contains 'apikey' and 'insttoken' fields.
    - Stores the valid API key in the session.

    If any step fails, it flashes an error message to the user and redirects to the index page.
    """
    api_key_file = request.files.get('scopusApiKey')

    # Check if file is provided and if it's a valid JSON file
    if not api_key_file or not allowed_file(api_key_file.filename):
        flash("No API Key file uploaded or invalid file type. Please upload a valid JSON file.", "error")
        return redirect(url_for('index'))

    try:
        # Try to load the file as JSON
        api_key_json = json.load(api_key_file)

        # Validate that 'apikey' and 'insttoken' are present in the JSON
        if 'apikey' in api_key_json and 'insttoken' in api_key_json:
            # Store the valid API key in the session
            session['scopus_api_key'] = api_key_json
            return None  # No error, continue execution
        else:
            # Missing required fields in the JSON
            flash("Invalid API Key structure: Missing 'apikey' or 'insttoken'.", "error")
            return redirect(url_for('index'))

    except json.JSONDecodeError as e:
        # Handle invalid JSON format
        flash("Invalid JSON file. Please upload a correctly formatted JSON file.", "error")
        return redirect(url_for('index'))

    except Exception as e:
        # Catch any other unexpected errors
        flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('index'))

@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    """
    Process the uploaded seed corpus and Scopus API key to extract initial keywords.

    This route handles the form submission from the index page. It performs the following steps:
    - Validates that both 'seedCorpus' and 'scopusApiKey' files are provided.
    - Retrieves form data for 'threshold', 'num_keywords', and 'iterations'.
    - Calls 'extract_seed' to process the uploaded PDF files and extract text.
    - Calls 'load_scopus_api_key' to validate and store the Scopus API key.
    - Calls 'get_keywords' to extract keywords from the seed data.
    - Logs extracted keywords and parameters.
    - Renders 'refine_keywords.html' template for user to refine the keywords.

    Returns:
    - Response: Renders 'refine_keywords.html' with extracted keywords.
    - Redirect: Redirects to 'index' if validation fails.
    """
    # Check if the user has uploaded both seedCorpus and scopusApiKey
    if 'seedCorpus' not in request.files or 'scopusApiKey' not in request.files:
        flash("Please upload both the seed corpus and Scopus API key.", "error")
        return redirect(url_for('index'))

    files = request.files.getlist('seedCorpus')
    threshold = request.form.get('threshold', default=100, type=int)
    num_keywords = request.form.get('num_keywords', default=20, type=int)
    iterations = request.form.get('iterations', default=10, type=int)

    # Process the uploaded files and extract seed data
    seed_data = extract_seed(files)

    # Process Scopus API Key, and handle the case where it redirects
    api_key_error = load_scopus_api_key(request)
    if api_key_error:
        return api_key_error  # If an error occurred, stop and redirect

    # Extract keywords using backend function
    keywords = get_keywords(seed_data, num_keywords)

    # Store parameters in session for later use
    session['threshold'] = threshold
    session['iterations'] = iterations

    app.logger.info(f"Extracted keywords: {[kw['word'] for kw in keywords]}")
    app.logger.info(f"Threshold: {threshold}, Number of Keywords: {num_keywords}")
    app.logger.info(f"Iterations: {iterations}")

    return render_template('refine_keywords.html', keywords=keywords)

@app.route('/process_refined_keywords', methods=['POST'])
def process_refined_keywords():
    """
    Process the refined keywords and their weights submitted by the user.

    This route handles the form submission from the 'refine_keywords' page. It performs the following steps:
    - Retrieves selected keywords and their corresponding weights from the form data.
    - Handles any new keywords and weights added by the user.
    - Constructs a 'weight_dict' mapping keywords to their weights.
    - Stores 'weight_dict' in the session for use in the sampling process.
    - Logs the processed keywords and weights.
    - Renders 'auto_submit_start_sampling.html' to automatically initiate sampling.

    Returns:
    - Response: Renders 'auto_submit_start_sampling.html' to start sampling.
    """
    # Retrieve selected keywords and weights from the form
    selected_keywords = request.form.getlist('selected_keywords')
    weight_dict = {}

    # Access weights using the composite key 'weights[keyword]'
    for keyword in selected_keywords:
        weight_key = f'weights[{keyword}]'
        weight_value = request.form.get(weight_key, default='0')
        try:
            weight = float(weight_value)
        except ValueError:
            weight = 0.0
        weight_dict[keyword] = weight

    # Handle new keywords added by the user
    new_keywords = request.form.getlist('new_keywords[]')
    new_weights = request.form.getlist('new_weights[]')

    for keyword, weight in zip(new_keywords, new_weights):
        if keyword.strip():  # Ignore empty strings
            try:
                weight_float = float(weight) if weight else 0.0
            except ValueError:
                weight_float = 0.0
            weight_dict[keyword.strip()] = weight_float

    # Store the weight dictionary in the session
    session['weight_dict'] = weight_dict

    app.logger.info(f"Processed refined keywords: {weight_dict}")

    # Render the auto-submit page to start the sampling process
    return render_template('auto_submit_start_sampling.html')

def update_progress(sampling_id, outer_iter, outer_iterations, query, match_count):
    """
    Update the progress information for a given sampling ID during the sampling process.

    Parameters:
    - sampling_id (str): Unique identifier for the sampling process.
    - outer_iter (int): Current outer iteration number.
    - outer_iterations (int): Total number of outer iterations.
    - query (str): The current search query being processed.
    - match_count (int): Number of matches returned by the current query.

    This function updates the 'progress_info' global dictionary with the latest iteration details,
    including the current iteration, query, match count, and appends to the history.

    The 'progress_info' is used to track the sampling progress and is accessed by the
    '/sampling_progress/<sampling_id>' route to provide real-time updates to the user.
    """
    if sampling_id not in progress_info:
        progress_info[sampling_id] = {
            'current_outer_iteration': 0,
            'outer_iterations': 0,
            'current_query': '',
            'last_match_count': 0,
            'status': 'running',
            'history': []
        }
    progress_info[sampling_id]['current_outer_iteration'] = outer_iter
    progress_info[sampling_id]['current_query'] = query
    progress_info[sampling_id]['last_match_count'] = match_count
    progress_info[sampling_id]['outer_iterations'] = outer_iterations

    # Append to history
    progress_info[sampling_id]['history'].append({
        'outer_iteration': outer_iter,
        'query': query,
        'match_count': match_count
    })

@app.route('/start_sampling', methods=['POST'])
def start_sampling():
    """
    Initiate the sampling process using the refined keywords and parameters.

    This route performs the following steps:
    - Retrieves 'weight_dict', 'threshold', and 'iterations' from the session.
    - Validates again the presence of the Scopus API key.
    - Generates a unique 'sampling_id' and initializes progress tracking.
    - Defines and starts a background thread to run the sampling process.
    - Renders 'processing.html' to display sampling progress to the user.

    Returns:
    - Response: Renders 'processing.html' with the 'sampling_id'.
    - Response: Returns an error message if required data is missing.
    """
    # Retrieve refined keywords and parameters from the session
    weight_dict = session.get('weight_dict', {})
    threshold = session.get('threshold', 100)  # Default to 100 if not set
    outer_iterations = session.get('iterations', 10)  # Default to 10 if not set

    # Retrieve Scopus API Key from the session
    scopus_api_key = session.get('scopus_api_key', {})
    if not scopus_api_key:
        app.logger.warning("No Scopus API Key found in session.")
        return "Scopus API Key not found. Please upload your API key.", 400

    if not weight_dict:
        app.logger.warning("No keywords available for sampling. Redirecting to index.", flush=True)
        return redirect(url_for('index'))

    # Generate a unique sampling ID
    sampling_id = str(uuid.uuid4())

    # Initialize progress information
    progress_info[sampling_id] = {
        'current_outer_iteration': 0,
        'outer_iterations': outer_iterations,
        'queries': [],
        'matches_per_query': [],
        'status': 'running',
        'history': []
    }

    # Store sampling_id in the session
    session['sampling_id'] = sampling_id

    # Define the sampling thread function
    def run_sampling():
        app.logger.info(f"Starting sampling thread for Sampling ID: {sampling_id}")

        # Define the progress_callback
        def progress_callback(outer_iter, query, match_count):
            update_progress(sampling_id, outer_iter, outer_iterations, query, match_count)
            #app.logger.info(f"Progress Update - Outer Iteration {outer_iter}: Query='{query}' | Matches={match_count}")

        # Call scopus_sampling_process with the API key
        ranked = scopus_sampling_process(
            weight_dict=weight_dict,
            threshold=threshold,
            outer_iterations=outer_iterations,
            progress_callback=progress_callback,
            scopus_api_key=scopus_api_key
        )

        ranked_results[sampling_id] = ranked
        progress_info[sampling_id]['status'] = 'completed'
        app.logger.info(f"Sampling thread for Sampling ID: {sampling_id} completed.")

    # Start the sampling in a separate thread
    thread = threading.Thread(target=run_sampling, daemon=True)
    thread.start()

    return render_template('processing.html', sampling_id=sampling_id)

@app.route('/sampling_progress/<sampling_id>')
def sampling_progress(sampling_id):
    """
    Provide real-time progress updates for the sampling process.

    Parameters:
    - sampling_id (str): Unique identifier for the sampling process.

    Returns:
    - Response: A JSON object containing the current progress information for the given 'sampling_id'.

    The progress information includes:
    - 'current_outer_iteration': The current outer iteration number.
    - 'outer_iterations': The total number of outer iterations.
    - 'current_query': The most recent query executed.
    - 'last_match_count': The number of matches from the last query.
    - 'status': The current status of the sampling process ('running', 'completed', etc.).
    - 'history': A list of dictionaries recording the history of queries and match counts.
    """
    info = progress_info.get(sampling_id, {})
    return jsonify(info)

@app.route('/results')
def results():
    """
    Display the results of the sampling process to the user.

    This route performs the following steps:
    - Retrieves 'sampling_id' from the session.
    - Validates that sampling results are available for the 'sampling_id'.
    - Retrieves the ranked list of papers from 'ranked_results'.
    - Logs warnings if results are missing.
    - Clears 'scopus_api_key' from the session for security.
    - Renders 'results.html' with the list of ranked papers.

    Returns:
    - Response: Renders 'results.html' with the sampling results.
    - Redirect: Redirects to 'index' if no results are found.
    """
    sampling_id = session.get('sampling_id', None)
    if not sampling_id or sampling_id not in ranked_results:
        app.logger.warning(f"No sampling results found for Sampling ID: {sampling_id}")
        return redirect(url_for('index'))

    ranked_papers = ranked_results.get(sampling_id, [])

    if not ranked_papers:
        app.logger.warning(f"Ranked papers list is empty for Sampling ID: {sampling_id}")

    session.pop('scopus_api_key', None)
    return render_template('results.html', papers=ranked_papers)

@app.route('/download_results')
def download_results():
    """
    Provide a downloadable CSV file containing the sampling results.

    This route performs the following steps:
    - Retrieves 'sampling_id' from the session.
    - Validates that sampling results are available.
    - Constructs a CSV file in memory containing the ranked papers.
    - Sets appropriate headers to prompt the user to download the file.

    Returns:
    - Response: An HTTP response with the CSV data and headers for file download.
    - Redirect: Redirects to 'index' if no results are found.

    The CSV file includes the following fields:
    - 'Occurrences', 'First Author', 'Year', 'Title', 'Journal', 'Citations', 'Open Access', 'Link'.
    """
    sampling_id = session.get('sampling_id', None)
    if not sampling_id or sampling_id not in ranked_results:
        return redirect(url_for('index'))

    papers = ranked_results[sampling_id]

    # Create CSV data
    import csv
    import io

    # Initialize BytesIO and TextIOWrapper without 'with' statement
    si = io.BytesIO()
    text_io = io.TextIOWrapper(si, encoding='utf-8-sig', newline='')

    fieldnames = ['Occurrences', 'First Author', 'Year', 'Title', 'Journal', 'Citations', 'Open Access', 'Link']
    writer = csv.DictWriter(text_io, fieldnames=fieldnames)
    writer.writeheader()
    for paper in papers:
        writer.writerow({
            'Occurrences': paper['occurrences'],
            'First Author': paper['first_author'],
            'Year': paper['year'],
            'Title': paper['title'],
            'Journal': paper['journal'],
            'Citations': paper['citations'],
            'Open Access': paper['open_access'],
            'Link': paper['link']
        })

    # Flush the TextIOWrapper to ensure all data is written to BytesIO
    text_io.flush()
    # Seek to the beginning of BytesIO
    si.seek(0)

    # Read the content from BytesIO
    output = si.getvalue()

    # Close TextIOWrapper and BytesIO if desired
    text_io.close()
    si.close()

    # Return the CSV data as an HTTP response with appropriate headers
    return Response(
        output,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment;filename=results.csv'}
    )

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """
    Handle the settings page where users can view or update application settings.

    Methods:
    - GET: Renders the 'settings.html' template to display current settings.
    - POST: Processes form submissions to update settings (currently a placeholder).

    Returns:
    - Response: Renders 'settings.html' template.
    """
    if request.method == 'POST':
        # Handle settings update
        pass
    return render_template('settings.html')

if __name__ == '__main__':
    app.run(debug=True)
