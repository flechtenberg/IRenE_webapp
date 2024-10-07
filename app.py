# app.py

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import os
import threading
import uuid
import json
from werkzeug.utils import secure_filename
from backend.processing import (extract_seed, get_keywords, scopus_sampling_process)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')
app.config['MAX_CONTENT_LENGTH'] = 2 * 16 * 1024 * 1024  # 32 MB

# Global dictionaries to store progress and results
progress_info = {}
ranked_results = {}

# Allowed extensions for API key upload
ALLOWED_EXTENSIONS = {'json'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    # Check if the user has uploaded both seedCorpus and scopusApiKey
    if 'seedCorpus' not in request.files or 'scopusApiKey' not in request.files:
        return redirect(url_for('index'))

    files = request.files.getlist('seedCorpus')
    threshold = request.form.get('threshold', default=100, type=int)
    num_keywords = request.form.get('num_keywords', default=20, type=int)
    iterations = request.form.get('iterations', default=10, type=int)

    # Process the uploaded files and extract seed data
    seed_data = extract_seed(files)

    # Extract keywords using backend function
    keywords = get_keywords(seed_data, num_keywords)

    # Store parameters in session for later use
    session['threshold'] = threshold
    session['iterations'] = iterations

    # Process Scopus API Key
    api_key_file = request.files['scopusApiKey']
    if api_key_file and allowed_file(api_key_file.filename):
        filename = secure_filename(api_key_file.filename)
        try:
            api_key_json = json.load(api_key_file)
            # Validate JSON structure
            if 'apikey' in api_key_json and 'insttoken' in api_key_json:
                session['scopus_api_key'] = api_key_json
                print("Scopus API Key successfully loaded and stored in session.")
            else:
                print("Invalid API Key JSON structure.")
                return "Invalid API Key JSON structure. Please upload a valid API key.", 400
        except json.JSONDecodeError:
            print("Failed to decode JSON from API Key file.")
            return "Invalid JSON file. Please upload a valid API key in JSON format.", 400
    else:
        print("No API Key file uploaded or invalid file type.")
        return "No API Key file uploaded or invalid file type. Please upload a JSON file.", 400

    print(f"Extracted keywords: {[kw['word'] for kw in keywords]}", flush=True)
    print(f"Threshold: {threshold}, Number of Keywords: {num_keywords}", flush=True)
    print(f"Iterations: {iterations}", flush=True)

    return render_template('refine_keywords.html', keywords=keywords)


@app.route('/process_refined_keywords', methods=['POST'])
def process_refined_keywords():
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

    print(f"Processed refined keywords: {weight_dict}", flush=True)

    # Render the auto-submit page to start the sampling process
    return render_template('auto_submit_start_sampling.html')


def update_progress(sampling_id, outer_iter, outer_iterations, query, match_count):
    """
    Update the progress_info dictionary with the latest iteration details.
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
    # Retrieve refined keywords and parameters from the session
    weight_dict = session.get('weight_dict', {})
    threshold = session.get('threshold', 100)  # Default to 100 if not set
    outer_iterations = session.get('iterations', 10)  # Default to 10 if not set

    # Retrieve Scopus API Key from the session
    scopus_api_key = session.get('scopus_api_key', {})
    if not scopus_api_key:
        print("No Scopus API Key found in session.")
        return "Scopus API Key not found. Please upload your API key.", 400

    if not weight_dict:
        print("No keywords available for sampling. Redirecting to index.", flush=True)
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
        print(f"Starting sampling thread for Sampling ID: {sampling_id}")

        # Define the progress_callback
        def progress_callback(outer_iter, query, match_count):
            update_progress(sampling_id, outer_iter, outer_iterations, query, match_count)
            print(f"Progress Update - Outer Iteration {outer_iter}: Query='{query}' | Matches={match_count}")

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
        print(f"Sampling thread for Sampling ID: {sampling_id} completed.")

    # Start the sampling in a separate thread
    thread = threading.Thread(target=run_sampling, daemon=True)
    thread.start()

    return render_template('processing.html', sampling_id=sampling_id)


@app.route('/sampling_progress/<sampling_id>')
def sampling_progress(sampling_id):
    info = progress_info.get(sampling_id, {})
    return jsonify(info)


@app.route('/results')
def results():
    sampling_id = session.get('sampling_id', None)
    if not sampling_id or sampling_id not in ranked_results:
        print(f"No sampling results found for Sampling ID: {sampling_id}", flush=True)
        return redirect(url_for('index'))

    ranked_papers = ranked_results.get(sampling_id, [])

    if not ranked_papers:
        print(f"Ranked papers list is empty for Sampling ID: {sampling_id}", flush=True)

    return render_template('results.html', papers=ranked_papers)


@app.route('/download_results')
def download_results():
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

    # Return the CSV data as an HTTP response with appropriate headers
    return Response(
        output,
        mimetype='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment;filename=results.csv'}
    )



@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        # Handle settings update
        pass
    return render_template('settings.html')


if __name__ == '__main__':
    app.run(debug=True)
