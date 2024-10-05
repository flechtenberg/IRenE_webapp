# app.py

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import threading
import uuid
from backend.processing import extract_seed, get_keywords, mock_sampling_process

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')

# Global dictionaries to store progress and results
progress_info = {}
ranked_results = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/extract_keywords', methods=['POST'])
def extract_keywords():
    if 'seedCorpus' not in request.files:
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


def update_progress(sampling_id, outer_iter, selected_keyword, query, match_count):
    """
    Update the progress_info dictionary with the latest iteration details.
    """
    if sampling_id not in progress_info:
        progress_info[sampling_id] = {
            'current_outer_iteration': 0,
            'outer_iterations': 0,
            'queries': [],
            'matches_per_query': [],
            'status': 'running'
        }

    progress_info[sampling_id]['current_outer_iteration'] = outer_iter
    progress_info[sampling_id]['queries'].append(query)
    progress_info[sampling_id]['matches_per_query'].append(match_count)

@app.route('/start_sampling', methods=['POST'])
def start_sampling():
    # Retrieve refined keywords and parameters from the session
    weight_dict = session.get('weight_dict', {})
    threshold = session.get('threshold', 100)  # Default to 100 if not set
    outer_iterations = session.get('iterations', 10)  # Default to 10 if not set

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
        'status': 'running'
    }

    # Store sampling_id in the session
    session['sampling_id'] = sampling_id

    # Define the sampling thread function
    def run_sampling():
        print(f"Starting sampling thread for Sampling ID: {sampling_id}")

        # Define the progress_callback
        def progress_callback(outer_iter, selected_keyword, query, match_count):
            update_progress(sampling_id, outer_iter, selected_keyword, query, match_count)
            print(f"Progress Update - Outer Iteration {outer_iter}: Query='{query}' | Matches={match_count}")

        ranked = mock_sampling_process(weight_dict, threshold, outer_iterations, progress_callback=progress_callback)
        ranked_results[sampling_id] = ranked
        progress_info[sampling_id]['status'] = 'completed'
        print(f"Sampling thread for Sampling ID: {sampling_id} completed.")



    # Start the mock sampling in a separate thread
    thread = threading.Thread(target=run_sampling)
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


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        # Handle settings update
        pass
    return render_template('settings.html')


if __name__ == '__main__':
    app.run(debug=True)
