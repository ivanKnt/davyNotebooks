from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import subprocess
import sys
import os
import json
import csv

"""
Configuration and Path Setup
-----------------------------
This section defines all the critical paths for the Davy Notebooks Project.
The structure is organized to keep scripts, inputs, and outputs clearly separated.

Directory Structure:
- scripts/: Contains all processing scripts (preprocessing, text reuse analysis)
- poetry_filter/: Scripts for poetry classification
- preprocessing/: Output directory for extracted and cleaned text data
- classifications/: Output directory for content classification results
- poetry_files/: Output directory for poetry-specific analysis
- results_text_reuse/: Output directory for text reuse detection results

The project root is determined relative to this file's location (davy_web/app.py)
"""
DAVY_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Script directories - where processing scripts live
SCRIPTS_DIR = DAVY_PROJECT_ROOT / "scripts"
PREPROCESSING_SCRIPTS_DIR = SCRIPTS_DIR / "preprocessing_scripts"
POETRY_FILTER_DIR = DAVY_PROJECT_ROOT / "poetry_filter"
TEXT_REUSE_SCRIPTS_DIR = SCRIPTS_DIR / "text_reuse"

# Output directories - where processed data is stored
PREPROCESSING_OUTPUT_DIR = DAVY_PROJECT_ROOT / "preprocessing"
CLASSIFICATION_OUTPUT_DIR = DAVY_PROJECT_ROOT / "classifications"
POETRY_OUTPUT_DIR = DAVY_PROJECT_ROOT / "poetry_files"
TEXT_REUSE_OUTPUT_DIR = DAVY_PROJECT_ROOT / "results_text_reuse"

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the React frontend (different port)


def run_python_script(script_path: Path, args=None):
    """
    Execute a Python script using the active virtual environment's interpreter.
    
    This function is the core mechanism for running all backend processing scripts
    (preprocessing, classification, text reuse analysis). It ensures that:
    1. The correct Python interpreter (from venv) is used
    2. All output is captured for logging and error reporting
    3. Encoding is properly set to UTF-8 to handle historical text characters
    
    Args:
        script_path: Full path to the Python script to execute
        args: Optional list of command-line arguments to pass to the script
    
    Returns:
        tuple: (success: bool, output: str) where output is either stdout or error message
    
    Note: Scripts are executed from the project root to ensure relative paths work correctly.
    """
    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    # Use custom Python path if set (for development), otherwise use the active interpreter
    interpreter = os.environ.get('DAVY_SCRIPT_PYTHON') or sys.executable
    command = [interpreter, str(script_path)]
    if args:
        command.extend(args)

    try:
        # Force UTF-8 encoding to handle special characters in historical texts
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run(
            command,
            capture_output=True,  # Capture both stdout and stderr
            text=True,  # Return strings instead of bytes
            check=True,  # Raise exception on non-zero exit code
            cwd=str(DAVY_PROJECT_ROOT),  # Run from project root
            env=env,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        # Combine stdout and stderr for complete error context
        combined = (e.stdout or '') + '\n' + (e.stderr or '')
        return False, combined.strip()
    except Exception as e:
        return False, str(e)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Davy Notebooks API!"})


# --- Preprocessing Endpoints --- #
@app.route("/api/preprocessing/run", methods=["POST"])
def run_preprocessing():
    script_path = PREPROCESSING_SCRIPTS_DIR / "preprocess_files.py"
    success, message = run_python_script(script_path)
    if success:
        return jsonify({
            "status": "success",
            "message": "Preprocessing started/completed.",
            "output": message
        }), 200
    return jsonify({
        "status": "error",
        "message": "Preprocessing failed.",
        "details": message
    }), 500


@app.route("/api/preprocessing/status", methods=["GET"])
def preprocessing_status():
    exists = PREPROCESSING_OUTPUT_DIR.exists() and any(PREPROCESSING_OUTPUT_DIR.iterdir())
    return jsonify({
        "status": "completed" if exists else "pending",
        "path": str(PREPROCESSING_OUTPUT_DIR)
    }), 200


"""
Page Key Normalization Helpers
-------------------------------
These functions handle the inconsistency in page numbering across different notebooks.
Some notebooks use "1", others use "01" or "001". The frontend might request "1" 
but the data files might have "01" as the key. These helpers try all reasonable 
variations to find the correct page.

This is crucial because failing to match page numbers would result in "page not found" 
errors even when the data exists.
"""

def _trim_leading_zeros(value: str) -> str:
    """
    Remove leading zeros from a page number string.
    Converts "01" -> "1", "001" -> "1", etc.
    
    If the value can't be converted to int (e.g., "1a"), returns it as-is.
    This handles edge cases where page numbers might have suffixes.
    """
    try:
        return str(int(str(value)))
    except Exception:
        return str(value)


def _candidate_page_keys(requested: str) -> list[str]:
    """
    Generate all possible page key variations for a requested page number.
    
    For example, if requested="1", this returns ["1", "01", "001"]
    If requested="01", this returns ["01", "1", "001"]
    
    This allows us to match pages regardless of how they're formatted in the data files.
    The order matters - we try the original format first, then common variations.
    
    Returns:
        List of unique page key candidates, preserving order for efficient lookup
    """
    s = str(requested)
    trimmed = _trim_leading_zeros(s)
    candidates = [s]  # Always try the original format first
    
    # Add common zero-padded versions (2-digit and 3-digit)
    for width in (2, 3):
        candidates.append(trimmed.zfill(width))
    
    # Always include the trimmed (no leading zeros) version
    candidates.append(trimmed)
    
    # Remove duplicates while preserving order (important for performance)
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


# --- Classification Endpoints --- #
@app.route("/api/classification/run", methods=["POST"])
def run_classification():
    script_path = POETRY_FILTER_DIR / "classifyContents.py"
    success, message = run_python_script(script_path)
    if success:
        return jsonify({
            "status": "success",
            "message": "Classification processing initiated successfully.",
            "output": message
        }), 200
    return jsonify({
        "status": "error",
        "message": "Classification processing failed.",
        "details": message
    }), 500


@app.route("/api/classification/notebooks", methods=["GET"])
def get_classification_notebooks():
    notebooks = []
    if CLASSIFICATION_OUTPUT_DIR.exists():
        for notebook_dir in CLASSIFICATION_OUTPUT_DIR.iterdir():
            if notebook_dir.is_dir():
                notebook_id = notebook_dir.name
                class_page_file = notebook_dir / "classifications_page.json"
                if class_page_file.exists():
                    try:
                        with open(class_page_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        notebooks.append({
                            "id": notebook_id,
                            "title": data.get("notebook_title", f"Notebook {notebook_id}"),
                            "consensus": data.get("consensus_book", "unknown")
                        })
                    except Exception as e:
                        notebooks.append({
                            "id": notebook_id,
                            "title": f"Notebook {notebook_id}",
                            "consensus": "error",
                            "error": str(e)
                        })
                else:
                    notebooks.append({
                        "id": notebook_id,
                        "title": f"Notebook {notebook_id}",
                        "consensus": "not processed"
                    })
    return jsonify(sorted(notebooks, key=lambda x: x["id"])), 200


@app.route("/api/classification/notebook/<string:notebook_id>", methods=["GET"])
def get_notebook_classification_data(notebook_id: str):
    notebook_path = CLASSIFICATION_OUTPUT_DIR / notebook_id
    class_page_file = notebook_path / "classifications_page.json"
    if not class_page_file.exists():
        return jsonify({
            "status": "error",
            "message": f"Classification data not found for notebook {notebook_id}."
        }), 404
    try:
        with open(class_page_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error loading classification data for notebook {notebook_id}: {e}"
        }), 500


@app.route("/api/classification/page/<string:notebook_id>/<string:page_number>", methods=["GET"])
def get_page_classification_data(notebook_id: str, page_number: str):
    notebook_class_path = CLASSIFICATION_OUTPUT_DIR / notebook_id
    class_page_file = notebook_class_path / "classifications_page.json"

    notebook_text_path = PREPROCESSING_OUTPUT_DIR / notebook_id
    page_to_text_file = notebook_text_path / "page_to_text.json"

    if not class_page_file.exists() or not page_to_text_file.exists():
        return jsonify({
            "status": "error",
            "message": f"Data not found for notebook {notebook_id} page {page_number}. Ensure preprocessing and classification are run."
        }), 404
    try:
        with open(class_page_file, "r", encoding="utf-8") as f_class:
            class_data = json.load(f_class)
        with open(page_to_text_file, "r", encoding="utf-8") as f_text:
            text_data = json.load(f_text)

        page_classification = {}
        page_text = ""
        resolved_key = None
        for key in _candidate_page_keys(page_number):
            page_classification = class_data.get(key, {})
            page_text = text_data.get(key, "")
            if page_classification or page_text:
                resolved_key = key
                break

        if not page_classification and not page_text:
            return jsonify({
                "status": "error",
                "message": f"Page {page_number} not found in notebook {notebook_id}."
            }), 404

        return jsonify({
            "notebook_id": notebook_id,
            "page_number": _trim_leading_zeros(resolved_key or page_number),
            "text": page_text,
            "classification": page_classification
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error loading page data for {notebook_id}/{page_number}: {e}"
        }), 500


# --- Poetry Classification Endpoints --- #
@app.route("/api/poetry/run", methods=["POST"])
def run_poetry_classification():
    script_path = POETRY_FILTER_DIR / "classifyPoetry.py"
    success, message = run_python_script(script_path)
    if success:
        return jsonify({
            "status": "success",
            "message": "Poetry classification initiated successfully.",
            "output": message
        }), 200
    return jsonify({
        "status": "error",
        "message": "Poetry classification failed.",
        "details": message
    }), 500


@app.route("/api/poetry/notebooks", methods=["GET"])
def get_poetry_notebooks():
    poetry_notebooks_csv = POETRY_OUTPUT_DIR / "overall_poetry_notebooks.csv"
    notebooks = []
    if poetry_notebooks_csv.exists():
        with open(poetry_notebooks_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                notebooks.append(row)
    return jsonify(notebooks), 200


@app.route("/api/poetry/pages", methods=["GET"])
def get_poetry_pages():
    poetry_pages_csv = POETRY_OUTPUT_DIR / "poetry_pages.csv"
    pages = []
    if poetry_pages_csv.exists():
        with open(poetry_pages_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'all_classifications_json' in row and row['all_classifications_json']:
                    try:
                        row['all_classifications'] = json.loads(row.pop('all_classifications_json'))
                    except json.JSONDecodeError:
                        row['all_classifications'] = {}
                # normalize page number for display
                if 'page_number' in row:
                    row['page_number'] = _trim_leading_zeros(row['page_number'])
                pages.append(row)
    return jsonify(pages), 200


@app.route("/api/poetry/pages/<string:notebook_id>", methods=["GET"])
def get_poetry_pages_for_notebook(notebook_id: str):
    poetry_pages_csv = POETRY_OUTPUT_DIR / "poetry_pages.csv"
    results = []
    if poetry_pages_csv.exists():
        with open(poetry_pages_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("notebook_id") == notebook_id:
                    if 'all_classifications_json' in row and row['all_classifications_json']:
                        try:
                            row['all_classifications'] = json.loads(row.pop('all_classifications_json'))
                        except json.JSONDecodeError:
                            row['all_classifications'] = {}
                    if 'page_number' in row:
                        row['page_number'] = _trim_leading_zeros(row['page_number'])
                    results.append(row)
    return jsonify(results), 200


# --- Text Reuse Endpoints --- #
def _list_available_notebooks_for_text_reuse():
    notebooks = []
    if PREPROCESSING_OUTPUT_DIR.exists():
        for d in sorted(PREPROCESSING_OUTPUT_DIR.iterdir()):
            if d.is_dir() and (d / "page_to_text.json").exists():
                notebooks.append(d.name)
    return notebooks


def _text_reuse_configs_for(algorithm: str):
    if algorithm == 'ngram':
        return [
            {"config_id": 2, "description": "2-gram + stemming + no stopwords"},
            {"config_id": 5, "description": "4-gram + unstemmed + with stopwords"},
        ]
    if algorithm == 'gst':
        return [
            {"config_id": 2, "description": "GST min-match-3 + stemming + no stopwords"},
            {"config_id": 3, "description": "GST min-match-4 + unstemmed + with stopwords"},
            {"config_id": 4, "description": "GST min-match-5 + stemming + no stopwords"},
        ]
    if algorithm == 'tfidf':
        return [
            {"config_id": 2, "description": "TF-IDF 1-gram cosine + stemming + no stopwords"},
            {"config_id": 6, "description": "TF-IDF 1–3gram cosine + stemming + no stopwords"},
        ]
    return []


def _find_results_file(algorithm: str, config_id: int, notebooks: list[str]):
    """
    Locate the results file for a specific text reuse analysis.
    
    Results files follow a naming convention:
        page_to_text_<config_name>__nb_<notebook1>-<notebook2>_<alg>_results.json
    
    For example:
        page_to_text_2gram_stemmed_no_stopwords__nb_14e-14g_ngram_results.json
    
    This function parses these filenames to find the exact match for the requested
    algorithm, configuration, and notebook pair, this is done this way  because multiple
    configurations might exist for the same notebooks.
    
    Args:
        algorithm: One of 'ngram', 'gst', or 'tfidf'
        config_id: The configuration ID (e.g., 2, 3, 4)
        notebooks: List of notebook IDs (e.g., ['14e', '14g'])
    
    Returns:
        Path to the results file if found, None otherwise
    """
    alg_key = 'ngram' if algorithm == 'ngram' else ('gst' if algorithm == 'gst' else 'tfidf')
    base_dir = TEXT_REUSE_OUTPUT_DIR / f"results_{alg_key}" / f"config_{config_id}"
    
    if not base_dir.exists():
        return None
    
    suffix = f"_{alg_key}_results.json"
    expected_notebooks = sorted(notebooks)  # Sort for consistent comparison
    fallback = None  # Keep first file as fallback if exact match not found
    
    for p in base_dir.glob(f"*{suffix}"):
        # Parse the notebook pair from filename: __nb_14e-14g
        parts = p.stem.split('__nb_')
        if len(parts) == 2:
            nb_part = parts[1]
            # Remove the algorithm suffix if present
            if nb_part.endswith(f"_{alg_key}_results"):
                nb_part = nb_part[: -len(f"_{alg_key}_results")]
            # Extract and sort the notebook IDs
            nb_list = sorted(nb_part.split('-'))
            # Check if this matches the requested notebooks
            if nb_list == expected_notebooks:
                return p
        else:
            # Store first file as fallback
            fallback = fallback or p
    
    # Return fallback only if no notebooks specified (for legacy support)
    if not notebooks:
        return fallback
    return fallback if fallback and not expected_notebooks else None


@app.route("/api/text-reuse/notebooks", methods=["GET"])
def tr_get_notebooks():
    return jsonify(_list_available_notebooks_for_text_reuse()), 200


@app.route("/api/text-reuse/configs/<string:algorithm>", methods=["GET"])
def tr_get_configs(algorithm: str):
    algorithm = algorithm.lower()
    if algorithm not in {"ngram", "gst", "tfidf"}:
        return jsonify({"status": "error", "message": "Invalid algorithm"}), 400
    return jsonify(_text_reuse_configs_for(algorithm)), 200


@app.route("/api/text-reuse/results/<string:algorithm>/<int:config_id>/<string:notebooks>", methods=["GET"])
def tr_get_results(algorithm: str, config_id: int, notebooks: str):
    algorithm = algorithm.lower()
    nb_list = [n.strip() for n in notebooks.split(',') if n.strip()]
    if len(nb_list) < 2:
        return jsonify({"status": "error", "message": "Provide at least two notebooks, e.g. 14e,14g"}), 400
    res_file = _find_results_file(algorithm, config_id, nb_list)
    if not res_file:
        return jsonify({"status": "not_found", "message": "No results located for given parameters"}), 404
    try:
        with open(res_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({"status": "ok", "file": str(res_file), "data": data}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
@app.route("/api/text-reuse/run", methods=["POST"])
def tr_run():
    """
    Run or retrieve text reuse analysis for a pair of notebooks.
    
    This endpoint handles the most computationally expensive operation in the system.
    Text reuse analysis can take several minutes for large notebooks, so we:
    1. First check if results already exist (avoid re-running)
    2. If not, run the appropriate algorithm script
    3. Return results immediately after computation completes
    
    The frontend should show a loading indicator while this runs.
    
    Request body:
        {
            "algorithm": "ngram" | "gst" | "tfidf",
            "config_id": 2,  // Which configuration to use
            "notebooks": ["14e", "14g"],  // Exactly 2 notebooks
            "filename": "page_to_text.json"  // Optional, defaults to page_to_text.json
        }
    
    Returns:
        - 200: Results found and returned (either existing or newly computed)
        - 202: Script completed but results file not yet located (rare timing issue)
        - 400: Invalid parameters
        - 500: Script execution failed
    """
    payload = request.get_json(silent=True) or {}
    algorithm = (payload.get('algorithm') or '').lower()
    config_id = payload.get('config_id')
    notebooks = payload.get('notebooks') or []
    filename = payload.get('filename') or 'page_to_text.json'

    # Validate parameters
    if algorithm not in {"ngram", "gst", "tfidf"}:
        return jsonify({"status": "error", "message": "algorithm must be one of: ngram, gst, tfidf"}), 400
    if not isinstance(config_id, int):
        return jsonify({"status": "error", "message": "config_id must be an integer"}), 400
    if not isinstance(notebooks, list) or len(notebooks) != 2:
        return jsonify({"status": "error", "message": "Provide exactly two notebooks in an array"}), 400

    # Check if results already exist (avoid expensive re-computation)
    existing = _find_results_file(algorithm, config_id, notebooks)
    if existing:
        with open(existing, 'r', encoding='utf-8') as f:
            return jsonify({
                "status": "already_exists",
                "file": str(existing),
                "data": json.load(f)
            }), 200

    # Select the appropriate analysis script
    if algorithm == 'ngram':
        script = TEXT_REUSE_SCRIPTS_DIR / 'ngram_code.py'
    elif algorithm == 'gst':
        script = TEXT_REUSE_SCRIPTS_DIR / 'gst_code.py'
    else:
        script = TEXT_REUSE_SCRIPTS_DIR / 'tf_idf_code.py'

    # Run the analysis script with the specified parameters
    # The scripts will automatically save results in the correct location
    args = [
        '--notebooks', ','.join(notebooks),
        '--combo-size', '2',  # We only support pairwise comparison for now
        '--filenames', filename,
        '--config-id', str(config_id)  # Tell script which configuration to use
    ]
    success, message = run_python_script(script, args=args)
    if not success:
        return jsonify({"status": "error", "message": message}), 500

    # Try to locate the results file that was just created
    res_file = _find_results_file(algorithm, config_id, notebooks)
    if not res_file:
        # This is rare - the script completed but we can't find the output file yet
        # Could be a timing/filesystem issue. Return 202 to indicate processing complete.
        return jsonify({
            "status": "partial",
            "message": "Run completed but results file not located yet",
            "log": message
        }), 202
    
    # Load and return the newly created results
    with open(res_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify({"status": "success", "file": str(res_file), "data": data}), 200
if __name__ == "__main__":
    # Use host=0.0.0.0 to allow external access if needed
    app.run(debug=True, port=5000)


