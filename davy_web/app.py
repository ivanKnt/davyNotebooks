from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import subprocess
import sys
import os
import json
import csv

# --- Configuration --- #
# Project root = parent of davy_web
DAVY_PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS_DIR = DAVY_PROJECT_ROOT / "scripts"
PREPROCESSING_SCRIPTS_DIR = SCRIPTS_DIR / "preprocessing_scripts"
POETRY_FILTER_DIR = DAVY_PROJECT_ROOT / "poetry_filter"
TEXT_REUSE_SCRIPTS_DIR = SCRIPTS_DIR / "text_reuse"

PREPROCESSING_OUTPUT_DIR = DAVY_PROJECT_ROOT / "preprocessing"
CLASSIFICATION_OUTPUT_DIR = DAVY_PROJECT_ROOT / "classifications"
POETRY_OUTPUT_DIR = DAVY_PROJECT_ROOT / "poetry_files"
TEXT_REUSE_OUTPUT_DIR = DAVY_PROJECT_ROOT / "results_text_reuse"

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


def run_python_script(script_path: Path, args=None):
    """Run a Python script using the current interpreter (venv-aware)."""
    if not script_path.exists():
        return False, f"Script not found: {script_path}"

    interpreter = os.environ.get('DAVY_SCRIPT_PYTHON') or sys.executable
    command = [interpreter, str(script_path)]
    if args:
        command.extend(args)

    try:
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(DAVY_PROJECT_ROOT),
            env=env,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
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


# --- Helpers: page key normalization --- #
def _trim_leading_zeros(value: str) -> str:
    try:
        return str(int(str(value)))
    except Exception:
        return str(value)


def _candidate_page_keys(requested: str) -> list[str]:
    s = str(requested)
    trimmed = _trim_leading_zeros(s)
    candidates = [s]
    # common paddings
    for width in (2, 3):
        candidates.append(trimmed.zfill(width))
    # always include trimmed
    candidates.append(trimmed)
    # de-dup while preserving order
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
    alg_key = 'ngram' if algorithm == 'ngram' else ('gst' if algorithm == 'gst' else 'tfidf')
    base_dir = TEXT_REUSE_OUTPUT_DIR / f"results_{alg_key}" / f"config_{config_id}"
    if not base_dir.exists():
        return None
    suffix = f"_{alg_key}_results.json"
    expected_notebooks = sorted(notebooks)
    fallback = None
    for p in base_dir.glob(f"*{suffix}"):
        parts = p.stem.split('__nb_')
        if len(parts) == 2:
            nb_part = parts[1]
            if nb_part.endswith(f"_{alg_key}_results"):
                nb_part = nb_part[: -len(f"_{alg_key}_results")]
            nb_list = sorted(nb_part.split('-'))
            if nb_list == expected_notebooks:
                return p
        else:
            fallback = fallback or p
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
    payload = request.get_json(silent=True) or {}
    algorithm = (payload.get('algorithm') or '').lower()
    config_id = payload.get('config_id')
    notebooks = payload.get('notebooks') or []
    filename = payload.get('filename') or 'page_to_text.json'

    if algorithm not in {"ngram", "gst", "tfidf"}:
        return jsonify({"status": "error", "message": "algorithm must be one of: ngram, gst, tfidf"}), 400
    if not isinstance(config_id, int):
        return jsonify({"status": "error", "message": "config_id must be an integer"}), 400
    if not isinstance(notebooks, list) or len(notebooks) != 2:
        return jsonify({"status": "error", "message": "Provide exactly two notebooks in an array"}), 400

    # If results already exist, return them
    existing = _find_results_file(algorithm, config_id, notebooks)
    if existing:
        with open(existing, 'r', encoding='utf-8') as f:
            return jsonify({
                "status": "already_exists",
                "file": str(existing),
                "data": json.load(f)
            }), 200

    # Choose script path
    if algorithm == 'ngram':
        script = TEXT_REUSE_SCRIPTS_DIR / 'ngram_code.py'
    elif algorithm == 'gst':
        script = TEXT_REUSE_SCRIPTS_DIR / 'gst_code.py'
    else:
        script = TEXT_REUSE_SCRIPTS_DIR / 'tf_idf_code.py'

    # Run script for only the two notebooks; scripts will write results with group tag
    args = [
        '--notebooks', ','.join(notebooks),
        '--combo-size', '2',
        '--filenames', filename,
        '--config-id', str(config_id)
    ]
    success, message = run_python_script(script, args=args)
    if not success:
        return jsonify({"status": "error", "message": message}), 500

    # Attempt to locate the expected results file post-run
    res_file = _find_results_file(algorithm, config_id, notebooks)
    if not res_file:
        return jsonify({
            "status": "partial",
            "message": "Run completed but results file not located yet",
            "log": message
        }), 202
    with open(res_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify({"status": "success", "file": str(res_file), "data": data}), 200
if __name__ == "__main__":
    # Use host=0.0.0.0 to allow external access if needed
    app.run(debug=True, port=5000)


