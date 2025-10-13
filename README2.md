# Davy Notebooks Project - Complete Repository Documentation

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Data Pipeline](#data-pipeline)
- [Directory Documentation](#directory-documentation)
  - [Root Level Scripts](#root-level-scripts)
  - [scripts/](#scripts)
  - [Backend (davy_web/)](#backend-davy_web)
  - [Frontend (davy-frontend/)](#frontend-davy-frontend)
  - [Data Directories](#data-directories)
- [API Endpoints](#api-endpoints)
- [Development Guide](#development-guide)

---

## Project Overview

The **Davy Notebooks Project** is a digital humanities research platform for analyzing the notebooks of Sir Humphry Davy. This repository contains:

1. **Data Processing Pipeline**: Extract and preprocess text from TEI XML files
2. **Content Classification**: Analyze notebook pages by subject matter
3. **Poetry Detection**: Identify poetry content using ML/keyword approaches
4. **Text Reuse Analysis**: Detect text reuse across notebooks using multiple algorithms (N-gram, GST, TF-IDF)
5. **Web Application**: Flask backend + React frontend for exploration and visualization

### Key Technologies
- **Backend**: Python 3.x, Flask, NLTK, scikit-learn
- **Frontend**: React 18, Vite, TailwindCSS, Recharts
- **Data Format**: TEI XML (input), JSON (processed output)
- **Analysis Methods**: NLP, text similarity, classification algorithms

---

## Repository Structure

```
theDavyNotebooksProjectPython/
│
├── items/                          # Source data: TEI XML files, transcriptions, metadata
│   ├── 01a1/, 01a2/, ... gs65/     # One directory per notebook
│   │   ├── tei/doc                 # TEI XML file with marked-up text
│   │   └── transcription/source/   # Volunteer classification data (CSV)
│   │       └── classifications
│
├── preprocessing/                  # Processed text and entity data
│   ├── 01a1/, 01a2/, ... gs65/     # One directory per notebook
│   │   ├── page_to_text.json       # Clean text per page
│   │   ├── page_to_entities.json   # Entities per page (persons, places, chemicals, etc.)
│   │   ├── all_entities_metadata.json  # Complete entity metadata
│   │   └── classifications.json    # Processed volunteer classifications
│
├── classifications/                # Content classification results
│   ├── 01a1/, 01a2/, ... gs65/     # One directory per notebook
│   │   ├── classifications_page.json   # Classification percentages per page
│   │   └── summary.txt             # Human-readable classification summary
│
├── poetry_files/                   # Poetry detection results
│   ├── poetry_pages.csv            # All pages with poetry content
│   ├── poetry_pages.txt            # Human-readable list
│   ├── overall_poetry_notebooks.csv    # Notebooks with overall poetry consensus
│   └── overall_poetry_notebooks.txt
│
├── results_text_reuse/             # Text reuse detection results
│   ├── results_ngram/              # N-gram analysis results
│   │   ├── config_2/, config_5/    # Different configurations
│   │   │   ├── *_ngram_results.json        # Main results
│   │   │   ├── *_ngram_instances.csv       # Specific instances
│   │   │   └── *_detailed_report.txt       # Human-readable report
│   ├── results_gst/                # Greedy String Tiling results
│   │   ├── config_2/, config_3/, config_4/
│   │   │   ├── *_gst_results.json
│   │   │   ├── *_gst_instances.csv
│   │   │   └── *_detailed_report.txt
│   └── results_tfidf/              # TF-IDF similarity results
│       ├── config_2/, config_6/, config_112/
│       │   ├── *_tfidf_results.json
│       │   ├── *_metrics_summary.txt
│       │   └── *_detailed_report.txt
│
├── poetry_filter/                  # Root-level poetry scripts
│   ├── classifyPoetry.py           # Poetry identification (page-level)
│   └── classifyContents.py         # Content classification aggregation
│
├── scripts/                        # Main processing scripts
│   ├── preprocessing_scripts/
│   │   ├── preprocess_files.py
│   │   └── checkFilesAvailability.py
│   ├── text_reuse/
│   │   ├── ngram_code.py
│   │   ├── gst_code.py
│   │   ├── tf_idf_code.py
│   │   └── common_instances.py
│   └── poetry_filter/              # (Future LLM-based poetry detection)
│       ├── identifyPoem.py
│       └── identifyPoem2.py
│
├── davy_web/                       # Flask backend API
│   └── app.py                      # Main Flask application
│
├── davy-frontend/                  # React frontend
│   ├── src/
│   │   ├── pages/                  # Page components
│   │   ├── services/api.js         # API communication layer
│   │   └── App.jsx                 # Main app component
│   ├── package.json
│   └── vite.config.js
│
├── file_scan_output/               # File availability analysis
├── main.py                         # Sample/template file
├── README.md                       # Original project README
└── README2.md                      # This file

```

---

## Getting Started

### Prerequisites

**Python Backend:**
```bash
python 3.8+
pip install flask flask-cors nltk scikit-learn pandas numpy beautifulsoup4
```

**Frontend:**
```bash
node.js 16+
npm or yarn
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd theDavyNotebooksProjectPython
```

2. **Set up Python environment**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # If available, or install manually
```

3. **Download NLTK data** (required for text processing)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

4. **Set up Frontend**
```bash
cd davy-frontend
npm install
```

### Running the Application

**Start Backend:**
```bash
cd davy_web
python app.py
# Backend runs on http://localhost:5001
```

**Start Frontend:**
```bash
cd davy-frontend
npm run dev
# Frontend runs on http://localhost:5173
```

---

## Data Pipeline

The complete data processing pipeline follows this sequence:

```
1. Source Data (items/)
   ↓
2. Preprocessing (scripts/preprocessing_scripts/preprocess_files.py)
   → Extracts text from TEI XML
   → Processes volunteer classifications
   → Outputs to preprocessing/
   ↓
3. Content Classification (poetry_filter/classifyContents.py)
   → Aggregates page classifications
   → Determines consensus
   → Outputs to classifications/
   ↓
4. Poetry Detection (poetry_filter/classifyPoetry.py)
   → Identifies poetry pages/notebooks
   → Outputs to poetry_files/
   ↓
5. Text Reuse Analysis (scripts/text_reuse/*.py)
   → N-gram, GST, TF-IDF algorithms
   → Compares notebooks pairwise or in groups
   → Outputs to results_text_reuse/
```

---

## Directory Documentation

### Root Level Scripts

#### `poetry_filter/`

Contains scripts for identifying and classifying poetry content in the notebooks.

##### `classifyPoetry.py`
**Purpose**: Identify poetry content at page and notebook level

**Core Methods**:
- `load_classification_data(notebook_id, classifications_dir)` - Load classification JSON for a notebook
- `is_poetry_classification(classification)` - Determine if text is poetry using keywords
- `extract_poetry_notebooks_and_pages(classifications_dir)` - Scan all notebooks for poetry
- `save_results_to_csv(poetry_pages, poetry_notebooks, output_dir)` - Export results
- `generate_summary_report(poetry_pages, poetry_notebooks, output_dir)` - Create text report

**Input Files**:
- `classifications/<notebook_id>/classifications_page.json` - Classification percentages per page

**Output Files**:
- `poetry_files/poetry_pages.csv` - All pages containing poetry
- `poetry_files/poetry_pages.txt` - Human-readable list
- `poetry_files/overall_poetry_notebooks.csv` - Notebooks with overall poetry consensus
- `poetry_files/overall_poetry_notebooks.txt` - Summary report

**Usage**:
```bash
python poetry_filter/classifyPoetry.py
```

---

##### `classifyContents.py`
**Purpose**: Process and aggregate volunteer classification data

**Core Methods**:
- `load_classifications(notebook_path)` - Load raw classification data
- `process_page_classifications(page_data)` - Convert volunteer votes to percentages
- `calculate_book_consensus(pages_data)` - Determine overall notebook classification
- `process_all_notebooks(preprocessing_dir, output_dir)` - Batch process all notebooks
- `save_classifications(notebook_id, data, output_dir)` - Save results as JSON and TXT

**Input Files**:
- `preprocessing/<notebook_id>/classifications.json` - Raw volunteer classifications

**Output Files**:
- `classifications/<notebook_id>/classifications_page.json` - Structured classification data
- `classifications/<notebook_id>/summary.txt` - Human-readable summary

**Data Format Example**:
```json
{
  "notebook_title": "Notebook 01A2",
  "consensus_book": "lecture notes",
  "1": {
    "Lecture notes": 0.5,
    "Electrochemistry": 0.333,
    "Other electric": 0.167,
    "page_consensus": "lecture notes"
  }
}
```

**Usage**:
```bash
python poetry_filter/classifyContents.py
```

---

### `scripts/`

Main processing scripts organized by function.

#### `scripts/preprocessing_scripts/`

##### `preprocess_files.py`
**Purpose**: Extract and clean text from TEI XML files, process volunteer classifications

**Core Methods**:
- `extract_text_from_tei(notebook_id)` - Parse TEI XML and extract text by page
- `deduplicate_successive_words(text)` - Remove accidental word duplications
- `load_entity_metadata(soup)` - Extract entity definitions from `<standOff>` section
- `extract_page_entities(page_element, entity_map)` - Map entities to pages
- `process_classifications(notebook_id)` - Load and structure volunteer classification data
- `main()` - Batch process specified notebooks or all available

**Input Files**:
- `items/<notebook_id>/tei/doc` - TEI XML file
- `items/<notebook_id>/transcription/source/classifications` - Volunteer classification CSV

**Output Files**:
- `preprocessing/<notebook_id>/page_to_text.json` - Clean text per page
  ```json
  {
    "1": "Page 1 text content...",
    "2": "Page 2 text content..."
  }
  ```
- `preprocessing/<notebook_id>/page_to_entities.json` - Entities appearing on each page
  ```json
  {
    "1": {},
    "4": {
      "persons": [
        {"name": "Aristotle", "id": "person_138", "description": "..."}
      ],
      "places": [...],
      "chemicals": [...]
    }
  }
  ```
- `preprocessing/<notebook_id>/all_entities_metadata.json` - Complete entity catalog
- `preprocessing/<notebook_id>/classifications.json` - Raw classification data

**TEI Structure Handled**:
- `<pb>` (page breaks) - Define page boundaries
- `<lb>` (line breaks) - Preserve line structure
- `<rs ref="#entity_id">` - Entity references
- `<standOff>` - Entity metadata
- `<note>` - Editorial notes (removed from text)

**Usage**:
```bash
# Process specific notebooks
python scripts/preprocessing_scripts/preprocess_files.py

# Modify notebook_ids in main() to select notebooks
```

**Command-line Arguments**: Currently hardcoded; modify `notebook_ids` list in `main()` function.

---

##### `checkFilesAvailability.py`
**Purpose**: Scan repository for file availability and generate reports

**Core Methods**:
- `get_notebook_list()` - List all notebook directories
- `check_file_availability(notebooks)` - Check which files exist for each notebook
- `generate_report(results)` - Create availability summary

**Input Files**: 
- Scans `items/` directory structure

**Output Files**:
- `file_scan_output/file_scan_results.txt` - Detailed file availability
- `file_scan_output/scan_summary.txt` - Summary statistics

**Usage**:
```bash
python scripts/preprocessing_scripts/checkFilesAvailability.py
```

---

#### `scripts/text_reuse/`

Text reuse detection using multiple algorithms.

##### `ngram_code.py`
**Purpose**: Detect text reuse using N-gram overlap analysis

**Core Methods**:
- `__init__(n_gram_size, similarity_threshold, use_stemming, remove_stopwords)` - Configure detector
- `load_texts(base_dir, notebooks, filenames)` - Load preprocessed text
- `preprocess_text_advanced(text)` - Clean and normalize historical text
- `generate_ngrams(text)` - Create n-gram sequences
- `compare_ngrams(ngrams1, ngrams2)` - Calculate similarity via Jaccard coefficient
- `detect_reuse_with_context(texts, metadata)` - Find reuse instances with surrounding context
- `save_results(results, output_dir, config_name)` - Export JSON, CSV, and text reports

**Algorithm**: 
1. Tokenize text into words
2. Generate overlapping n-grams (e.g., 2-grams: "the experiment" → ["the experiment", "experiment was"])
3. Calculate Jaccard similarity: `|ngrams1 ∩ ngrams2| / |ngrams1 ∪ ngrams2|`
4. Report matches above threshold

**Configuration Options**:
- `n_gram_size`: Size of n-grams (2, 3, 4, etc.)
- `use_stemming`: Apply Porter stemmer to reduce words to roots
- `remove_stopwords`: Filter common words (the, and, of, etc.)
- `similarity_threshold`: Minimum similarity to report (0.0-1.0)

**Input Files**:
- `preprocessing/<notebook_id>/page_to_text.json`
- `preprocessing/<notebook_id>/page_to_entities.json`

**Output Files**:
- `results_text_reuse/results_ngram/config_X/<filename>_ngram_results.json` - Main results
- `results_text_reuse/results_ngram/config_X/<filename>_ngram_instances.csv` - Specific matches
- `results_text_reuse/results_ngram/config_X/<filename>_detailed_report.txt` - Human-readable

**Output Format Example**:
```json
{
  "01a2_page_5": {
    "01a3_page_7": {
      "similarity": 0.67,
      "shared_ngrams": 42,
      "total_ngrams_1": 120,
      "total_ngrams_2": 115
    }
  }
}
```

**Usage**:
```bash
# Edit configuration in main()
python scripts/text_reuse/ngram_code.py
```

**Key Parameters to Adjust**:
```python
detector = LibraryBasedNgramDetector(
    n_gram_size=2,              # 2-grams
    similarity_threshold=0.2,    # 20% similarity minimum
    use_stemming=True,          # Use stemming
    remove_stopwords=True        # Remove stopwords
)
```

---

##### `gst_code.py`
**Purpose**: Detect text reuse using Greedy String Tiling (GST) algorithm

**Core Methods**:
- `compute_similarity(tokens1, tokens2)` - Main GST similarity calculation
- `_greedy_string_tiling(seq1, seq2)` - Core GST algorithm
- `_find_longest_match(seq1, seq2, marked1, marked2)` - Find longest common substring
- `_extend_match(...)` - Extend match from starting position
- `_mark_match(marked1, marked2, match)` - Mark tokens as matched
- `process_notebooks(notebooks, base_dir)` - Batch comparison
- `save_detailed_results(results, output_path)` - Export results

**Algorithm**:
1. Find the longest common substring between two texts
2. Mark matched tokens as "used"
3. Repeat until no matches ≥ min_match_length
4. Calculate coverage = matched_tokens / total_tokens

**GST Advantages**:
- Detects contiguous matches (not just scattered words)
- Robust to small insertions/deletions
- Good for plagiarism detection

**Configuration Options**:
- `min_match_length`: Minimum length of matches (e.g., 3 = at least 3 consecutive tokens)
- `use_stemming`: Reduce words to root forms
- `remove_stopwords`: Filter common words

**Input Files**:
- `preprocessing/<notebook_id>/page_to_text.json`

**Output Files**:
- `results_text_reuse/results_gst/config_X/*_gst_results.json`
- `results_text_reuse/results_gst/config_X/*_gst_instances.csv`
- `results_text_reuse/results_gst/config_X/*_detailed_report.txt`

**Output Format Example**:
```json
{
  "01a2_page_5": {
    "01a4_page_3": {
      "gst_similarity": 0.45,
      "total_match_length": 85,
      "matches_found": 3,
      "matches": [
        {
          "length": 42,
          "pos1": 15,
          "pos2": 22,
          "tokens": ["experiment", "with", "oxygen", ...]
        }
      ]
    }
  }
}
```

**Usage**:
```bash
python scripts/text_reuse/gst_code.py
```

**Key Parameters**:
```python
gst = GreedyStringTiling(min_match_length=3)
```

---

##### `tf_idf_code.py`
**Purpose**: Detect text similarity using TF-IDF (Term Frequency-Inverse Document Frequency) vectors

**Core Methods**:
- `__init__(similarity_threshold, ngram_range, max_features, similarity_metric)` - Configure
- `load_texts(base_dir, notebooks, filenames)` - Load text data
- `preprocess_text_advanced(text)` - Clean and normalize text
- `vectorize_texts(all_texts)` - Convert texts to TF-IDF vectors
- `calculate_similarity_matrix(tfidf_matrix)` - Compute pairwise similarities
- `identify_similar_segments(similarity_matrix, page_ids, threshold)` - Find matches
- `save_results(results, metrics, output_dir, config_name)` - Export

**Algorithm**:
1. **TF (Term Frequency)**: How often does a term appear in a document?
   - `TF(t, d) = (count of t in d) / (total terms in d)`
2. **IDF (Inverse Document Frequency)**: How rare is the term across all documents?
   - `IDF(t) = log(total documents / documents containing t)`
3. **TF-IDF**: `TF * IDF` - High for terms that are common in one doc but rare overall
4. **Similarity**: Cosine similarity between TF-IDF vectors

**Configuration Options**:
- `ngram_range`: Range of n-grams to extract, e.g., `(1, 3)` = 1-grams, 2-grams, 3-grams
- `max_features`: Maximum vocabulary size (10000 = top 10K most important terms)
- `similarity_metric`: 'cosine', 'euclidean', or 'manhattan'
- `similarity_threshold`: Minimum similarity to report

**Advantages**:
- Captures semantic similarity (not just exact matches)
- Robust to paraphrasing
- Standard in information retrieval

**Input Files**:
- `preprocessing/<notebook_id>/page_to_text.json`

**Output Files**:
- `results_text_reuse/results_tfidf/config_X/*_tfidf_results.json`
- `results_text_reuse/results_tfidf/config_X/*_metrics_summary.txt`
- `results_text_reuse/results_tfidf/config_X/*_detailed_report.txt`

**Output Format Example**:
```json
{
  "01a2_page_5": {
    "01a3_page_8": {
      "cosine_similarity": 0.82,
      "shared_vocabulary": 156,
      "unique_terms_1": 203,
      "unique_terms_2": 198
    }
  }
}
```

**Usage**:
```bash
python scripts/text_reuse/tf_idf_code.py
```

**Key Parameters**:
```python
detector = LibraryBasedTFIDFDetector(
    similarity_threshold=0.3,
    ngram_range=(1, 3),
    max_features=10000,
    similarity_metric='cosine'
)
```

---

##### `common_instances.py`
**Purpose**: Compare results across different text reuse algorithms

**Status**: Placeholder/stub file (1 line) - not yet implemented

**Intended Purpose**: 
- Load results from N-gram, GST, and TF-IDF
- Find instances detected by multiple algorithms
- Generate comparison reports

---

#### `scripts/poetry_filter/`

**Status**: Not fully implemented - reserved for future LLM-based poetry detection

##### `identifyPoem.py` and `identifyPoem2.py`
**Purpose**: Identify poetry using Large Language Models (LLMs)

**Current Status**: Placeholder files for future implementation

**Intended Approach**:
- Use GPT/Claude/other LLMs to classify pages as poetry vs. prose
- More sophisticated than keyword matching
- Handle edge cases (poetic prose, excerpts, etc.)

---

### Backend (`davy_web/`)

#### `app.py`
**Purpose**: Flask REST API server for the Davy Notebooks web application

**Core Endpoints**:

##### Preprocessing
- `POST /api/preprocessing/run`
  - Runs `preprocess_files.py`
  - Parameters: `notebook_ids` (array)
  - Returns: Processing status and output

- `GET /api/preprocessing/status/<notebook_id>`
  - Check if preprocessing completed
  - Returns: File availability status

##### Classification
- `POST /api/classification/run`
  - Runs `classifyContents.py`
  - Returns: Classification processing results

- `GET /api/classification/results/<notebook_id>`
  - Load classification results
  - Returns: JSON from `classifications/<notebook_id>/classifications_page.json`

- `GET /api/classification/summary/<notebook_id>`
  - Load classification summary
  - Returns: Text from `classifications/<notebook_id>/summary.txt`

##### Poetry Detection
- `POST /api/poetry/run`
  - Runs `classifyPoetry.py`
  - Returns: Poetry detection results

- `GET /api/poetry/results`
  - Load all poetry detection results
  - Returns: Combined data from `poetry_files/` CSVs

##### Text Reuse
- `POST /api/text-reuse/ngram/run`
  - Runs N-gram analysis
  - Parameters: `notebooks`, `n_gram_size`, `similarity_threshold`, `use_stemming`, `remove_stopwords`

- `POST /api/text-reuse/gst/run`
  - Runs GST analysis
  - Parameters: `notebooks`, `min_match_length`, `use_stemming`, `remove_stopwords`

- `POST /api/text-reuse/tfidf/run`
  - Runs TF-IDF analysis
  - Parameters: `notebooks`, `ngram_range`, `similarity_threshold`, `similarity_metric`

- `GET /api/text-reuse/results/<method>/<config>`
  - Load results for specific method and configuration
  - Methods: `ngram`, `gst`, `tfidf`

##### Data Access
- `GET /api/notebooks/list`
  - List all available notebooks
  - Returns: Array of notebook IDs

- `GET /api/notebooks/<notebook_id>/text`
  - Get preprocessed text for a notebook
  - Returns: JSON from `preprocessing/<notebook_id>/page_to_text.json`

- `GET /api/notebooks/<notebook_id>/entities`
  - Get entity data for a notebook
  - Returns: JSON from `preprocessing/<notebook_id>/page_to_entities.json`

**Core Functions**:
- `run_python_script(script_path, args)` - Execute Python scripts with proper encoding
- `load_json_safe(file_path)` - Load JSON with error handling
- `check_file_exists(file_path)` - Validate file availability

**Configuration**:
```python
DAVY_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = DAVY_PROJECT_ROOT / "scripts"
PREPROCESSING_OUTPUT_DIR = DAVY_PROJECT_ROOT / "preprocessing"
CLASSIFICATION_OUTPUT_DIR = DAVY_PROJECT_ROOT / "classifications"
POETRY_OUTPUT_DIR = DAVY_PROJECT_ROOT / "poetry_files"
TEXT_REUSE_OUTPUT_DIR = DAVY_PROJECT_ROOT / "results_text_reuse"
```

**Running**:
```bash
cd davy_web
python app.py
# Runs on http://localhost:5001
```

---

### Frontend (`davy-frontend/`)

React-based single-page application for exploring Davy Notebooks data.

#### Technology Stack
- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **TailwindCSS** - Styling
- **Axios** - API communication
- **Recharts** - Data visualization
- **Lucide React** - Icons

#### Key Files

##### `src/App.jsx`
Main application component with routing:
```jsx
<Routes>
  <Route path="/" element={<HomePage />} />
  <Route path="/preprocessing" element={<PreprocessingPage />} />
  <Route path="/classification" element={<ClassificationPage />} />
  <Route path="/poetry-traditional" element={<PoetryTraditionalPage />} />
  <Route path="/poetry-advanced" element={<PoetryAdvancedPage />} />
  <Route path="/text-reuse-traditional" element={<TextReuseTraditionalPage />} />
  <Route path="/text-reuse-advanced" element={<TextReuseAdvancedPage />} />
  <Route path="/inventory" element={<InventoryPage />} />
</Routes>
```

##### `src/services/api.js`
Centralized API client:
```javascript
const API_BASE = 'http://localhost:5001/api';

export const preprocessingAPI = {
  runPreprocessing: (notebookIds) => axios.post(...),
  getStatus: (notebookId) => axios.get(...)
};

export const classificationAPI = { ... };
export const poetryAPI = { ... };
export const textReuseAPI = { ... };
```

##### `src/pages/`

**HomePage.jsx**
- Project overview and navigation
- Quick links to all features

**PreprocessingPage.jsx**
- Select notebooks to preprocess
- Trigger TEI XML extraction
- View processing status and logs

**ClassificationPage.jsx**
- View classification results
- Interactive charts showing content distribution
- Page-by-page classification breakdown

**PoetryTraditionalPage.jsx**
- Keyword-based poetry detection
- List notebooks with poetry
- View poetry pages by notebook

**PoetryAdvancedPage.jsx**
- Placeholder for future LLM-based detection

**TextReuseTraditionalPage.jsx**
- Configure and run N-gram, GST, TF-IDF algorithms
- View similarity matrices
- Explore specific reuse instances

**TextReuseAdvancedPage.jsx**
- Placeholder for advanced algorithms

**InventoryPage.jsx**
- File availability overview
- Show which notebooks have been preprocessed
- Diagnostic information

#### Running the Frontend
```bash
cd davy-frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

#### Building for Production
```bash
npm run build
# Output in dist/
```

---

## Data Directories

### `items/`
**Source**: Original TEI XML files from Davy Notebooks Project

**Structure**:
```
items/<notebook_id>/
  ├── tei/doc                         # TEI XML file
  ├── transcription/
  │   └── source/
  │       ├── classifications         # Volunteer classification CSV
  │       └── transcription_*.csv     # Transcription metadata
  ├── manifest/                       # IIIF manifests
  └── config/                         # Notebook configuration
```

**Notebook IDs**: `01a1`, `01a2`, ..., `22c`, `gs61`-`gs65`

---

### `preprocessing/`
**Generated by**: `scripts/preprocessing_scripts/preprocess_files.py`

**Structure**:
```
preprocessing/<notebook_id>/
  ├── page_to_text.json              # Clean text per page
  ├── page_to_entities.json          # Entities per page
  ├── all_entities_metadata.json     # Complete entity catalog
  └── classifications.json           # Raw volunteer classifications
```

**Purpose**: Intermediate processed data used by all downstream analyses

---

### `classifications/`
**Generated by**: `poetry_filter/classifyContents.py`

**Structure**:
```
classifications/<notebook_id>/
  ├── classifications_page.json      # Classification percentages
  └── summary.txt                    # Human-readable summary
```

**Content Categories**:
- Lecture notes
- Electrochemistry
- Philosophy
- Poetry
- Geology
- Refers to other writers/their works
- Other electric (static, electromagnetism, etc.)
- Other (anything that doesn't fit!)

---

### `poetry_files/`
**Generated by**: `poetry_filter/classifyPoetry.py`

**Files**:
- `poetry_pages.csv` - All pages with poetry (columns: notebook_id, page_num, classification)
- `poetry_pages.txt` - Human-readable list
- `overall_poetry_notebooks.csv` - Notebooks with overall poetry consensus
- `overall_poetry_notebooks.txt` - Summary statistics

---

### `results_text_reuse/`
**Generated by**: `scripts/text_reuse/*.py`

**Structure**:
```
results_text_reuse/
  ├── results_ngram/
  │   └── config_X/                   # X = configuration number
  │       ├── *_ngram_results.json
  │       ├── *_ngram_instances.csv
  │       └── *_detailed_report.txt
  ├── results_gst/
  │   └── config_X/
  │       ├── *_gst_results.json
  │       ├── *_gst_instances.csv
  │       └── *_detailed_report.txt
  └── results_tfidf/
      └── config_X/
          ├── *_tfidf_results.json
          ├── *_metrics_summary.txt
          └── *_detailed_report.txt
```

**Configuration Naming**:
- `config_2`: Often 2-gram, stemmed, no stopwords
- `config_3`: Often 3-gram or GST with different settings
- `config_5`: 4-gram or TF-IDF variations

**Filename Patterns**:
- `page_to_text_2gram_stemmed_no_stopwords_*` - Descriptive configuration in name
- `*__nb_14e-14g_*` - Indicates notebooks compared (14e, 14f, 14g)

---

## API Endpoints

### Complete API Reference

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| **Preprocessing** | | | |
| POST | `/api/preprocessing/run` | Run preprocessing pipeline | `notebook_ids: string[]` |
| GET | `/api/preprocessing/status/<id>` | Check preprocessing status | - |
| **Classification** | | | |
| POST | `/api/classification/run` | Run classification aggregation | - |
| GET | `/api/classification/results/<id>` | Get classification results | - |
| GET | `/api/classification/summary/<id>` | Get classification summary | - |
| **Poetry** | | | |
| POST | `/api/poetry/run` | Run poetry detection | - |
| GET | `/api/poetry/results` | Get all poetry results | - |
| **Text Reuse** | | | |
| POST | `/api/text-reuse/ngram/run` | Run N-gram analysis | `notebooks, n_gram_size, similarity_threshold, use_stemming, remove_stopwords` |
| POST | `/api/text-reuse/gst/run` | Run GST analysis | `notebooks, min_match_length, use_stemming, remove_stopwords` |
| POST | `/api/text-reuse/tfidf/run` | Run TF-IDF analysis | `notebooks, ngram_range, similarity_threshold, similarity_metric` |
| GET | `/api/text-reuse/results/<method>/<config>` | Get text reuse results | - |
| **Data** | | | |
| GET | `/api/notebooks/list` | List all notebooks | - |
| GET | `/api/notebooks/<id>/text` | Get notebook text | - |
| GET | `/api/notebooks/<id>/entities` | Get notebook entities | - |

---

## Development Guide

### Adding a New Notebook

1. **Add source files to `items/<notebook_id>/`**
   - TEI XML file at `items/<notebook_id>/tei/doc`
   - Classifications CSV at `items/<notebook_id>/transcription/source/classifications`

2. **Run preprocessing**:
```bash
python scripts/preprocessing_scripts/preprocess_files.py
# Edit notebook_ids list in main() to include new notebook
```

3. **Run classification**:
```bash
python poetry_filter/classifyContents.py
```

4. **Run poetry detection** (if applicable):
```bash
python poetry_filter/classifyPoetry.py
```

5. **Run text reuse analysis** (optional):
```bash
python scripts/text_reuse/ngram_code.py
# etc.
```

### Adding a New Text Reuse Algorithm

1. Create `scripts/text_reuse/your_algorithm.py`
2. Follow the pattern of existing scripts:
   - Load texts from `preprocessing/`
   - Implement comparison logic
   - Save results to `results_text_reuse/results_<algorithm>/`
3. Add API endpoint in `davy_web/app.py`
4. Add frontend interface in `davy-frontend/src/pages/`

### Modifying Classification Categories

1. Update volunteer classification categories in source CSV files
2. Modify `classifyContents.py` if new aggregation logic needed
3. Update `classifyPoetry.py` if poetry detection keywords need changes
4. Update frontend display in `ClassificationPage.jsx`

### Running Tests

```bash
# Backend tests (if available)
python -m pytest

# Frontend tests
cd davy-frontend
npm test
```

### Code Style

- **Python**: PEP 8
- **JavaScript**: ESLint + Prettier (configuration in `davy-frontend/`)

---

## Common Workflows

### Full Pipeline Execution

```bash
# 1. Preprocess all notebooks
python scripts/preprocessing_scripts/preprocess_files.py

# 2. Generate classifications
python poetry_filter/classifyContents.py

# 3. Detect poetry
python poetry_filter/classifyPoetry.py

# 4. Run text reuse analysis (example: N-gram)
python scripts/text_reuse/ngram_code.py
```

### Analyzing Specific Notebooks

1. Edit `main()` in `preprocess_files.py`:
```python
notebook_ids = ['01a2', '01a3', '14e']  # Your selected notebooks
```

2. Run preprocessing and subsequent steps

3. Configure text reuse script to compare only those notebooks:
```python
notebooks_to_compare = ['01a2', '01a3', '14e']
```

### Comparing Text Reuse Methods

1. Run all three algorithms on the same notebooks:
```bash
python scripts/text_reuse/ngram_code.py
python scripts/text_reuse/gst_code.py
python scripts/text_reuse/tf_idf_code.py
```

2. Compare results in `results_text_reuse/` subdirectories

3. Use `common_instances.py` (when implemented) for cross-method comparison

---

## Troubleshooting

### Common Issues

**Issue**: `NLTK data not found`
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Issue**: `Module not found` errors
```bash
pip install -r requirements.txt  # or manually install missing packages
```

**Issue**: Frontend can't connect to backend
- Check backend is running on port 5001
- Check CORS is enabled in `davy_web/app.py`
- Verify `API_BASE` in `davy-frontend/src/services/api.js`

**Issue**: Preprocessing fails for a notebook
- Check TEI XML file exists at `items/<notebook_id>/tei/doc`
- Check XML is well-formed (use XML validator)
- Check for encoding issues (should be UTF-8)

**Issue**: Text reuse takes too long
- Reduce number of notebooks being compared
- Increase `similarity_threshold` to filter more aggressively
- Use smaller `n_gram_size` for faster N-gram analysis

---

## Future Enhancements

### Planned Features
- [ ] LLM-based poetry detection (`scripts/poetry_filter/identifyPoem.py`)
- [ ] Cross-method text reuse comparison (`scripts/text_reuse/common_instances.py`)
- [ ] Entity-based text reuse (compare based on shared entities)
- [ ] Interactive similarity visualizations
- [ ] Batch export functionality
- [ ] User authentication for web app
- [ ] Database backend (currently file-based)

### Research Opportunities
- Compare text reuse methods (N-gram vs. GST vs. TF-IDF)
- Analyze evolution of Davy's writing style over time
- Study entity networks across notebooks
- Correlate poetry presence with other content types
- Temporal analysis of notebook content

---

## Contributing

### Code Contributions
1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Test thoroughly
5. Submit a pull request

### Data Contributions
- Report errors in TEI XML files
- Improve entity annotations
- Validate classification results

---

## Contact & Resources

**Original Project**: [Davy Notebooks Project](https://wp.lancs.ac.uk/davynotebooks)

**Data Source**: 
- [Royal Institution](https://www.rigb.org/)
- [Kresen Kernow](https://kresenkernow.org/)

**Transcription Platform**: [Zooniverse](https://www.zooniverse.org)

---

## License

[Specify license - typically MIT, GPL, or Creative Commons for data]

---

## Acknowledgments

This project is built on the work of the Davy Notebooks Project team and the volunteer transcribers on Zooniverse.

---

**Last Updated**: 2025  
**Version**: 2.0  
**Maintainer**: [Your name/team]

