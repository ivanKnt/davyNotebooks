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

The **Davy Notebooks Project** is a digital humanities research project for analyzing the notebooks of Sir Humphry Davy. This repository contains:

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
pip install -r requirements.txt
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

## Data Quality & Known Issues

### 🚨 Important: Current State of the Data

**This section documents real-world data problems we've encountered and what still needs to be fixed.** If you're picking up this project, understanding these issues will save you hours of debugging.

#### Missing and Incomplete Files

Based on the latest file availability scan (see `file_scan_output/scan_summary.txt`), here's what's actually missing:

**Overall Statistics:**
- **Total notebooks scanned**: 134
- **TEI XML files**: 134/134 (100% - all present!)
- **Metadata XML**: 134/134 (100%)
- **Valid text files**: 133/134 (99.3%)
- **Tagged text files**: 133/134 (99.3%)
- **Zooniverse files**: 133/134 (99.3%)
- **Classification files**: 128/134 (95.5%)

**Notebooks with Missing Files:**

1. **Notebook 08** - Multiple missing files:
   - ❌ Missing: `valid_text` (processed text)
   - ❌ Missing: `tagged_text` (annotated text)
   - ❌ Missing: `zoo_files` (Zooniverse transcription data)
   - ❌ Missing: `classifications` (volunteer classification CSV)
   - ✅ Has: TEI XML and metadata
   - **Impact**: Cannot preprocess this notebook at all - no text extraction possible
   - **Action needed**: Obtain Zooniverse export files for notebook 08

2. **gs61, gs62, gs63, gs64, gs65** - Missing classification data:
   - ❌ Missing: `classifications` (volunteer classification CSV)
   - ✅ Has: TEI XML, valid_text, tagged_text, zoo_files, metadata
   - **Impact**: Can preprocess (extract text), but cannot classify content or detect poetry
   - **Action needed**: 
     - Check if these notebooks were classified on Zooniverse
     - If yes, export classification data
     - If no, run a classification campaign for these 5 notebooks

**Why This Matters:**

**Notebook 08:**
- This is a complete gap in the data - we have the TEI file but nothing else
- Text reuse analysis cannot include this notebook
- Any analysis results are missing whatever content notebook 08 contains

**GS Series (gs61-gs65):**
- These are recently added Kresen Kernow notebooks
- We CAN extract text and analyze text reuse
- We CANNOT classify content types (Electrochemistry, Poetry, etc.) or include in poetry analysis
- This limits researcher's ability to understand what these notebooks contain

**Action Items:**

**High Priority:**
1. **Fix notebook 08**:
   - Contact Davy Notebooks Project team about missing files
   - Check if notebook 08 was ever transcribed on Zooniverse
   - If not transcribed, it may need to be added to the transcription queue

2. **Get classifications for gs61-gs65**:
   - Check Zooniverse project for classification data
   - If available, export and add to repository
   - If not available, create classification workflow for these pages

**Medium Priority:**
3. Run `checkFilesAvailability.py` regularly to detect any new missing files
4. Document which notebooks are used in each analysis (e.g., "text reuse analysis of 133 notebooks, excluding 08")

**Low Priority:**
5. Investigate if there are additional notebooks beyond the current 134 that should be added

---

#### Classification Data Inconsistencies

As documented in detail in the preprocessing section, the **volunteer classification data from Zooniverse is highly inconsistent**. Here are the specific problems:

**Problem 1: Multiple Export Formats**

The classification CSV files use at least 5 different formats:
- Some have individual vote rows (`page_num,classification`)
- Some have pre-aggregated percentages in columns
- Some have vote counts (`page_num,classification,count`)
- Some embed JSON strings in CSV cells
- Some use wide format with X marks or 1s

**Impact:** The preprocessing script has complex format detection logic, but new format variants may break it.

**Current Solution:** We handle the formats we've seen, but each new notebook may require code updates.

**Proposed Long-term Solution:**
- Standardize Zooniverse exports to a single format before importing
- Create a data validation script that checks CSV structure before processing
- Document the expected format in a schema file

**Problem 2: Inconsistent Classification Categories**

Volunteers used slightly different labels over time:
- "Poetry" vs "Poem" vs "Poetic content"
- "Electrochemistry" vs "Electro-chemistry" vs "Electrochemical"
- "Other electric" vs "Electricity (not electrochemistry)"
- Free-text entries that don't match any category

**Impact:** `classifyPoetry.py` uses keyword matching which might miss variations.

**Current Solution:** Case-insensitive substring matching catches most variations.

**Proposed Long-term Solution:**
- Create a classification normalization mapping (e.g., {"Poem": "Poetry", "Poetic": "Poetry"})
- Add fuzzy string matching for close variants
- Manually review and standardize the original CSV exports

**Problem 3: Missing Classification Data**

Some notebooks have complete TEI XML but no classification CSV:
- **14e**: Has TEI but classification file is empty
- **Some gs series**: Classification CSV files are missing entirely

**Impact:** These notebooks get preprocessed but classification/poetry scripts skip them.

**Proposed Solution:**
- Re-export classification data from Zooniverse for affected notebooks
- If data was never collected, recruit volunteers to classify these pages
- Mark notebooks without classifications clearly in the documentation

**Problem 4: Incomplete Page Coverage**

Some classification CSVs skip pages:
- Page numbers jump (e.g., pages 1, 2, 5, 8 - where are 3, 4, 6, 7?)
- Early or late pages in notebooks often have fewer volunteer classifications
- Some pages have only 1 volunteer classification (not enough for consensus)

**Impact:** Analysis results are incomplete for these notebooks.

**Proposed Solution:**
- Identify pages with <3 volunteer classifications
- Re-run these pages through Zooniverse to get more votes
- Document which notebooks have incomplete coverage

---

#### Data Homogeneity Issues

**The Problem:**

For reliable analysis, we need **homogeneous data** - meaning all notebooks should have:
- Same metadata structure
- Same classification format
- Same entity annotation standards
- Same TEI XML schema version

Currently, this is NOT the case:

**Issue 1: TEI XML Schema Drift**

Notebooks transcribed at different times use different TEI schemas:
- **Early notebooks (2020-2021)**: Simpler entity markup, fewer `<rs>` tags
- **Later notebooks (2022-2023)**: More detailed entity annotations, consistent `<standOff>` structure
- **Recent notebooks (2024)**: Additional metadata fields

**Impact:** Entity extraction works differently for old vs. new notebooks.

**Proposed Solution:**
- Upgrade old TEI files to match the current schema
- Create a TEI validation script that checks for required elements
- Document which schema version each notebook uses

**Issue 2: Transcription Quality Variations**

Some notebooks have:
- Better character recognition (fewer `[unclear]` markers)
- More consistent spelling
- Complete page coverage vs. partial transcriptions

**Impact:** Text reuse algorithms may miss matches due to transcription errors.

**Proposed Solution:**
- Run spell-checking and normalization on extracted text
- Flag low-confidence transcriptions for review
- Add transcription quality metrics to preprocessing output

**Issue 3: Entity Annotation Inconsistency**

Entity markup varies wildly:
- **Notebook 01a2**: 150 entities annotated
- **Notebook 01a3**: Only 12 entities annotated (similar length!)
- **Notebook 14e**: Extensive chemical annotations
- **Notebook 14f**: Almost no chemical annotations (but discusses chemistry!)

**Impact:** Entity-based analysis is unreliable when some notebooks are under-annotated.

**Proposed Solution:**
- Use NER (Named Entity Recognition) to automatically annotate under-annotated notebooks
- Standardize entity annotation guidelines
- Re-process notebooks with <20 entities

---

#### What Needs to Be Fixed (Priority Order)

**High Priority (Blocks Analysis):**

1. **Standardize classification CSV format**
   - Choose one canonical format
   - Write converter scripts for other formats
   - Re-export all data in standard format

2. **Complete missing notebook series**
   - Obtain TEI files for series 05, 09, remaining 08
   - Process through preprocessing pipeline
   - Verify classifications are available

3. **Fix format detection bugs**
   - Test preprocessing on ALL notebooks
   - Document which notebooks cause errors
   - Add error handling for edge cases

**Medium Priority (Improves Quality):**

4. **Normalize classification categories**
   - Create mapping file: `{"Poem": "Poetry", "Poetic": "Poetry", ...}`
   - Apply normalization in `process_classifications()`
   - Re-run classification aggregation

5. **Validate TEI XML**
   - Write schema validation script
   - Check all notebooks for required elements
   - Report notebooks that don't meet standards

6. **Add transcription quality metrics**
   - Count `[unclear]` markers
   - Calculate confidence scores
   - Flag low-quality transcriptions

**Low Priority (Nice to Have):**

7. **Entity annotation enhancement**
   - Run automated NER on all notebooks
   - Compare with manual annotations
   - Fill gaps in under-annotated notebooks

8. **Cross-notebook consistency checks**
   - Verify entity names are consistent (e.g., "Davy" vs "H. Davy" vs "Humphry Davy")
   - Standardize place names (e.g., "London" vs "london")
   - Create authority files for common entities

---

#### Recommendations for Data Collection Going Forward

If you're adding new notebooks or re-processing existing ones, follow these guidelines:

**For Zooniverse Exports:**
1. Use the latest export format (dictionary with percentages)
2. Ensure at least 5 volunteers classify each page
3. Export with page numbers, not workflow IDs
4. Include timestamp data for audit trails

**For TEI XML:**
1. Use the latest TEI schema (check with Davy Notebooks Project team)
2. Ensure consistent entity markup (`<rs type="person">`, `<rs type="place">`, etc.)
3. Include `<standOff>` section with complete entity metadata
4. Validate XML against schema before committing

**For Preprocessing:**
1. Always run `checkFilesAvailability.py` first
2. Process notebooks in series order (helps spot patterns)
3. Save error logs for debugging
4. Verify outputs before running downstream scripts

**For Documentation:**
1. Document any format variants you encounter
2. Note which notebooks have issues
3. Update this README with new solutions
4. Maintain a changelog of data fixes

---

## Directory Documentation

### `scripts/preprocessing_scripts/` - Where Everything Starts

**📋 What This Section Is About**

This is where everything begins. Before you can classify content, detect poetry, or analyze text reuse, you need clean, structured data. These preprocessing scripts are responsible for taking the raw TEI XML files (which are complex, marked-up historical documents) and volunteer classification CSV exports, then transforming them into simple JSON files that all other parts of the project can easily read and use.

Think of preprocessing as translating the notebooks from their archival format into a format that computers can work with efficiently.

**Important Note:** These scripts are located in `scripts/preprocessing_scripts/` directory.

**Why You Must Start Here:**
- The TEI XML files contain all sorts of markup tags (`<pb>`, `<lb>`, `<rs>`, etc.) that need to be parsed
- Entity annotations (people, places, chemicals) are embedded in the XML and need to be extracted
- Volunteer classification data comes in CSV format and needs to be standardized
- Without these preprocessing outputs, none of the other scripts have anything to analyze

**What These Scripts Produce:**
For each notebook, you'll get four JSON files that contain:
1. Clean text for every page (no XML tags, just the actual words)
2. A catalog of entities mentioned on each page (who, what, where)
3. Complete metadata about all entities in the notebook
4. Normalized volunteer classification votes

---

#### `preprocess_files.py`

**Purpose**: This is the main workhorse script that extracts and processes everything from the TEI XML files.

**How It Works - Step by Step:**

When you run this script, it goes through each notebook you specify and performs several operations:

1. **Parse the TEI XML**: The script opens the TEI file at `items/<notebook_id>/tei/doc` and uses BeautifulSoup to parse the XML structure. TEI (Text Encoding Initiative) is a standardized way of marking up historical texts, and it includes special tags for page breaks, line breaks, entity references, and more.

2. **Extract Text by Page**: As it reads through the XML, it looks for `<pb>` (page break) tags to know when a new page starts. All the text between page breaks gets collected and cleaned. The script respects `<lb>` (line break) tags to preserve the document's structure.

3. **Handle Entities**: When the script encounters `<rs ref="#entity_id">` tags (these mark references to people, places, chemicals, etc.), it looks up the entity details in the `<standOff>` section of the XML and records which entities appear on which pages.

4. **Clean the Text**: Sometimes transcriptions contain accidental duplications like "the the experiment". The `deduplicate_successive_words()` function removes these while being careful not to remove intentional repetitions.

5. **Process Classifications**: The script also reads the volunteer classification CSV file that contains how volunteers labeled each page (e.g., "Electrochemistry", "Poetry", "Lecture notes") and converts this into a structured JSON format.

6. **Save Everything**: Finally, all the extracted data gets saved as four separate JSON files in the `preprocessing/<notebook_id>/` directory.

**🚨 Critical: Handling Messy Classification Data (The Hard Part!)**

This is one of the most important things to understand about the preprocessing pipeline, because we spent considerable time working around data format inconsistencies. **The volunteer classification data does NOT come in a single, clean format** - it varies wildly from notebook to notebook, and we had to build very robust handling to deal with all the variations.

**The Problem We Faced:**

When volunteers classified pages on Zooniverse, the CSV export format changed depending on:
- When the notebook was transcribed (different time periods = different export formats)
- How many volunteers classified each page
- The specific Zooniverse project workflow settings at the time
- Whether project administrators pre-processed the data before exporting
- Different versions of the Zooniverse export tool

This means when you open classification CSV files from different notebooks, you'll find completely different structures!

**Real Format Examples We Encountered:**

**Format 1: Raw Individual Vote Lists**
```csv
page_num,classification
5,Electrochemistry
5,Electrochemistry
5,Lecture notes
5,Philosophy
```
This is the cleanest format - each row is one volunteer's vote for one page.

**Format 2: Pre-aggregated Percentage Columns**
```csv
page_num,Electrochemistry,Lecture notes,Philosophy,Poetry
5,0.50,0.33,0.17,0.00
6,0.20,0.80,0.00,0.00
```
Some notebooks came with percentages already calculated.

**Format 3: Vote Count Columns**
```csv
page_num,classification,vote_count
5,Electrochemistry,3
5,Lecture notes,2
5,Philosophy,1
```
Instead of repeating rows, some files had explicit counts.

**Format 4: JSON Strings Embedded in CSV**
```csv
page_num,classification_data
5,"{\"Electrochemistry\": 3, \"Lecture notes\": 2, \"Philosophy\": 1}"
6,"{\"Poetry\": 5, \"Lecture notes\": 1}"
```
Yes, really - JSON strings inside CSV fields that need double-parsing!

**Format 5: Wide Format with Empty Cells**
```csv
page,Electrochemistry,Lecture notes,Philosophy,Poetry,Geology,Other
5,X,X,,X,,,
6,,X,X,,,,
```
Some used "X" marks, some used "1", some used the classification name repeated.

**Our Solution:**

The `process_classifications()` function includes sophisticated detection and normalization logic:

```python
def process_classifications(notebook_id):
    # 1. Try to open and read the CSV
    # 2. Detect which format it's in by:
    #    - Checking column names
    #    - Looking at first few rows
    #    - Identifying data types (strings vs numbers vs JSON)
    # 3. Parse accordingly
    # 4. Normalize into our standard format
    # 5. Handle edge cases (missing pages, empty classifications, etc.)
```

The output is **always** saved in one of two consistent formats:

**List Format** (when we have individual votes):
```json
{
  "1": ["Electrochemistry", "Lecture notes", "Electrochemistry", "Philosophy"],
  "2": ["Poetry", "Poetry", "Poem"]
}
```

**Dictionary Format** (when we have percentages):
```json
{
  "1": {"Electrochemistry": 0.50, "Lecture notes": 0.33, "Philosophy": 0.17},
  "2": {"Poetry": 0.90, "Poem": 0.10}
}
```

**Why We Keep Both Formats:**

You might wonder why we don't convert everything to one format. The reason is that **both formats contain valuable information**:
- List format preserves the number of individual votes (useful for confidence analysis)
- Dictionary format preserves pre-calculated percentages (useful when source data lost individual votes)

The downstream `classifyContents.py` script is smart enough to handle BOTH formats, so we preserve whatever the source data provided.

**What Happens in classifyContents.py:**

This is where the real magic happens. The `process_page_classifications()` function handles both input formats:

```python
def process_page_classifications(page_data):
    if isinstance(page_data, list):
        # Handle list format: count votes and calculate percentages
        counts = Counter(page_data)
        total = len(page_data)
        percentages = {classification: count/total for classification, count in counts.items()}
        
    elif isinstance(page_data, dict):
        # Handle dict format: already have percentages, just validate
        percentages = page_data
        # Normalize if values are > 1 (might be counts, not percentages)
        if any(v > 1 for v in percentages.values()):
            total = sum(percentages.values())
            percentages = {k: v/total for k, v in percentages.items()}
    
    # Determine consensus (highest percentage)
    consensus = max(percentages.keys(), key=lambda x: percentages[x])
    
    return {
        **percentages,
        "page_consensus": consensus
    }
```

**Edge Cases We Handle:**

1. **Empty pages**: Some pages have no classifications → stored as empty dict `{}`
2. **Tie votes**: When two classifications tie, we pick one deterministically (alphabetical)
3. **Single votes**: Pages with only one volunteer classification → 100% for that classification
4. **Invalid entries**: Sometimes volunteers entered free text or typos → we try to map them to known categories
5. **Missing page numbers**: Some CSVs skip pages → we mark them as unclassified

**Why This Matters for YOU:**

If you're adding a new notebook and get errors like:
- `KeyError` when processing classifications
- `TypeError: 'str' object is not iterable`
- `ValueError: could not convert string to float`

It probably means you've encountered a format variant we haven't handled yet. Here's what to do:

1. **Open the CSV file** in a text editor and examine its structure
2. **Check the first 10-20 rows** to understand the pattern
3. **Look at the `process_classifications()` function** in `preprocess_files.py`
4. **Add detection logic** for your new format variant
5. **Convert it** to either list or dict format
6. **Test thoroughly** with multiple notebooks

This data format handling was honestly one of the hardest technical challenges in the project. The volunteer export data is messy, inconsistent, and changes over time. But now you know why the code might seem more complex than expected - it's because it has to handle all these real-world variations!

**Core Methods and What They Do:**

- `extract_text_from_tei(notebook_id)` - This is the main function that orchestrates the entire extraction process for one notebook. It opens the TEI XML file, parses it, extracts text page by page, handles entities, and saves all the outputs.

- `deduplicate_successive_words(text)` - A utility function that cleans up text by removing consecutive duplicate words. For example, "the the experiment" becomes "the experiment". It's case-sensitive, so "The the" won't be deduplicated.

- `load_entity_metadata(soup)` - Parses the `<standOff>` section of the TEI XML, which contains definitions of all entities (persons, places, chemicals, events, organizations, works) mentioned in the notebook. Returns a dictionary mapping entity IDs to their metadata.

- `extract_page_entities(page_element, entity_map)` - For a given page in the XML, finds all `<rs>` (referencing string) tags and looks up the corresponding entities, organizing them by type (persons, places, etc.).

- `process_classifications(notebook_id)` - Loads the volunteer classification CSV file and converts it into a structured JSON format that's easier for other scripts to work with.

- `main()` - The entry point of the script. Here you'll find a `notebook_ids` list that you can edit to specify which notebooks to process. The function loops through each ID and runs the extraction pipeline.

**Input Files:**
- `items/<notebook_id>/tei/doc` - The TEI XML file containing the marked-up notebook text
- `items/<notebook_id>/transcription/source/classifications` - CSV file with volunteer classification data

**Output Files:**
- `preprocessing/<notebook_id>/page_to_text.json` - Clean text for each page
  ```json
  {
    "1": "Page 1 text content...",
    "2": "Page 2 text content...",
    "3": "Page 3 text content..."
  }
  ```
  
- `preprocessing/<notebook_id>/page_to_entities.json` - Entities mentioned on each page
  ```json
  {
    "1": {},
    "4": {
      "persons": [
        {"name": "Aristotle", "id": "person_138", "description": "..."}
      ],
      "places": [],
      "chemicals": [{"name": "Oxygen", "id": "chem_42"}]
    }
  }
  ```
  
- `preprocessing/<notebook_id>/all_entities_metadata.json` - Complete catalog of all entities in the notebook

- `preprocessing/<notebook_id>/classifications.json` - Structured volunteer classification data

**How to Use It:**
```bash
python scripts/preprocessing_scripts/preprocess_files.py
```

Before running, open the file and edit the `notebook_ids` list in the `main()` function to specify which notebooks you want to process. For example:
```python
notebook_ids = ['01a2', '01a3', '14e']  # Process these three notebooks
```

---

#### `checkFilesAvailability.py`

**Purpose**: A diagnostic utility that scans your entire repository and tells you which notebooks have which files.

**Why This Is Useful:**

When you're working with 100+ notebooks, it's easy to lose track of which ones you've preprocessed, which have classification results, which have been analyzed for text reuse, etc. This script gives you a bird's-eye view of your data pipeline status.

It's especially helpful when:
- You're setting up the project for the first time and want to know which notebooks have source data
- You've run some preprocessing and want to confirm the outputs were created
- You're debugging why a downstream script can't find data for a particular notebook
- You want to generate a report for documentation purposes

**How It Works:**

The script is organized as a class called `DavyNotebooksFileScanner` that encapsulates all the scanning logic.

**Key Methods:**

- `run_file_scan()` - This is the main controller method that orchestrates the entire scan. It discovers notebooks, checks file availability, generates reports, and saves everything to disk.

- `get_notebook_list()` - Scans the `items/` directory to find all notebook folders. Returns a list of notebook IDs sorted alphabetically.

- `check_file_availability(notebooks)` - For each notebook, checks whether specific files exist:
  - TEI XML file (`items/<id>/tei/doc`)
  - Transcription CSV (`items/<id>/transcription/source/classifications`)
  - Preprocessing outputs (`preprocessing/<id>/page_to_text.json`, etc.)
  - Classification results (`classifications/<id>/classifications_page.json`)
  - Poetry files (`poetry_files/*.csv`)
  - Text reuse results (`results_text_reuse/*/`)
  
- `generate_report(results)` - Creates human-readable text summaries showing which notebooks have which files, organized by category.

- `save_results(...)` - Writes the detailed file availability matrix and summary statistics to text files in `file_scan_output/`.

**Output Files:**
- `file_scan_output/file_scan_results.txt` - Detailed report showing file-by-file availability for every notebook
- `file_scan_output/scan_summary.txt` - Summary statistics (e.g., "85 of 100 notebooks have preprocessing outputs")

**How to Use It:**
```bash
python scripts/preprocessing_scripts/checkFilesAvailability.py
```

The script will print a summary to the console and save detailed reports to the `file_scan_output/` directory.

---

### `poetry_filter/` - Classification Scripts (Root Directory)

**📋 What This Section Is About**

**Location:** These scripts are in the root-level `poetry_filter/` directory (NOT in `scripts/`).

After preprocessing extracts the raw data, these scripts make sense of what volunteers labeled in the notebooks. Davy's notebooks aren't just chemistry experiments - they contain poetry, philosophy, lecture notes, personal reflections, and more. Understanding what type of content is on each page helps researchers find what they're looking for and reveals patterns in how Davy organized his work.

This folder contains two scripts that work in sequence:
1. First, `classifyContents.py` takes the volunteer votes and determines consensus
2. Then, `classifyPoetry.py` focuses specifically on finding poetry

**Why Classification Matters:**

When volunteers transcribed these notebooks on Zooniverse, they also classified each page's content. But raw volunteer data can be messy - some volunteers might say "Poetry", others might say "Poem", and you need to aggregate their votes to determine what the page actually contains. That's what these scripts do.

The classification categories include:
- Electrochemistry
- Lecture notes
- Philosophy
- Poetry / Poems
- Geology
- References to other writers/their works
- Other electric (static electricity, electromagnetism, etc.)
- Other (anything that doesn't fit elsewhere)

---

#### `classifyContents.py`

**Purpose**: Transform volunteer classification votes into structured, consensus-based page and notebook classifications.

**The Problem This Solves:**

When volunteers classified pages, each page might have been labeled by multiple people, and they might not always agree. For example, page 5 of notebook 01a2 might have been classified by 7 volunteers like this:
- 4 said "Lecture notes"
- 2 said "Electrochemistry"  
- 1 said "Philosophy"

This script calculates that page 5 is 57% "Lecture notes", 29% "Electrochemistry", and 14% "Philosophy", then determines the consensus is "Lecture notes" (the most common classification).

It also looks at all the pages in a notebook and determines what the notebook as a whole is primarily about.

**How It Uses Preprocessing Data:**

This script directly depends on `preprocess_files.py`. It reads the `preprocessing/<notebook_id>/classifications.json` file that was generated during preprocessing, which contains the normalized volunteer vote data.

After processing, it saves results to `classifications/<notebook_id>/` where both the poetry detection script and the web frontend will look for them.

**Core Functions and What They Do:**

- `load_classifications(notebook_path)` - Opens and reads the `classifications.json` file from the preprocessing directory. Returns the raw volunteer data for all pages in a notebook.

- `process_page_classifications(page_data)` - This is where the magic happens. It handles two possible input formats:
  - **List format**: `['Electrochemistry', 'Lecture notes', 'Electrochemistry']` - raw votes from individual volunteers
  - **Dictionary format**: `{"Electrochemistry": 0.857, "Poetry": 0.143}` - pre-aggregated percentages
  
  **Why two formats?** See the detailed explanation in the preprocessing section above - the source CSV data comes in wildly different formats, and we preserve both list and dict formats in the intermediate JSON files. This function handles BOTH so it works regardless of which format the preprocessing script produced.
  
  The function counts votes (for lists), calculates or normalizes percentages (for both), and determines which classification won (the "consensus"). Returns a standardized dictionary with percentages and consensus label.

- `calculate_book_consensus(pages_data)` - After processing all individual pages, this function looks at the entire notebook and determines what it's primarily about. It counts how many pages have each classification as their consensus and picks the most common one.

- `summarise_page_classifications(page_results)` - Generates human-readable summaries for the text report. For example: "Page 5: Lecture notes (57%), Electrochemistry (29%), Philosophy (14%)".

- `process_all_notebooks(preprocessing_dir, output_dir)` - The orchestrator function. It:
  1. Scans the preprocessing directory to find all notebooks
  2. Loads classifications for each notebook
  3. Processes page-by-page classifications
  4. Calculates notebook-level consensus
  5. Saves both JSON and text outputs

- `save_classifications(notebook_id, data, output_dir)` - Writes two files:
  - A JSON file with structured data (used by other scripts)
  - A TXT file with human-readable summaries (used by researchers)

- `main()` - Entry point that sets up logging and calls `process_all_notebooks()`.

**Input Files:**
- `preprocessing/<notebook_id>/classifications.json` - Raw volunteer classification data (created by `preprocess_files.py`)

**Output Files:**
- `classifications/<notebook_id>/classifications_page.json` - Structured classification data with percentages and consensus labels

  Example structure:
  ```json
  {
    "notebook_title": "Notebook 01A2 (T6, 2023; lecture notes)",
    "consensus_book": "lecture notes",
    "1": {
      "Lecture notes": 0.5,
      "Electrochemistry": 0.333,
      "Other electric": 0.167,
      "page_consensus": "lecture notes"
    },
    "2": {
      "Lecture notes": 0.571,
      "Philosophy": 0.143,
      "page_consensus": "lecture notes"
    }
  }
  ```

- `classifications/<notebook_id>/summary.txt` - Human-readable summary report

  Example content:
  ```
  Notebook 01A2
  Overall Classification: lecture notes
  
  Page 1: Lecture notes (50%), Electrochemistry (33%), Other electric (17%)
  Page 2: Lecture notes (57%), Philosophy (14%)
  ...
  ```

**How to Use It:**
```bash
python poetry_filter/classifyContents.py
```

The script will automatically process all notebooks it finds in the `preprocessing/` directory and save results to `classifications/`.

---

#### `classifyPoetry.py`

**Purpose**: Identify which notebooks and pages contain poetry using the classification data generated by `classifyContents.py`.

**Why This Script Exists:**

The Davy Notebooks Project revealed that Humphry Davy wasn't just a scientist - he was also a poet and literary figure. His notebooks contain poetry alongside scientific observations, and researchers wanted an easy way to find all poetry-related content.

While `classifyContents.py` gives you all classification data, this script specifically filters for poetry and creates dedicated reports that make it easy to answer questions like:
- Which notebooks are primarily poetry notebooks?
- Which pages in any notebook contain poetry?
- How much poetry is there across the entire corpus?

**How It Depends on classifyContents.py:**

This script reads the `classifications/<notebook_id>/classifications_page.json` files that were created by `classifyContents.py`. It cannot run until those files exist. This is why you must run `classifyContents.py` first.

The workflow is:
1. `preprocess_files.py` → creates `preprocessing/<id>/classifications.json`
2. `classifyContents.py` → reads preprocessing output, creates `classifications/<id>/classifications_page.json`
3. `classifyPoetry.py` → reads classifications output, creates `poetry_files/*.csv`

**Core Functions and What They Do:**

- `load_classification_data(notebook_id, classifications_dir)` - Opens and reads the `classifications_page.json` file for a specific notebook. Returns the structured classification data with consensus labels.

- `is_poetry_classification(classification)` - A simple but important keyword matcher. It checks if a classification string contains poetry-related terms like:
  - "poetry"
  - "poem"
  - "verse"
  - "poetic"
  
  The check is case-insensitive, so "Poetry", "POETRY", and "poetry" all match. It also works with compound labels like "Poetry and Philosophy".

- `extract_poetry_notebooks_and_pages(classifications_dir)` - The main analysis function. It:
  1. Scans all notebooks in the classifications directory
  2. For each notebook, checks if the overall consensus is poetry (making it a "poetry notebook")
  3. Also checks each individual page to find any page with poetry, even in non-poetry notebooks
  4. Returns two lists:
     - Poetry notebooks (where the whole notebook is primarily poetry)
     - Poetry pages (every page that contains poetry, grouped by notebook)

- `save_results_to_csv(poetry_pages, poetry_notebooks, output_dir)` - Exports the findings as CSV files for easy analysis in spreadsheet programs or data science tools.

- `generate_summary_report(poetry_pages, poetry_notebooks, output_dir)` - Creates human-readable text reports with statistics like:
  - Total number of poetry notebooks found
  - Total number of poetry pages found
  - List of poetry pages organized by notebook

- `main()` - Entry point that:
  1. Sets up logging
  2. Calls the extraction function
  3. Saves CSV and text reports
  4. Prints summary statistics to the console

**Input Files:**
- `classifications/<notebook_id>/classifications_page.json` - Classification data with consensus labels (created by `classifyContents.py`)

**Output Files:**
- `poetry_files/poetry_pages.csv` - Machine-readable list of all pages containing poetry
  ```csv
  notebook_id,page_num,classification
  13a,15,Poetry
  14e,42,Poetry
  14e,43,Poem
  ```

- `poetry_files/poetry_pages.txt` - Human-readable list organized by notebook
  ```
  Poetry Pages Found:
  
  Notebook 13a:
    - Page 15: Poetry
  
  Notebook 14e:
    - Page 42: Poetry
    - Page 43: Poem
  ```

- `poetry_files/overall_poetry_notebooks.csv` - Notebooks where the overall consensus is poetry
  ```csv
  notebook_id,notebook_title,consensus
  13a,Notebook 13A (Poetry),poetry
  ```

- `poetry_files/overall_poetry_notebooks.txt` - Human-readable summary with statistics
  ```
  Poetry Notebooks Summary
  
  Total poetry notebooks found: 1
  Total pages with poetry: 3
  
  Poetry Notebooks:
  - 13a: Notebook 13A (Poetry)
  ```

**How to Use It:**
```bash
python poetry_filter/classifyPoetry.py
```

The script will automatically process all notebooks it finds in the `classifications/` directory and save poetry-specific reports to `poetry_files/`.

---

### `scripts/text_reuse/` - Text Reuse Detection Algorithms

**📋 What This Section Is About**

**Location:** These scripts are in `scripts/text_reuse/` directory.

Text reuse detection is all about finding when Davy copied, adapted, or reused text from one notebook to another. This is fascinating for researchers because it reveals:
- How Davy's ideas evolved over time
- When he refined lecture material across multiple iterations
- Connections between notebooks that aren't obvious from dates or titles
- How scientific concepts developed through repeated writing and experimentation

Think of it like plagiarism detection, but instead of catching cheating, we're tracing the evolution of scientific thought!

**Why Multiple Algorithms?**

You'll notice we have three different scripts for text reuse detection. This isn't redundancy - each algorithm has different strengths and weaknesses, and researchers often run all three to get a complete picture:

1. **N-gram Analysis** (`ngram_code.py`) - Fast and good at finding scattered similarities
2. **Greedy String Tiling (GST)** (`gst_code.py`) - Best at finding continuous, exact copying
3. **TF-IDF** (`tf_idf_code.py`) - Best at finding semantic similarity and paraphrasing

**Important Prerequisites:**

All these scripts depend on the preprocessing pipeline. They read the `preprocessing/<notebook_id>/page_to_text.json` files that contain clean text. **You must run `preprocess_files.py` before using any text reuse scripts.**

---

#### `ngram_code.py`

**Purpose**: Detect text reuse by breaking text into overlapping word sequences (n-grams) and comparing them.

**What Are N-grams?**

An n-gram is simply a sequence of n words. For example, from the sentence "the experiment was successful":
- **2-grams (bigrams)**: ["the experiment", "experiment was", "was successful"]
- **3-grams (trigrams)**: ["the experiment was", "experiment was successful"]  
- **4-grams**: ["the experiment was successful"]

By comparing which n-grams appear in multiple notebooks, we can detect text reuse.

**How This Algorithm Works:**

1. **Tokenize**: Break text into individual words
2. **Clean** (optional): Apply stemming (`running` → `run`) and/or remove stopwords (`the`, `and`, `of`, etc.)
3. **Generate N-grams**: Create overlapping word sequences of length n
4. **Compare**: For each pair of pages, calculate how many n-grams they share
5. **Calculate Similarity**: Use the Jaccard coefficient: `shared n-grams / total unique n-grams`
6. **Report**: Save matches above the similarity threshold

**Example:**

Page 5 of notebook 01a2:
> "The experiment with oxygen was successful in demonstrating the principle"

Page 3 of notebook 01a4:
> "The experiment with nitrogen was successful in proving the theory"

With 3-grams, they share:
- "the experiment with"
- "was successful in"

Similarity = `2 shared / 10 total unique` = 0.20 (20% similarity)

**Class: `LibraryBasedNgramDetector`**

This is the main class that encapsulates all n-gram detection logic.

**Core Methods and What They Do:**

- `__init__(n_gram_size, similarity_threshold, use_stemming, remove_stopwords, min_segment_length, min_words)` - Initialize the detector with configuration parameters. This is where you set:
  - `n_gram_size`: How many words per n-gram (2, 3, 4, etc.)
  - `similarity_threshold`: Minimum similarity to report (0.0 to 1.0)
  - `use_stemming`: Whether to reduce words to root forms
  - `remove_stopwords`: Whether to filter common words
  - `min_segment_length`: Minimum characters for text segments
  - `min_words`: Minimum word count for segments

- `load_texts(base_dir, notebooks, filenames)` - Load preprocessed text from the `preprocessing/` directory. Opens `page_to_text.json` for each specified notebook and also loads entity metadata from `page_to_entities.json` for context.

- `preprocess_text_advanced(text)` - Clean and normalize 18th-century historical text. This function:
  - Normalizes whitespace
  - Handles special characters common in historical texts
  - Fixes encoding issues
  - Applies stemming if configured
  - Removes stopwords if configured
  
- `generate_ngrams(text)` - Takes preprocessed text and generates all n-grams. Returns a set of tuples (for efficient comparison).

- `compare_ngrams(ngrams1, ngrams2)` - Calculate Jaccard similarity between two sets of n-grams:
  ```python
  similarity = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
  ```
  
- `detect_reuse_with_context(texts, metadata)` - Main analysis function that:
  1. Compares all possible page pairs
  2. Identifies matches above threshold
  3. Extracts surrounding context for each match
  4. Associates entity mentions with reused passages
  
- `save_results(results, output_dir, config_name)` - Export three types of files:
  - JSON with full similarity data
  - CSV with specific reuse instances
  - TXT with human-readable report

**Configuration Options:**

You configure the detector when creating an instance:

```python
detector = LibraryBasedNgramDetector(
    n_gram_size=2,              # Use 2-grams (bigrams)
    similarity_threshold=0.2,    # Report matches above 20% similarity
    use_stemming=True,          # Apply Porter stemmer
    remove_stopwords=True        # Remove common words
)
```

**Common Configurations:**

- **Fast, broad search**: `n_gram_size=2`, `similarity_threshold=0.1`, `remove_stopwords=True`
- **Precise matching**: `n_gram_size=4`, `similarity_threshold=0.3`, `remove_stopwords=False`
- **Semantic overlap**: `n_gram_size=3`, `similarity_threshold=0.2`, `use_stemming=True`

**Input Files:**
- `preprocessing/<notebook_id>/page_to_text.json` - Clean text per page
- `preprocessing/<notebook_id>/page_to_entities.json` - Entity metadata (optional, for context)

**Output Files:**
- `results_text_reuse/results_ngram/config_X/<filename>_ngram_results.json` - Complete similarity matrix
  ```json
  {
    "01a2_page_5": {
      "01a4_page_3": {
        "similarity": 0.45,
        "shared_ngrams": 42,
        "total_ngrams_1": 120,
        "total_ngrams_2": 115,
        "context": "...experiment with oxygen..."
      }
    }
  }
  ```

- `results_text_reuse/results_ngram/config_X/<filename>_ngram_instances.csv` - Specific instances for easy filtering
  ```csv
  notebook1,page1,notebook2,page2,similarity,shared_ngrams,context1,context2
  01a2,5,01a4,3,0.45,42,"...oxygen...","...nitrogen..."
  ```

- `results_text_reuse/results_ngram/config_X/<filename>_detailed_report.txt` - Human-readable summary with statistics

**How to Use It:**

1. Open the script and find the `main()` function
2. Edit the configuration:
```python
notebooks_to_compare = ['01a2', '01a3', '14e']  # Which notebooks
n_gram_size = 2  # Bigrams
similarity_threshold = 0.2  # 20% minimum
```

3. Run:
```bash
python scripts/text_reuse/ngram_code.py
```

**Performance Tips:**

- Smaller n-grams = faster but more false positives
- Larger n-grams = slower but more precise
- Removing stopwords = fewer n-grams to compare = faster
- Comparing many notebooks = O(n²) comparisons = slow!

For 10 notebooks with ~100 pages each, expect:
- 2-grams, stopwords removed: ~5-10 minutes
- 4-grams, all words: ~30-60 minutes

---

#### `gst_code.py`
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

**📋 OVERVIEW - What This Folder Will Do (Future Implementation)**

This folder is reserved for **next-generation poetry detection** using Large Language Models (LLMs) like GPT-4, Claude, or fine-tuned models. This will be more sophisticated than the keyword-based approach in the root `poetry_filter/` folder.

**Why LLM-Based Detection?**

The current keyword-based approach (`poetry_filter/classifyPoetry.py`) works well but has limitations:
- ❌ Can miss poetry that doesn't get labeled as "Poetry" by volunteers
- ❌ Can't distinguish between actual poems and references to poetry
- ❌ Struggles with mixed content (prose with poetic elements)
- ❌ Doesn't detect quality or style of poetry

**What LLM-Based Detection Will Offer:**
- ✅ Understand context and nuance (is this a poem or a quote about poetry?)
- ✅ Detect poetic prose and literary language
- ✅ Classify poetry by style (Romantic, Classical, etc.)
- ✅ Identify partial poems, excerpts, and paraphrases
- ✅ Recognize poetry even without explicit "Poetry" labels

**Planned Workflow:**
```bash
# Future implementation:
# 1. Load preprocessed text
# 2. Send pages to LLM API with prompt: "Is this poetry?"
# 3. Get structured response with confidence scores
# 4. Compare with keyword-based results for validation
python scripts/poetry_filter/identifyPoem.py
```

**Current Status:** Placeholder files only - not yet implemented

**To Contribute:**
If you want to implement this feature:
1. Choose an LLM API (OpenAI, Anthropic, local model, etc.)
2. Design prompts for poetry detection
3. Implement batch processing with rate limiting
4. Add confidence scoring and validation
5. Create comparison reports vs. keyword method

---

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

