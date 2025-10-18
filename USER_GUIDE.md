# Davy Notebooks Project - Quick Start Guide for Beginners

**👋 Welcome!** This guide will help you get the Davy Notebooks web application running on your computer, even if you're not a programmer.

## What Is This Project?

The Davy Notebooks Project lets you explore the scientific notebooks of Sir Humphry Davy (a famous 18th-century chemist). You can:
- Search for specific types of content (poetry, chemistry experiments, lecture notes, etc.)
- Find where Davy reused or copied text between different notebooks
- See which notebooks contain poetry
- Analyze patterns in his writing

---

## 🚀 Quick Setup (5 Steps)

### Step 1: Install Python

**What's Python?** It's a programming language. The backend (server) of this project is written in Python.

1. Go to [python.org/downloads](https://www.python.org/downloads/)
2. Download Python 3.8 or newer
3. **Important**: During installation, check the box that says "Add Python to PATH"
4. Verify it worked:
   - Open Command Prompt (Windows) or Terminal (Mac/Linux)
   - Type: `python --version`
   - You should see something like `Python 3.11.0`

### Step 2: Install Node.js

**What's Node.js?** It's a JavaScript runtime. The frontend (user interface) needs it to run.

1. Go to [nodejs.org](https://nodejs.org/)
2. Download the "LTS" (Long Term Support) version
3. Install it (default settings are fine)
4. Verify it worked:
   - Open Command Prompt/Terminal
   - Type: `node --version`
   - You should see something like `v18.17.0`

### Step 3: Get the Project Files

**Option A: If you have Git:**
```bash
git clone https://github.com/ivanKnt/davyNotebooks.git
cd davyNotebooks
git checkout dev
```

**Option B: If you don't have Git:**
1. Go to the GitHub repository
2. Click the green "Code" button
3. Click "Download ZIP"
4. Extract the ZIP file
5. Open Command Prompt/Terminal and navigate to the folder

### Step 4: Install Backend Dependencies

**What are dependencies?** Think of them as ingredients. Your project needs specific Python packages to work.

```bash
# Create a virtual environment (keeps things organized)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Download language data (for text analysis)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### Step 5: Install Frontend Dependencies

```bash
cd davy-frontend
npm install
```

---

## 📦 Important: About the Data Files

**⚠️ The notebook data files are NOT included in this repository!**

The actual notebook files (TEI XML files, transcriptions, etc.) are in the `items/` folder, but this folder is listed in `.gitignore`, which means it's not pushed to GitHub because:
- The files are very large (several GB)
- They contain original research data
- They're available from the official Davy Notebooks Project

**To get the data:**
1. Contact the [Davy Notebooks Project team](https://wp.lancs.ac.uk/davynotebooks)
2. Request access to the TEI XML files and transcription data
3. Place the files in the `items/` directory following this structure:
   ```
   items/
     01a1/
       tei/doc
       transcription/source/classifications
     01a2/
       tei/doc
       transcription/source/classifications
     ...
   ```

**Without the data files, you can:**
- ✅ Run the user interface
- ✅ See the application layout
- ❌ Cannot run actual analyses (no data to analyze)

---

## 🎯 Running the Application

You need to run TWO things: the backend (server) and the frontend (user interface).

### Start the Backend Server

Open a terminal/command prompt:

```bash
# Make sure you're in the project root directory
cd theDavyNotebooksProjectPython

# Activate your virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Start the backend
cd davy_web
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5001
```

**Leave this terminal window open!** The backend needs to keep running.

### Start the Frontend

Open a **NEW** terminal/command prompt (keep the backend running in the first one):

```bash
# Navigate to the frontend folder
cd davy-frontend

# Start the frontend
npm run dev
```

You should see:
```
Local: http://localhost:5173
```

### Open the Application

Open your web browser and go to: **http://localhost:5173**

You should see the Davy Notebooks homepage!

---

## 🧭 Navigating the User Interface

### Home Page

This is your starting point. You'll see cards for different features:

- **Preprocessing** - Prepare raw notebook files for analysis
- **Classification** - See what type of content each page contains
- **Poetry Detection** - Find notebooks and pages with poetry
- **Text Reuse Analysis** - Find where text was copied between notebooks
- **Inventory** - See which notebooks and files you have

### 1. Preprocessing Page - Getting Your Data Ready

**What is preprocessing?** 

Think of the original notebook files like a book written in a complex language with lots of formatting marks, footnotes, and annotations. Before you can analyze the actual words, you need to translate them into plain text that the computer can understand. That's what preprocessing does.

**What you'll see when you open this page:**

At the top, you'll see a big header that says "Preprocessing" and some explanation text. Below that, you'll see:

1. **A list of all notebooks** - These are displayed in rows, each with:
   - The notebook ID (like "01a2", "14e", "gs61")
   - A checkbox next to each one
   - Maybe a status indicator showing if it's already been preprocessed

2. **Two buttons**:
   - "Select All" / "Deselect All" - Quick shortcuts to check/uncheck everything
   - "Run Preprocessing" - The main action button (usually in a bold color like blue or green)

3. **A results area** - This is empty at first, but after you run preprocessing, you'll see:
   - Progress messages scrolling by
   - Success/error messages for each notebook
   - A final summary

**Step-by-Step: How to Use It**

**Step 1**: Look at the list of notebooks. If you're just starting, you probably haven't preprocessed any yet, so they'll all show as "Not Processed" or similar.

**Step 2**: Decide which notebooks to process:
   - **First time?** Start with just 2-3 notebooks to see how it works
   - **Processing everything?** Click "Select All" to check all boxes
   - **Specific notebooks?** Manually check the boxes for the ones you want

**Step 3**: Click the "Run Preprocessing" button. What happens next:

- The button might turn gray and say "Processing..." so you know it's working
- You'll see messages appear like:
  - "Processing notebook 01a2..."
  - "Extracting text from TEI XML..."
  - "Processing classifications..."
  - "Saving results to preprocessing/01a2/"
  - "✓ Successfully processed 01a2"

**Step 4**: Wait for it to finish. For each notebook, preprocessing does these things:

1. **Opens the TEI XML file** - This is the complex formatted version of the notebook
2. **Extracts the text** - Pulls out the actual words, page by page, removing all the markup
3. **Identifies entities** - Finds mentions of people (like "Aristotle"), places (like "London"), chemicals (like "oxygen")
4. **Processes classifications** - Reads the volunteer data about what type of content is on each page
5. **Saves 4 JSON files** to `preprocessing/<notebook_id>/`:
   - `page_to_text.json` - The clean text for each page
   - `page_to_entities.json` - Which entities appear on which pages
   - `all_entities_metadata.json` - Details about all entities mentioned
   - `classifications.json` - The volunteer classification votes

**What the results mean:**

After preprocessing finishes, you'll see a summary like:
```
Successfully processed: 3 notebooks
Failed: 0 notebooks
Total time: 45 seconds
```

If any fail, you'll see error messages explaining why (usually missing source files).

**When to use this:** 

This is **ALWAYS your first step**! None of the other features work until you've preprocessed the notebooks. The good news is you only need to do it once per notebook - the results are saved, so you can close the app and come back later without re-preprocessing.

**Pro tip:** If you get an error, check the Inventory page to see if the source files exist for that notebook.

### 2. Classification Page - Understanding What's In Each Notebook

**What is classification?**

When the Davy Notebooks were transcribed, volunteers on Zooniverse didn't just type out the words - they also classified what type of content each page contained. Is it about chemistry experiments? Poetry? Lecture notes? Personal reflections? This page shows you those classifications and helps you understand what each notebook contains.

**What you'll see when you open this page:**

The page has several sections:

1. **At the very top**: A big button that says "Process Classifications"
   - This button is how you generate the classification summaries
   - You'll only see this button if you haven't run classification processing yet

2. **Once processed, a dropdown menu** labeled "Select Notebook"
   - Lists all your notebooks (01a2, 14e, gs61, etc.)
   - Click on it to see the full list
   - Currently selected notebook is shown in the dropdown

3. **Notebook Overview Section** showing:
   - The full notebook title (e.g., "Notebook 01A2 (T6, 2023; lecture notes)")
   - **Overall Classification** - What the entire notebook is primarily about
   - A big, colorful pie chart showing the content breakdown

4. **Page-by-Page Details Section** at the bottom:
   - A table or list of every page in the notebook
   - Each page shows its classification percentages
   - Color-coded for easy reading

**Step-by-Step: How to Use It**

**FIRST TIME SETUP:**

Before you can view anything, you need to process the classifications. Here's how:

**Step 1**: Click the "Process Classifications" button at the top

What happens:
- The system reads the raw volunteer data from `preprocessing/<notebook>/classifications.json`
- It counts up how many volunteers said each classification for each page
- It calculates percentages (e.g., if 5 out of 7 volunteers said "Electrochemistry", that's 71%)
- It determines the "consensus" (the most common answer) for each page
- It figures out what the whole notebook is mostly about
- It saves everything to `classifications/<notebook>/classifications_page.json`

You'll see messages like:
- "Processing classifications..."
- "Processed notebook 01a2"
- "Processed notebook 14e"
- "✓ Classification processing complete!"

This might take 30 seconds to a minute depending on how many notebooks you have.

**Step 2**: Once processing is done, the page refreshes and shows:
- The dropdown menu with all your notebooks
- The first notebook is automatically selected and displayed

**VIEWING A NOTEBOOK:**

**Step 3**: Click on the dropdown menu to select a notebook. Let's say you choose "01a2". Here's what you'll see:

**The Notebook Title Bar:**
```
Notebook 01A2 (T6, 2023; lecture notes)
Overall Classification: Lecture notes
```

This tells you:
- The notebook ID (01a2)
- Its cataloging reference (T6, 2023)
- What it's primarily about (lecture notes)

**The Pie Chart:**

You'll see a circular chart divided into colored slices, like:
- 🔵 Blue slice (50%): "Lecture notes"
- 🟢 Green slice (30%): "Electrochemistry"
- 🟡 Yellow slice (15%): "Philosophy"
- 🔴 Red slice (5%): "Poetry"

This shows you the overall content breakdown of the notebook. The bigger the slice, the more pages of that type.

**How the percentages are calculated:**
- The system looks at every page's consensus
- If 10 out of 20 pages are "Lecture notes", that's 50%
- The pie chart shows you the proportions visually

**Hovering over the chart:**
- Move your mouse over a slice
- A tooltip appears showing the exact percentage and number of pages
- Example: "Lecture notes: 50% (10 pages)"

**The Page-by-Page Table:**

Scroll down and you'll see something like this:

```
Page 1:
  Lecture notes: 50%
  Electrochemistry: 33%
  Other electric: 17%
  → Consensus: Lecture notes

Page 2:
  Lecture notes: 71%
  Philosophy: 14%
  Other: 14%
  → Consensus: Lecture notes

Page 3:
  Poetry: 80%
  Lecture notes: 20%
  → Consensus: Poetry
```

**Understanding what you're seeing:**

Each page shows:
- **The classification categories** that volunteers chose
- **Percentages** showing how many volunteers chose each
- **The consensus** (marked with an arrow →) - the most popular choice

**What the percentages mean:**

Let's say Page 1 shows "Lecture notes: 50%":
- This means 50% of volunteers who classified this page said it was "Lecture notes"
- If 6 volunteers classified the page, 3 of them said "Lecture notes"

**Why multiple categories?**

Volunteers sometimes disagreed! Page 1 might have:
- 50% saying "Lecture notes" (3 volunteers)
- 33% saying "Electrochemistry" (2 volunteers)  
- 17% saying "Other electric" (1 volunteer)

The system keeps all this information so you can see:
- How confident the classification is (80% = very confident, 40% = uncertain)
- What else the page might contain (mixed content)

**Using the Classifications:**

Now that you can see what's in each notebook, you can:

1. **Find specific content types**:
   - Want to study Davy's poetry? Look for high "Poetry" percentages
   - Researching electrochemistry? Find pages with high "Electrochemistry" scores

2. **Assess classification confidence**:
   - High percentage (70%+) = volunteers agreed, classification is reliable
   - Low percentage (40-60%) = volunteers disagreed, might be mixed content
   - Very low (20-30%) = very mixed or ambiguous content

3. **Understand notebook structure**:
   - Is it a specialized notebook (all one type) or mixed?
   - Where do topic changes happen?
   - Are certain types of content grouped together?

4. **Plan further analysis**:
   - Use high-confidence pages for detailed study
   - Investigate low-confidence pages to see why volunteers disagreed

**Common Classification Categories You'll See:**

- **Electrochemistry** - Experiments with electricity and chemical reactions
- **Lecture notes** - Material prepared for or from lectures
- **Philosophy** - Philosophical musings and reflections
- **Poetry** - Poems, verse, literary content
- **Geology** - Geological observations and theories
- **Other electric** - Electricity topics that aren't electrochemistry (static electricity, etc.)
- **Refers to other writers/their works** - Quotations, references, discussions of other authors
- **Other** - Anything that doesn't fit the above categories

**Troubleshooting:**

- **"No notebooks found"**: Run preprocessing first, then run classification processing
- **"Some notebooks missing from dropdown"**: They might not have classification data (check Inventory page)
- **"All pages show 0%"**: The classification file might be empty - check the source data

### 3. Poetry Detection Pages - Finding Davy's Literary Side

One fascinating discovery from the Davy Notebooks Project is that Humphry Davy wasn't just a scientist - he was also a poet! His notebooks contain poetry alongside scientific observations. This feature helps you find all that poetry quickly.

#### Traditional Poetry Detection - Keyword-Based Search

**What does this do?**

This feature scans through all the classification data (remember the volunteer classifications from the previous page?) and pulls out every page and notebook that was classified as containing poetry. It's like having someone read through all 134 notebooks and bookmark every poem for you!

**What you'll see on this page:**

1. **A "Detect Poetry" button** at the top
   - This runs the poetry detection process
   - You only need to click it once (results are saved)

2. **Two main result sections:**
   - "Overall Poetry Notebooks" - Entire notebooks that are primarily poetry
   - "Poetry Pages" - Individual pages with poetry (even if the notebook isn't mainly poetry)

3. **Summary statistics** showing:
   - Total number of poetry notebooks found
   - Total number of poetry pages found
   - Percentage of notebooks containing some poetry

**Step-by-Step: How to Use It**

**Step 1**: Make sure you've run classification processing first
- Go to the Classification page
- Click "Process Classifications"
- Wait for it to finish
- Then come back here

**Step 2**: Click the "Detect Poetry" button

What happens behind the scenes:
- The system opens each notebook's classification file
- It looks for keywords like "poetry", "poem", "verse", "poetic"
- It checks both individual page classifications AND overall notebook classifications
- It separates findings into two categories:
  - **Poetry Notebooks**: Where the consensus for the entire notebook is "poetry"
  - **Poetry Pages**: Any individual page classified as poetry (even in non-poetry notebooks)
- It saves the results to `poetry_files/` (4 files: 2 CSV, 2 TXT)

You'll see progress messages:
```
Scanning notebooks for poetry...
Found poetry in notebook 13a
Found poetry in notebook 14e
✓ Poetry detection complete!
Found 2 poetry notebooks
Found 15 poetry pages across 8 notebooks
```

**Step 3**: View the "Overall Poetry Notebooks" section

This shows notebooks where **most or all pages are poetry**. You'll see something like:

```
Overall Poetry Notebooks (2 found):

📓 Notebook 13a
   Title: Notebook 13A (Poetry)
   Overall Classification: Poetry
   View details →

📓 Notebook 14e  
   Title: Notebook 14E (Mixed content)
   Overall Classification: Poetry
   View details →
```

**What this means:**
- These notebooks are dedicated poetry collections or heavily poetic
- If you're studying Davy's literary work, start here
- Click "View details" to see which specific pages

**Step 4**: Scroll down to the "Poetry Pages" section

This shows **every single page** classified as poetry, organized by notebook:

```
Poetry Pages (15 pages found across 8 notebooks):

Notebook 01a2:
  → Page 15: Poetry (80% confidence)
  → Page 23: Poem (100% confidence)

Notebook 13a:
  → Page 1: Poetry (85% confidence)
  → Page 2: Poetry (90% confidence)
  → Page 3: Poetry (75% confidence)
  → Page 4: Poem (95% confidence)

Notebook 14e:
  → Page 42: Poetry (70% confidence)
  → Page 43: Poetry (65% confidence)
```

**Understanding the confidence scores:**
- **100%**: All volunteers agreed it's poetry
- **80-90%**: Most volunteers agreed (high confidence)
- **65-75%**: Majority agreed but some disagreed (moderate confidence)
- **50-60%**: Mixed opinions (might be borderline)

**How to use these results:**

1. **For literary research**:
   - Export the poetry pages list
   - Go through each page systematically
   - Compare poetic style across different notebooks

2. **Finding specific poems**:
   - Note the notebook and page numbers
   - Go to the Classification page to see that notebook
   - View the actual text in the preprocessing files

3. **Understanding Davy's work habits**:
   - Are poetry pages clustered together or scattered?
   - Do certain notebooks mix poetry with science?
   - When did he write poetry (check notebook dates)?

4. **Download the results**:
   - Look in the `poetry_files/` folder
   - `poetry_pages.csv` - Spreadsheet format for analysis
   - `poetry_pages.txt` - Human-readable list
   - `overall_poetry_notebooks.csv` - Poetry notebooks only
   - `overall_poetry_notebooks.txt` - Summary report

**Why some notebooks might be missing:**

If a notebook doesn't appear in the results, it could mean:
- It genuinely contains no poetry
- It hasn't been classified yet (check notebook 08, gs61-gs65)
- Volunteers classified it differently (maybe as "Refers to other writers")

**Pro tip:** Cross-reference with the Classification page
- If you see a notebook with poetry here, go view it on the Classification page
- Look at the percentages - maybe there's more poetry than you thought!
- Pages with low confidence might be worth manual inspection

#### Advanced Poetry Detection (Future Feature)

This will use AI (Large Language Models like GPT or Claude) to detect poetry more accurately. 

**Why would this be better?**
- Current method relies on volunteer classifications
- AI could detect poetic language even if volunteers didn't label it as "poetry"
- Could identify different poetry styles (sonnets, ballads, etc.)
- Could find "poetic prose" - literary language that isn't technically poetry

**Status**: Not yet implemented - this is a placeholder for future development.

### 4. Text Reuse Analysis Pages

**What is text reuse?** When Davy copied, adapted, or reused text from one notebook to another. This helps researchers see how his ideas evolved.

#### Traditional Text Reuse

This page lets you use three different algorithms to find text reuse. Let's explain each in simple terms:

---

### 📊 Understanding the Algorithms

#### **N-gram Analysis** - "Matching Word Sequences"

**Simple explanation:** Breaks text into overlapping word groups and finds which groups appear in multiple notebooks.

**Example:**
- Text A: "The oxygen experiment was successful"
- Text B: "The nitrogen experiment was successful"

Using 2-word groups (2-grams):
- Shared: "the experiment", "experiment was", "was successful"
- Different: "oxygen" vs "nitrogen"

**When to use:**
- When you want a quick overview of similarities
- When you're okay with scattered matches (not necessarily continuous copying)
- When you want to find general thematic overlap

**Configuration options:**

- **N-gram Size** (recommended: 2-4):
  - `2-grams`: Very fast, finds lots of matches (including common phrases)
  - `3-grams`: Balanced - good precision and recall
  - `4-grams`: Slower, but very precise (only substantial matches)
  - **Tip**: Start with 2-grams for a quick scan, then use 4-grams for precise matching

- **Similarity Threshold** (recommended: 0.2-0.4):
  - `0.1` (10%): Finds almost everything (may include false positives)
  - `0.2` (20%): Good starting point - finds meaningful similarities
  - `0.4` (40%): Very strict - only strong matches
  - **Tip**: Use 0.2 for exploration, 0.4 for confident copying

- **Use Stemming** (recommended: Yes):
  - **Yes**: Treats "experiment", "experiments", "experimenting" as the same
  - **No**: Requires exact word forms
  - **Tip**: Turn this ON unless you care about exact word forms

- **Remove Stopwords** (recommended: Yes):
  - **Yes**: Ignores common words like "the", "and", "of"
  - **No**: Includes all words
  - **Tip**: Turn this ON for faster processing and more meaningful matches

---

#### **GST (Greedy String Tiling)** - "Finding Exact Copying"

**Simple explanation:** Finds the longest matching sections of text, then marks them as "used" and finds the next longest match. Great for detecting plagiarism or direct copying.

**Example:**
- Text A: "The experiment with oxygen was conducted carefully and showed promising results"
- Text B: "The experiment with nitrogen was conducted carefully and proved the hypothesis"

GST finds:
- Match 1: "The experiment with" (4 words)
- Match 2: "was conducted carefully and" (4 words)

**When to use:**
- When you suspect direct copying (not just similar topics)
- When you want to find continuous passages that match
- When word order matters

**Configuration options:**

- **Minimum Match Length** (recommended: 3-5):
  - `3`: Finds many short matches (may include common phrases)
  - `4`: Balanced - substantial matches
  - `5` or more: Only long, deliberate copying
  - **Tip**: Use 3 for exploratory analysis, 5 for proven copying

- **Use Stemming** (recommended: No):
  - **No**: Requires exact words (better for detecting direct copying)
  - **Yes**: Allows word variations
  - **Tip**: Turn this OFF to find exact copying

- **Remove Stopwords** (recommended: No):
  - **No**: Includes all words (even "the", "and") for exact matching
  - **Yes**: Focuses on content words only
  - **Tip**: Turn this OFF unless you're only interested in content words

---

#### **TF-IDF** - "Finding Thematic Similarity"

**Simple explanation:** Weighs words by how unique they are. Common words like "the" have low importance, while unique words like "electrochemistry" have high importance.

**Example:**
- Text A: "The electrochemical experiment with oxygen"
- Text B: "The chemical reaction using nitrogen"

TF-IDF notices:
- "electrochemical" and "chemical" are related (semantic similarity)
- Both discuss chemistry even without exact word matches
- Ignores common words like "the"

**When to use:**
- When you want to find passages about the same topic (even if worded differently)
- When you're looking for thematic connections, not exact copying
- When paraphrasing might be involved

**Configuration options:**

- **N-gram Range** (recommended: 1-3):
  - `(1, 1)`: Only individual words
  - `(1, 2)`: Words and 2-word phrases
  - `(1, 3)`: Words, 2-word phrases, and 3-word phrases
  - **Tip**: Use (1, 3) for best results - captures both single terms and phrases

- **Similarity Threshold** (recommended: 0.3-0.5):
  - `0.2`: Finds distant thematic connections
  - `0.3`: Good balance
  - `0.5`: Only very similar topics
  - **Tip**: Start with 0.3 and adjust based on results

- **Similarity Metric** (recommended: cosine):
  - `cosine`: Most common, works well for text (recommended)
  - `euclidean`: Different distance measure
  - `manhattan`: Another distance measure
  - **Tip**: Stick with cosine unless you have a specific reason to change

- **Use Stemming** (recommended: Yes):
  - **Yes**: Groups word variations
  - **No**: Exact words only
  - **Tip**: Turn this ON for better semantic matching

- **Remove Stopwords** (recommended: Yes):
  - **Yes**: Focuses on meaningful words
  - **No**: Includes all words
  - **Tip**: Turn this ON (stopwords don't help TF-IDF)

---

### 🎓 Which Algorithm Should You Use?

**Use N-gram when:**
- ✅ You want a quick first pass
- ✅ You're looking for general similarity
- ✅ You don't mind scattered matches

**Use GST when:**
- ✅ You suspect direct copying
- ✅ You want to find continuous matching passages
- ✅ Word order is important

**Use TF-IDF when:**
- ✅ You're looking for thematic similarity
- ✅ The text might be paraphrased
- ✅ You want to find conceptually related passages

**Best practice:** Run all three! They complement each other:
1. Start with N-gram (fast overview)
2. Use GST for suspected copying
3. Use TF-IDF for thematic connections

---

### 💡 Recommended Configurations for Beginners

#### Quick Exploration
```
N-gram Analysis:
- N-gram Size: 2
- Similarity Threshold: 0.2
- Use Stemming: Yes
- Remove Stopwords: Yes
```

#### Finding Direct Copying
```
GST Analysis:
- Minimum Match Length: 4
- Use Stemming: No
- Remove Stopwords: No
```

#### Finding Related Topics
```
TF-IDF Analysis:
- N-gram Range: (1, 3)
- Similarity Threshold: 0.3
- Similarity Metric: cosine
- Use Stemming: Yes
- Remove Stopwords: Yes
```

---

### 5. Inventory Page - Your Data Health Check

**What does this show?**

Think of this as a diagnostic tool - it tells you exactly which notebooks you have, which files exist for each one, and what might be missing. It's like checking your cupboard to see what ingredients you have before cooking!

**What you'll see on this page:**

1. **A "Run File Scan" button** at the top
2. **Overall Statistics panel** showing:
   - Total notebooks found
   - How many have TEI XML files (the source data)
   - How many have preprocessing outputs
   - How many have classification data
   - Percentages for each file type

3. **Detailed notebook list** showing each notebook with checkmarks and X marks:
   ```
   01a2:
     ✓ TEI XML
     ✓ Valid Text
     ✓ Tagged Text  
     ✓ Zooniverse Files
     ✓ Classifications
     Status: Complete
   
   08:
     ✓ TEI XML
     ✗ Valid Text
     ✗ Tagged Text
     ✗ Zooniverse Files
     ✗ Classifications
     Status: Missing files
   
   gs61:
     ✓ TEI XML
     ✓ Valid Text
     ✓ Tagged Text
     ✓ Zooniverse Files
     ✗ Classifications
     Status: Missing classifications
   ```

**Step-by-Step: How to Use It**

**Step 1**: Click "Run File Scan"

The system checks:
- `items/` folder for source files (TEI XML, transcriptions)
- `preprocessing/` folder for preprocessed outputs
- `classifications/` folder for classification results
- `poetry_files/` folder for poetry detection results

This takes about 5-10 seconds.

**Step 2**: Review the overall statistics

You'll see something like:
```
Total Notebooks: 134
TEI XML files: 134/134 (100%)
Preprocessing complete: 128/134 (95.5%)
Classifications available: 128/134 (95.5%)
```

This tells you:
- You have all the source TEI XML files (good!)
- Most notebooks are preprocessed (128 out of 134)
- 6 notebooks need preprocessing or are missing files

**Step 3**: Scroll through the detailed list

Look for notebooks marked "Missing files" or "Incomplete":
- These are the ones blocking your analysis
- Note which files are missing (shown with ✗)
- Cross-reference with the "Data Quality & Known Issues" section in README2.md

**What each file type means:**

- **TEI XML**: The original formatted notebook file (required for everything)
- **Valid Text**: Cleaned text file (created during preprocessing)
- **Tagged Text**: Text with annotations (created during preprocessing)
- **Zooniverse Files**: Transcription data from volunteers (required for preprocessing)
- **Classifications**: Volunteer classification data (required for classification/poetry features)

**Using this information:**

**Before preprocessing:**
- Check that notebooks have TEI XML and Zooniverse files
- If missing, you can't preprocess them

**Before classification:**
- Check that notebooks have Classifications marked with ✓
- Notebooks without this can be preprocessed but not classified

**Troubleshooting:**
- **"Only 128 notebooks show up"**: Check if you have data files in `items/`
- **"Notebook X is missing Zooniverse files"**: Contact Davy Notebooks Project team
- **"Everything shows ✗"**: Make sure data files are in the correct folder structure

**Pro tip:** Run this scan after:
- Adding new notebooks to `items/`
- Running preprocessing
- Getting errors in other features (check what's missing here!)

---

## ❓ Common Questions

### "The backend isn't starting!"
- Make sure you activated the virtual environment
- Check if port 5001 is already in use
- Look at the error message for clues

### "The frontend shows errors!"
- Make sure the backend is running first
- Check that you're going to http://localhost:5173 (not 5174 or other port)
- Look at the browser console (F12) for error messages

### "I don't see any notebooks!"
- Make sure you have data files in the `items/` folder
- Run the preprocessing first
- Check the Inventory page to see what's available

### "Analysis is taking forever!"
- Text reuse analysis can be slow (especially with many notebooks)
- Start with just 2-3 notebooks to test
- Use larger n-grams for faster (but less sensitive) matching

### "I don't understand the results!"
- High similarity (>50%) = very similar text
- Medium similarity (20-50%) = some shared content
- Low similarity (<20%) = minimal overlap
- Check the detailed report (TXT file) for human-readable explanations

---

## 🆘 Getting Help

**If you're stuck:**

1. **Check the logs**: Look at the terminal windows where backend/frontend are running
2. **Read error messages carefully**: They often tell you exactly what's wrong
3. **Check the detailed README2.md**: For technical details
4. **Contact the Davy Notebooks Project team**: [davynotebooks@lancaster.ac.uk](mailto:davynotebooks@lancaster.ac.uk)

---

## 🎉 You're Ready!

Now you know:
- ✅ How to start the application
- ✅ What each page does
- ✅ How to configure text reuse algorithms
- ✅ How to interpret results

**Happy exploring!** The Davy Notebooks contain fascinating insights into 18th-century science and literature. Enjoy discovering Davy's work!

---

**Next Steps:**
- Read the detailed documentation in `README2.md` for technical information
- Contact the Davy Notebooks Project to get the full dataset
- Start with small analyses and work your way up to larger ones

