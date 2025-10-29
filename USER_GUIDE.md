# Davy Notebooks - User Guide for the Web Interface

**👋 Welcome!** This guide will help you use the Davy Notebooks web application to explore Sir Humphry Davy's scientific notebooks.

---

## What Can You Do With This App?

The Davy Notebooks web application lets you:
- 📖 Browse and classify notebook content (poetry, chemistry, lecture notes, etc.)
- 🎭 Find all the poetry Davy wrote
- 🔄 Discover where he reused or copied text between notebooks
- 📊 View statistics and visualizations about the notebooks
- 🔍 Search and explore 134 historical notebooks

**No coding required!** Everything is done through the web interface.

---

## 🚀 Starting the Application (Quick Setup)

### Prerequisites

You need:
- **Python 3.8+** installed ([download here](https://python.org/downloads))
- **Node.js** installed ([download here](https://nodejs.org))
- The project files on your computer

### Starting the Servers

You need to run **two things** - a backend server and a frontend interface.

**Step 1: Start the Backend**

Open a terminal/command prompt and run:

```bash
# Windows
cd path\to\theDavyNotebooksProjectPython
venv\Scripts\activate
cd davy_web
python app.py

# Mac/Linux
cd path/to/theDavyNotebooksProjectPython
source venv/bin/activate
cd davy_web
python app.py
```

You should see: `Running on http://127.0.0.1:5001`

**Keep this window open!**

**Step 2: Start the Frontend**

Open a **NEW** terminal/command prompt and run:

```bash
cd path/to/theDavyNotebooksProjectPython/davy-frontend
npm run dev
```

You should see: `Local: http://localhost:5173`

**Step 3: Open Your Browser**

Go to: **http://localhost:5173**

---

## 🏠 The Home Page

When you first open the app, you'll see the **Home Page** with several cards:

```
┌─────────────────────────────────────────────┐
│  Davy Notebooks Project                     │
├─────────────────────────────────────────────┤
│                                             │
│  [📝 Preprocessing]  [📊 Classification]   │
│                                             │
│  [🎭 Poetry]         [🔄 Text Reuse]       │
│                                             │
│  [📋 Inventory]                             │
└─────────────────────────────────────────────┘
```

**What each card does:**

- **📝 Preprocessing** - Prepare notebooks for analysis (run this first!)
- **📊 Classification** - See what type of content is on each page
- **🎭 Poetry** - Find all the poetry in the notebooks
- **🔄 Text Reuse** - Find where text was copied between notebooks
- **📋 Inventory** - Check which notebooks and files you have

Click any card to go to that feature.

---

## 📝 Page 1: Preprocessing

**When to use:** First time setup - you must do this before anything else works.

### What You'll See

The page shows:
- A list of all available notebooks (01a2, 14e, gs61, etc.)
- Checkboxes next to each notebook
- "Select All" and "Run Preprocessing" buttons

### How to Use It

1. **Select notebooks:**
   - Check the boxes for notebooks you want to process
   - OR click "Select All" to process everything
   - **First time? Start with 2-3 notebooks to test**

2. **Click "Run Preprocessing"**
   - The button turns gray and says "Processing..."
   - You'll see messages scrolling:
     ```
     Processing notebook 01a2...
     Extracting text...
     Processing classifications...
     ✓ Successfully processed 01a2
     ```

3. **Wait for completion**
   - Progress messages show what's happening
   - Green checkmarks (✓) mean success
   - Red X marks mean errors (usually missing files)

4. **Done!**
   - You'll see: "Successfully processed: 3 notebooks"
   - Now you can use the other features

**💡 Tip:** You only need to preprocess each notebook once. Results are saved!

---

## 📊 Page 2: Classification

**When to use:** After preprocessing - see what type of content is in each notebook.

### First Time Setup

1. Click the **"Process Classifications"** button at the top
2. Wait 30-60 seconds while it processes
3. You'll see: "✓ Classification processing complete!"

### Using the Page

Once processing is done, you'll see:

#### 1. Notebook Selector (Dropdown Menu)

```
┌─────────────────────────────┐
│ Select Notebook ▼           │
│  01a2                        │
│  01a3                        │
│  14e                         │
│  ...                         │
└─────────────────────────────┘
```

**Click the dropdown** to see all notebooks. Select one to view it.

#### 2. Notebook Overview

After selecting a notebook, you'll see:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Notebook 01A2 (T6, 2023; lecture notes)
Overall Classification: Lecture notes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

This tells you what the notebook is primarily about.

#### 3. Pie Chart - Content Breakdown

You'll see a colorful pie chart showing:

![Pie Chart Example]
- 🔵 **Blue**: Lecture notes (50%)
- 🟢 **Green**: Electrochemistry (30%)
- 🟡 **Yellow**: Philosophy (15%)
- 🔴 **Red**: Poetry (5%)

**Hover over any slice** with your mouse to see exact numbers!

#### 4. Page-by-Page Results

Scroll down to see individual pages:

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

**Understanding the percentages:**
- Shows what volunteers labeled each page as
- Higher percentage = more volunteers agreed
- The "Consensus" is what most people said

**Classification Types You'll See:**
- **Electrochemistry** - Chemistry experiments with electricity
- **Lecture notes** - Lecture materials
- **Philosophy** - Philosophical thoughts
- **Poetry** - Poems and verses
- **Geology** - Geological notes
- **Other electric** - Electricity (not electrochemistry)
- **Refers to other writers** - Quotes and references
- **Other** - Everything else

### How to Use This Information

- **Looking for specific content?** Use the pie chart to see if it's in this notebook
- **High percentages** (70%+) = reliable classification
- **Low percentages** (40-60%) = mixed content or volunteers disagreed
- **Try different notebooks** using the dropdown to explore

---

## 🎭 Page 3: Poetry Detection

**When to use:** Find all the poetry across all notebooks.

### What You'll See

The page has:
1. A **"Detect Poetry"** button at the top
2. Two result sections (after detecting):
   - "Overall Poetry Notebooks"
   - "Poetry Pages"

### How to Use It

**Step 1: Click "Detect Poetry"**
- The system scans all notebooks for poetry
- Takes 5-10 seconds
- You'll see: "✓ Poetry detection complete!"

**Step 2: View Results**

#### Overall Poetry Notebooks

Shows notebooks that are **primarily poetry**:

```
📓 Notebook 13a
   Title: Notebook 13A (Poetry)
   Overall Classification: Poetry
   [View Details →]

📓 Notebook 14e
   Title: Notebook 14E (Mixed content)
   Overall Classification: Poetry
   [View Details →]
```

**What this means:** These entire notebooks are poetry collections.

#### Poetry Pages

Shows **every page** with poetry, organized by notebook:

```
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

**Understanding confidence:**
- **100%**: All volunteers agreed
- **80-90%**: Most agreed (high confidence)
- **65-75%**: Majority agreed (moderate)
- **Below 60%**: Mixed opinions

### What You Can Do

- **Count**: See how many poetry pages total
- **Locate**: Find specific page numbers with poetry
- **Explore**: Click different notebooks to see their poetry
- **Research**: Note which notebooks mix poetry with science

**💡 Tip:** Cross-reference with the Classification page to see more details about any specific page!

---

## 🔄 Page 4: Text Reuse Analysis

**When to use:** Find where Davy copied or reused text between notebooks.

### What Is Text Reuse?

Text reuse is when the same or similar text appears in multiple notebooks. This shows:
- How ideas evolved across time
- When he refined lecture materials
- Connections between different notebooks

### The Three Analysis Methods

The page offers three different ways to find text reuse:

```
┌─────────────────────────────────┐
│  N-gram Analysis                │
│  [Configure] [Run Analysis]     │
├─────────────────────────────────┤
│  GST (Greedy String Tiling)     │
│  [Configure] [Run Analysis]     │
├─────────────────────────────────┤
│  TF-IDF Similarity              │
│  [Configure] [Run Analysis]     │
└─────────────────────────────────┘
```

Let me explain each in simple terms:

---

### Method 1: N-gram Analysis

**What it does:** Finds pages that share the same word sequences.

**Think of it like:** Comparing pages by breaking them into small word groups and seeing how many groups match.

**Example:**
- Page A: "The experiment was successful in demonstrating"
- Page B: "The experiment was successful in proving"
- They share: "The experiment was", "experiment was successful", "was successful in"

#### How to Use N-gram

**1. Click "Configure" to see settings:**

```
N-gram Size: [2 ▼]
Similarity Threshold: [0.2 ▼]
☑ Use Stemming
☑ Remove Stopwords
Notebooks to Compare: [Select ▼]
```

**What each setting means:**

- **N-gram Size** (choose 2, 3, or 4):
  - **2**: Fast, finds lots of matches
  - **3**: Balanced (recommended for beginners)
  - **4**: Slower, but very precise

- **Similarity Threshold** (0.1 to 1.0):
  - **0.2** = Find anything 20% similar or more (recommended)
  - **0.4** = Only strong matches (50%+ similar)
  - Lower = more results, higher = fewer but stronger results

- **Use Stemming**: 
  - ☑ Checked = "experiment", "experiments", "experimenting" count as same word
  - ☐ Unchecked = Requires exact word forms

- **Remove Stopwords**:
  - ☑ Checked = Ignores common words like "the", "and", "of" (faster)
  - ☐ Unchecked = Includes all words

**2. Select Notebooks:**
- Click the dropdown
- Check boxes for notebooks to compare
- **Tip:** Start with 2-3 notebooks (comparing many takes time!)

**3. Click "Run Analysis"**
- You'll see: "Running N-gram analysis..."
- Progress bar shows completion
- Takes 1-5 minutes depending on how many notebooks

**4. View Results:**

You'll see a table like:

```
╔══════════════════════════════════════════════════╗
║ Notebook 1 │ Page │ Notebook 2 │ Page │ Similarity ║
╠══════════════════════════════════════════════════╣
║ 01a2      │  5   │ 01a4      │  3   │   45%     ║
║ 01a2      │  12  │ 14e       │  8   │   67%     ║
║ 14e       │  42  │ 14g       │  15  │   89%     ║
╚══════════════════════════════════════════════════╝
```

**What it means:**
- Each row shows two pages that are similar
- Higher percentage = more similar
- **Click a row** to see the actual text side-by-side

---

### Method 2: GST (Greedy String Tiling)

**What it does:** Finds continuous sections of exact matching text.

**Think of it like:** Looking for copy-paste sections - where he literally copied chunks of text.

**Best for:** Finding direct copying (not just similar topics).

#### How to Use GST

**1. Click "Configure":**

```
Minimum Match Length: [4 ▼]
☐ Use Stemming
☐ Remove Stopwords
Notebooks to Compare: [Select ▼]
```

**Settings explained:**

- **Minimum Match Length**:
  - **3**: Finds short matches (may include common phrases)
  - **4**: Balanced (recommended)
  - **5+**: Only long, deliberate copying

- **Use Stemming**: Usually leave ☐ unchecked for GST (finds exact copying)
- **Remove Stopwords**: Usually leave ☐ unchecked (includes all words)

**2. Select notebooks and Click "Run Analysis"**

**3. View Results:**

Results show:
```
╔══════════════════════════════════════════════════════╗
║ Match │ Length │ Notebook 1 │ Notebook 2 │ Similarity ║
╠══════════════════════════════════════════════════════╣
║   1   │  42    │ 01a2, p5   │ 01a4, p3   │    45%     ║
║   2   │  85    │ 14e, p42   │ 14g, p15   │    78%     ║
╚══════════════════════════════════════════════════════╝
```

- **Length**: How many words matched in a row
- Longer length = more substantial copying
- **Click to see** the exact matching text

---

### Method 3: TF-IDF Similarity

**What it does:** Finds pages about the same topics, even if worded differently.

**Think of it like:** Smart topic matching - understands "electrochemistry" and "chemical reactions" are related.

**Best for:** Finding thematically similar pages (not exact copying).

#### How to Use TF-IDF

**1. Click "Configure":**

```
N-gram Range: [1-3 ▼]
Similarity Threshold: [0.3 ▼]
Similarity Metric: [cosine ▼]
☑ Use Stemming
☑ Remove Stopwords
Notebooks to Compare: [Select ▼]
```

**Settings explained:**

- **N-gram Range**: Keep at **1-3** (default is fine)
- **Similarity Threshold**: **0.3** is a good starting point
- **Similarity Metric**: Keep at **cosine** (recommended)
- **Use Stemming**: ☑ Check this (helps find related words)
- **Remove Stopwords**: ☑ Check this (focuses on meaningful words)

**2. Select notebooks and Click "Run Analysis"**

**3. View Results:**

Shows pages about similar topics, even if the exact words differ.

---

### 📊 Understanding Your Results

After running any analysis, you'll see:

**1. Results Table** - Lists all matches found

**2. Details View** - Click any row to see:
- The two pages side-by-side
- Highlighted matching sections
- Similarity percentage
- Context around the match

**3. Download Options** - Export results as:
- CSV (open in Excel)
- JSON (for further analysis)
- TXT (human-readable report)

### Which Method Should You Use?

- **Just exploring?** Start with **N-gram** (fast and easy)
- **Looking for copying?** Use **GST**
- **Finding related topics?** Use **TF-IDF**
- **Best approach:** Try all three and compare!

---

## 📋 Page 5: Inventory

**When to use:** Check which notebooks and files you have before running analyses.

### What You'll See

The Inventory page shows you the "health status" of your data.

### How to Use It

**1. Click "Run File Scan"**
- Takes 5-10 seconds
- Checks all notebooks and files

**2. View Overall Statistics**

You'll see a summary:

```
╔═════════════════════════════════════╗
║ Total Notebooks: 134                ║
║ TEI XML files: 134/134 (100%)      ║
║ Preprocessing: 128/134 (95.5%)     ║
║ Classifications: 128/134 (95.5%)   ║
╚═════════════════════════════════════╝
```

**What this tells you:**
- ✅ Green percentages (95%+) = Most notebooks ready
- ⚠️ Yellow percentages (80-95%) = Some missing
- ❌ Red percentages (<80%) = Many missing

**3. Scroll Through Detailed List**

Each notebook shows checkmarks or X marks:

```
01a2:
  ✓ Source Files
  ✓ Preprocessing Done
  ✓ Classifications Available
  Status: ✅ Complete

08:
  ✓ Source Files
  ✗ Preprocessing Done
  ✗ Classifications Available
  Status: ⚠️ Incomplete

gs61:
  ✓ Source Files
  ✓ Preprocessing Done
  ✗ Classifications Available
  Status: ⚠️ Missing Classifications
```

**Status indicators:**
- ✅ **Complete**: Everything available, ready to analyze
- ⚠️ **Incomplete**: Missing some files, limited functionality
- ❌ **Missing**: Can't process this notebook yet

### What to Do About Missing Files

**If you see incomplete notebooks:**

1. **Missing Preprocessing**: Go to Preprocessing page and process that notebook
2. **Missing Classifications**: Go to Classification page and run processing
3. **Missing Source Files**: Need to get data files from Davy Notebooks Project team

**💡 Tip:** Run this scan before doing big analyses to know which notebooks will work!

---

## 💡 Tips for Using the Application

### Getting Started
1. ✅ Always run **Preprocessing** first
2. ✅ Then run **Classification** processing
3. ✅ Now you can use **Poetry** and **Text Reuse**
4. ✅ Check **Inventory** if something doesn't work

### Best Practices
- **Start small**: Process 2-3 notebooks first to learn
- **Check results**: View one notebook in detail before processing all
- **Save work**: Results are saved automatically
- **Be patient**: Text reuse analysis takes time with many notebooks

### Navigation Tips
- Use your browser's **Back button** to return to previous pages
- Click the **Home** link (usually top-left) to return to the main menu
- Keep both terminal windows **open** while using the app
- **Refresh the page** (F5) if something looks stuck

### Performance Tips
- **Text Reuse Analysis** is slow with many notebooks:
  - Comparing 2 notebooks: ~1 minute
  - Comparing 10 notebooks: ~5-10 minutes  
  - Comparing 50 notebooks: ~30-60 minutes
  
- **Start with fewer notebooks** if it's too slow

- **Close other programs** if your computer is slow

---

## ❓ Troubleshooting

### "I don't see any notebooks!"
- ✅ Make sure you have data files in the `items/` folder
- ✅ Go to Inventory page and run File Scan to check

### "Preprocessing button does nothing"
- ✅ Check that backend is running (look at terminal window)
- ✅ Check browser console (press F12, look for errors)
- ✅ Try refreshing the page

### "No results in Classification page"
- ✅ Did you click "Process Classifications" first?
- ✅ Did you preprocess the notebooks?
- ✅ Check Inventory to see if classification files exist

### "Text Reuse is taking forever"
- ✅ This is normal! It's comparing lots of text
- ✅ Try selecting fewer notebooks (2-3 instead of 10+)
- ✅ Wait patiently - don't close the browser

### "The page looks broken"
- ✅ Make sure both backend AND frontend are running
- ✅ Go to correct URL: http://localhost:5173 (not 5001)
- ✅ Try refreshing the page (F5)
- ✅ Try a different browser

### "Backend crashed / error messages"
- ✅ Look at the backend terminal window for error details
- ✅ Restart the backend (close and run `python app.py` again)
- ✅ Check that you have all required Python packages

---

## 🎯 Common Tasks

### Task: "Find all poetry in the notebooks"
1. Go to **Preprocessing** → Process all notebooks
2. Go to **Classification** → Click "Process Classifications"
3. Go to **Poetry Detection** → Click "Detect Poetry"
4. Review the list of poetry pages

### Task: "See what notebook 14e contains"
1. Go to **Preprocessing** → Check box for 14e → Run Preprocessing
2. Go to **Classification** → Click "Process Classifications"
3. Select "14e" from dropdown
4. View the pie chart and page details

### Task: "Find if text was copied between notebooks 01a2 and 14e"
1. Make sure both are preprocessed
2. Go to **Text Reuse**
3. Choose **N-gram Analysis**
4. Select notebooks 01a2 and 14e
5. Click "Run Analysis"
6. Review results table

### Task: "Check which notebooks I can analyze"
1. Go to **Inventory**
2. Click "Run File Scan"
3. Look for ✅ Complete notebooks
4. Note any ⚠️ Incomplete ones

---

## 🆘 Getting More Help

**Need assistance?**

1. **Technical issues**: Check the backend terminal for error messages
2. **Missing data**: Contact the Davy Notebooks Project team for access to notebook files
3. **Questions about results**: Refer to the detailed technical documentation (README2.md)

**Contact:**
- Davy Notebooks Project: [davynotebooks@lancaster.ac.uk](mailto:davynotebooks@lancaster.ac.uk)
- Project website: [https://wp.lancs.ac.uk/davynotebooks](https://wp.lancs.ac.uk/davynotebooks)

---

## 🎉 You're Ready to Explore!

You now know how to:
- ✅ Navigate all 5 main pages
- ✅ Process and classify notebooks
- ✅ Find poetry content
- ✅ Analyze text reuse
- ✅ Check data status

**Happy exploring!** The Davy Notebooks contain fascinating insights into 18th-century science, literature, and the mind of one of history's great scientists.

---

**Quick Reference Card:**

```
┌─────────────────────────────────────────────┐
│  Page          │  What It Does              │
├─────────────────────────────────────────────┤
│  Preprocessing │  Extract text from files   │
│  Classification│  Show content types        │
│  Poetry        │  Find poetry pages         │
│  Text Reuse    │  Find copied text          │
│  Inventory     │  Check data status         │
└─────────────────────────────────────────────┘

First time: Preprocessing → Classification → Explore!
```
