#!/usr/bin/env python3

import os
import json
import csv
from pathlib import Path
from collections import defaultdict

def load_classification_data(notebook_id, classifications_dir):
    """Load classification data for a specific notebook."""
    classification_file = classifications_dir / notebook_id / "classifications_page.json"
    
    if not classification_file.exists():
        return None
    
    try:
        with open(classification_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {classification_file}: {e}")
        return None

def is_poetry_classification(classification):
    """Check if a classification indicates poetry content."""
    # Normalize classification to lowercase for comparison
    classification_lower = classification.lower()
    
    # Check for poetry-related terms
    poetry_terms = ['poetry', 'poem', 'verse', 'poetic']
    return any(term in classification_lower for term in poetry_terms)

def extract_poetry_notebooks_and_pages(classifications_dir):
    """Extract poetry notebooks and pages from classification data."""
    poetry_notebooks = []  # Notebooks with overall poetry consensus
    poetry_pages = defaultdict(list)  # All pages with poetry content, grouped by notebook
    
    # Get all notebook directories
    notebook_dirs = [d for d in classifications_dir.iterdir() if d.is_dir()]
    notebook_dirs.sort()  # Sort alphabetically
    
    for notebook_dir in notebook_dirs:
        notebook_id = notebook_dir.name
        classification_data = load_classification_data(notebook_id, classifications_dir)
        
        if classification_data is None:
            continue
        
        notebook_title = classification_data.get("notebook_title", f"Notebook {notebook_id}")
        consensus_book = classification_data.get("consensus_book", "").lower()
        
        # Check if the overall notebook consensus is poetry
        if is_poetry_classification(consensus_book):
            poetry_notebooks.append({
                "notebook_id": notebook_id,
                "notebook_title": notebook_title,
                "consensus_book": consensus_book
            })
        
        # Check individual pages for poetry content
        notebook_poetry_pages = []
        
        for key, value in classification_data.items():
            # Skip metadata keys
            if key in ["notebook_title", "consensus_book"]:
                continue
            
            # Check if this is a page entry
            if isinstance(value, dict) and "page_consensus" in value:
                page_num = key
                page_consensus = value.get("page_consensus", "").lower()
                
                # Check if page consensus is poetry
                if is_poetry_classification(page_consensus):
                    poetry_percentage = 0.0
                    
                    # Calculate poetry percentage for this page
                    for classification, percentage in value.items():
                        if classification != "page_consensus" and is_poetry_classification(classification):
                            poetry_percentage += percentage
                    
                    notebook_poetry_pages.append({
                        "page_number": page_num,
                        "page_consensus": page_consensus,
                        "poetry_percentage": round(poetry_percentage, 3),
                        "all_classifications": {k: v for k, v in value.items() if k != "page_consensus"}
                    })
        
        # If this notebook has any poetry pages, add to the collection
        if notebook_poetry_pages:
            poetry_pages[notebook_id] = {
                "notebook_title": notebook_title,
                "total_poetry_pages": len(notebook_poetry_pages),
                "pages": sorted(notebook_poetry_pages, key=lambda x: (
                    int(x["page_number"]) if x["page_number"].isdigit() else float('inf'), x["page_number"]
                ))
            }
    
    return poetry_notebooks, poetry_pages

def save_overall_poetry_notebooks(poetry_notebooks, output_file):
    """Save the list of notebooks with overall poetry consensus."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NOTEBOOKS WITH OVERALL POETRY CONSENSUS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total notebooks with poetry consensus: {len(poetry_notebooks)}\n\n")
        
        if not poetry_notebooks:
            f.write("No notebooks found with overall poetry consensus.\n")
            return
        
        f.write("NOTEBOOK LIST:\n")
        f.write("-" * 40 + "\n")
        
        for i, notebook in enumerate(poetry_notebooks, 1):
            f.write(f"{i:3}. {notebook['notebook_id']}\n")
            f.write(f"     Title: {notebook['notebook_title']}\n")
            f.write(f"     Consensus: {notebook['consensus_book'].title()}\n")
            f.write("\n")
        
        f.write("-" * 40 + "\n")
        f.write(f"Total: {len(poetry_notebooks)} notebooks\n")
        f.write("=" * 80 + "\n")

def save_poetry_pages(poetry_pages, output_file):
    """Save the list of all pages with poetry content, grouped by notebook."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ALL PAGES WITH POETRY CONTENT (GROUPED BY NOTEBOOK)\n")
        f.write("=" * 80 + "\n\n")
        
        total_notebooks = len(poetry_pages)
        total_pages = sum(notebook_data['total_poetry_pages'] for notebook_data in poetry_pages.values())
        
        f.write(f"Total notebooks with poetry pages: {total_notebooks}\n")
        f.write(f"Total poetry pages across all notebooks: {total_pages}\n\n")
        
        if not poetry_pages:
            f.write("No pages found with poetry content.\n")
            return
        
        # Sort notebooks by ID
        sorted_notebooks = sorted(poetry_pages.items())
        
        for notebook_id, notebook_data in sorted_notebooks:
            f.write("=" * 60 + "\n")
            f.write(f"NOTEBOOK: {notebook_id}\n")
            f.write(f"Title: {notebook_data['notebook_title']}\n")
            f.write(f"Poetry pages: {notebook_data['total_poetry_pages']}\n")
            f.write("=" * 60 + "\n")
            
            for page_info in notebook_data['pages']:
                f.write(f"  Page {page_info['page_number']:>3}: {page_info['page_consensus'].title()}")
                f.write(f" (Poetry: {page_info['poetry_percentage']:.1%})\n")
                
                # Show all classifications for this page
                classifications = page_info['all_classifications']
                classification_details = []
                for cls, pct in classifications.items():
                    classification_details.append(f"{cls}: {pct:.1%}")
                
                if classification_details:
                    f.write(f"              {', '.join(classification_details)}\n")
                f.write("\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"SUMMARY: {total_notebooks} notebooks, {total_pages} poetry pages total\n")
        f.write("=" * 80 + "\n")

def save_overall_poetry_notebooks_csv(poetry_notebooks, output_file):
    """Save the list of notebooks with overall poetry consensus as CSV."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['notebook_id', 'notebook_title', 'consensus_book'])
        
        # Write data
        for notebook in poetry_notebooks:
            writer.writerow([
                notebook['notebook_id'],
                notebook['notebook_title'],
                notebook['consensus_book']
            ])

def save_poetry_pages_csv(poetry_pages, output_file):
    """Save the list of all pages with poetry content as CSV."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'notebook_id', 
            'notebook_title', 
            'page_number', 
            'page_consensus', 
            'poetry_percentage',
            'all_classifications_json'
        ])
        
        # Sort notebooks by ID
        sorted_notebooks = sorted(poetry_pages.items())
        
        # Write data
        for notebook_id, notebook_data in sorted_notebooks:
            notebook_title = notebook_data['notebook_title']
            
            for page_info in notebook_data['pages']:
                writer.writerow([
                    notebook_id,
                    notebook_title,
                    page_info['page_number'],
                    page_info['page_consensus'],
                    page_info['poetry_percentage'],
                    json.dumps(page_info['all_classifications'])  # Store classifications as JSON string
                ])

def main():
    """Main function to process poetry classifications."""
    # Get project root (script is in poetry_filter/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Paths
    classifications_dir = project_root / "classifications"
    output_dir = project_root / "poetry_files"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Output files
    overall_poetry_file = output_dir / "overall_poetry_notebooks.txt"
    poetry_pages_file = output_dir / "poetry_pages.txt"
    overall_poetry_csv_file = output_dir / "overall_poetry_notebooks.csv"
    poetry_pages_csv_file = output_dir / "poetry_pages.csv"
    
    print("Starting poetry classification analysis...")
    print(f"Input directory: {classifications_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Check if classifications directory exists
    if not classifications_dir.exists():
        print(f"Error: Classifications directory not found: {classifications_dir}")
        return
    
    # Extract poetry data
    print("Analyzing classification data...")
    poetry_notebooks, poetry_pages = extract_poetry_notebooks_and_pages(classifications_dir)
    
    # Save results
    print(f"Saving overall poetry notebooks to: {overall_poetry_file}")
    save_overall_poetry_notebooks(poetry_notebooks, overall_poetry_file)
    
    print(f"Saving poetry pages to: {poetry_pages_file}")
    save_poetry_pages(poetry_pages, poetry_pages_file)
    
    # Save CSV versions
    print(f"Saving overall poetry notebooks CSV to: {overall_poetry_csv_file}")
    save_overall_poetry_notebooks_csv(poetry_notebooks, overall_poetry_csv_file)
    
    print(f"Saving poetry pages CSV to: {poetry_pages_csv_file}")
    save_poetry_pages_csv(poetry_pages, poetry_pages_csv_file)
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Notebooks with overall poetry consensus: {len(poetry_notebooks)}")
    print(f"Notebooks with poetry pages: {len(poetry_pages)}")
    
    total_poetry_pages = sum(notebook_data['total_poetry_pages'] for notebook_data in poetry_pages.values())
    print(f"Total poetry pages across all notebooks: {total_poetry_pages}")
    
    print(f"\nFiles saved:")
    print(f"  - {overall_poetry_file}")
    print(f"  - {poetry_pages_file}")
    print(f"  - {overall_poetry_csv_file}")
    print(f"  - {poetry_pages_csv_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
