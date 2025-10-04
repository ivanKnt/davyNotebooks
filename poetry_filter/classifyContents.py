import os
import json
from pathlib import Path
from collections import Counter, defaultdict


def load_classifications(notebook_path):
    """Load classifications.json for a notebook."""
    classifications_file = notebook_path / "classifications.json"
    if not classifications_file.exists():
        print(f"Warning: classifications.json not found for {notebook_path}")
        return None

    try:
        with open(classifications_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {classifications_file}: {e}")
        return None


def process_page_classifications(page_data):
    """Process a single page's classifications into percentages and consensus."""
    if isinstance(page_data, list):
        # Handle list format: ['Electrochemistry', 'Lecture notes', ...]
        classification_counts = Counter(page_data)
        total_classifications = len(page_data)

        if total_classifications == 0:
            return {"page_consensus": "unknown"}

        # Calculate percentages
        processed_data = {}
        for classification, count in classification_counts.items():
            percentage = count / total_classifications
            processed_data[classification] = round(percentage, 3)

        # Determine consensus (highest percentage)
        consensus = max(classification_counts.keys(), key=lambda x: classification_counts[x])
        processed_data["page_consensus"] = consensus.lower()

        return processed_data

    elif isinstance(page_data, dict):
        # Handle dictionary format: {"Electrochemistry": 0.857, "Poetry": 0.143}
        # This is pre-aggregated data, normalize and find consensus
        processed_data = {}

        # Normalize any numeric values to percentages if needed
        for classification, value in page_data.items():
            if isinstance(value, (int, float)):
                # Convert to decimal if it's a percentage (0-100), otherwise assume it's already decimal (0-1)
                if value > 1:
                    processed_data[classification] = round(value / 100, 3)
                else:
                    processed_data[classification] = round(value, 3)
            else:
                # If not numeric, skip this classification
                continue

        if not processed_data:
            return {"page_consensus": "unknown"}

        # Find consensus (highest percentage)
        consensus = max(processed_data.keys(), key=lambda x: processed_data[x])
        processed_data["page_consensus"] = consensus.lower()

        return processed_data

    else:
        # Unknown format
        return {"page_consensus": "unknown"}


def calculate_book_consensus(pages_data):
    """Calculate the overall book consensus based on page consensuses."""
    page_consensuses = []

    for page_data in pages_data.values():
        if isinstance(page_data, dict) and "page_consensus" in page_data:
            page_consensuses.append(page_data["page_consensus"])

    if not page_consensuses:
        return "unknown"

    # Find most common consensus
    consensus_counts = Counter(page_consensuses)
    book_consensus = max(consensus_counts.keys(), key=lambda x: consensus_counts[x])

    return book_consensus


def create_classifications_page_file(notebook_id, original_data, output_path):
    """Create the classifications_page.json file."""
    # Extract notebook title
    notebook_title = original_data.get("overall_classification", f"Notebook {notebook_id}")

    # Get per_page_classifications
    per_page_data = original_data.get("per_page_classifications", {})

    # Process each page
    pages_data = {}
    page_keys = []

    for key, value in per_page_data.items():
        try:
            # Try to convert to int for sorting
            page_num = int(key)
            page_keys.append((page_num, key))
        except ValueError:
            # If not a number, just add it
            page_keys.append((0, key))  # Put non-numeric keys first

    # Sort pages by page number
    page_keys.sort(key=lambda x: x[0])

    for _, page_key in page_keys:
        page_data = per_page_data[page_key]
        processed_page = process_page_classifications(page_data)
        if processed_page:  # Only add if processing succeeded
            pages_data[page_key] = processed_page

    # Calculate book consensus
    book_consensus = calculate_book_consensus(pages_data)

    # Create final structure
    result = {
        "notebook_title": notebook_title,
        "consensus_book": book_consensus,
        **pages_data
    }

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def create_summary_file(notebook_id, processed_data, output_path):
    """Create the summary.txt file."""
    notebook_title = processed_data.get("notebook_title", f"Notebook {notebook_id}")
    book_consensus = processed_data.get("consensus_book", "unknown")

    # Calculate overall statistics
    all_classifications = []
    page_consensuses = []

    for key, value in processed_data.items():
        if key not in ["notebook_title", "consensus_book"] and isinstance(value, dict):
            # Collect all classifications with their weights
            for classification, percentage in value.items():
                if classification != "page_consensus":
                    # Add classification multiple times based on percentage (approximate)
                    weight = int(percentage * 100)  # Convert to integer weight
                    all_classifications.extend([classification] * weight)
                elif classification == "page_consensus":
                    page_consensuses.append(value[classification])

    # Calculate percentages across the book
    if all_classifications:
        classification_counts = Counter(all_classifications)
        total_weighted = sum(classification_counts.values())

        # Get top classifications
        top_classifications = classification_counts.most_common(5)  # Top 5
    else:
        top_classifications = []

    # Count total pages processed
    total_pages = sum(1 for key, value in processed_data.items()
                     if key not in ["notebook_title", "consensus_book"] and isinstance(value, dict))

    # Write summary
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"CLASSIFICATION SUMMARY - {notebook_title.upper()}\n")
        f.write("=" * 60 + "\n\n")

        f.write("NOTEBOOK INFORMATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Notebook ID: {notebook_id}\n")
        f.write(f"Title: {notebook_title}\n")
        f.write(f"Overall Consensus: {book_consensus.title()}\n")
        f.write(f"Total Pages Processed: {total_pages}\n\n")

        if top_classifications:
            f.write("CLASSIFICATION DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            f.write("The notebook contains the following classifications:\n\n")

            for classification, count in top_classifications:
                percentage = (count / total_weighted) * 100
                f.write(f"• {classification}: {percentage:.1f}%\n")

            f.write("\n")

            # Additional insights
            if len(top_classifications) > 1:
                dominant = top_classifications[0][0]
                dominant_pct = (top_classifications[0][1] / total_weighted) * 100
                f.write(f"• The notebook is primarily focused on {dominant.lower()} content ")
                f.write(f"({dominant_pct:.1f}%)")
                if dominant_pct < 50:
                    f.write(", though with significant diversity in other areas.\n")
                else:
                    f.write(", showing strong thematic consistency.\n")

        if page_consensuses:
            f.write("\nPAGE CONSENSUS DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            consensus_counts = Counter(page_consensuses)
            for consensus, count in consensus_counts.most_common():
                pct = (count / len(page_consensuses)) * 100
                f.write(f"• {consensus.title()}: {pct:.1f}% ({count} pages)\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 60 + "\n")


def main():
    """Main function to process all notebook classifications."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Paths
    preprocessing_dir = project_root / "preprocessing"
    output_base_dir = project_root / "classifications"

    # Ensure output directory exists
    output_base_dir.mkdir(exist_ok=True)

    print("Starting classification processing...")
    print(f"Input directory: {preprocessing_dir}")
    print(f"Output directory: {output_base_dir}")
    print("=" * 60)

    # Find all notebook folders
    if not preprocessing_dir.exists():
        print(f"Error: Preprocessing directory not found: {preprocessing_dir}")
        return

    notebook_folders = [f for f in preprocessing_dir.iterdir() if f.is_dir()]
    notebook_folders.sort()  # Sort by name

    processed_count = 0

    for notebook_folder in notebook_folders:
        notebook_id = notebook_folder.name
        print(f"\nProcessing notebook: {notebook_id}")

        # Load classifications
        classifications_data = load_classifications(notebook_folder)
        if classifications_data is None:
            continue

        # Create output subdirectory
        notebook_output_dir = output_base_dir / notebook_id
        notebook_output_dir.mkdir(exist_ok=True)

        # Create classifications_page.json
        page_file_path = notebook_output_dir / "classifications_page.json"
        create_classifications_page_file(notebook_id, classifications_data, page_file_path)
        print(f"  ✓ Created: classifications_page.json")

        # Load the processed data for summary
        with open(page_file_path, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)

        # Create summary.txt
        summary_file_path = notebook_output_dir / "summary.txt"
        create_summary_file(notebook_id, processed_data, summary_file_path)
        print(f"  ✓ Created: summary.txt")

        processed_count += 1

    print(f"\n{'=' * 60}")
    print(f"Processing complete! Processed {processed_count} notebooks.")
    print(f"Files saved to: {output_base_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
