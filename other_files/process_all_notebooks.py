import os
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_notebook(tei_file_path):
    """Run extract_tei_text.py on a single TEI XML file."""
    try:
        # Run the extract_tei_text.py script as a subprocess
        result = subprocess.run(
            ['python', 'extract_tei_text.py', tei_file_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully processed: {tei_file_path}")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {tei_file_path}: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error processing {tei_file_path}: {e}")

def find_tei_files(items_dir):
    """Find all TEI XML files (doc) under items/ subdirectories."""
    tei_files = []
    for root, dirs, files in os.walk(items_dir):
        if 'tei' in dirs:
            tei_path = os.path.join(root, 'tei', 'doc')
            if os.path.isfile(tei_path):
                tei_files.append(tei_path)
    return tei_files

def process_all_notebooks():
    """Process all TEI XML files in the items/ directory."""
    items_dir = 'items'
    if not os.path.isdir(items_dir):
        logger.error(f"Items directory not found: {items_dir}")
        return

    tei_files = find_tei_files(items_dir)
    if not tei_files:
        logger.warning("No TEI XML files found in items/ directory.")
        return

    logger.info(f"Found {len(tei_files)} TEI XML files to process.")
    for tei_file in tei_files:
        process_notebook(tei_file)

    logger.info("Processing complete. Check logs for details.")

if __name__ == "__main__":
    process_all_notebooks()