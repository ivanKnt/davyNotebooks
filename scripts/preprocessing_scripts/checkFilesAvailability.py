"""
DAVY NOTEBOOKS FILE AVAILABILITY SCANNER
Focused on file availability analysis for Davy Notebooks collection

This program generates:
1. File availability summary across all notebooks
2. Missing file identification and reporting

Usage: python checkFilesAvailability.py
"""

import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class DavyNotebooksFileScanner:
    """File availability scanner for Davy Notebooks collection"""

    def __init__(self, repo_path=None):
        if repo_path is None:
            # Try to find the project root by going up from the script location
            script_dir = Path(__file__).parent
            # Check if we're in scripts/preprocessing_scripts, if so go up two levels
            if script_dir.name == "preprocessing_scripts" and script_dir.parent.name == "scripts":
                self.repo_path = script_dir.parent.parent
            else:
                # Assume we're running from project root
                self.repo_path = Path(".")
        else:
            self.repo_path = Path(repo_path)
        self.items_path = self.repo_path / "items"
        self.results = {}
        self.start_time = datetime.now()

        print("📄 DAVY NOTEBOOKS FILE AVAILABILITY SCANNER")
        print("=" * 50)
        print(f"Repository path: {self.repo_path.absolute()}")
        print(f"Analysis started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    def run_file_scan(self):
        """Run file availability scan"""

        # 1. Get all notebooks
        print("📊 Step 1: Getting notebook list...")
        notebooks = self.get_notebook_list()

        # 2. File availability analysis
        print("📄 Step 2: File Availability Analysis...")
        self.results['file_availability'] = self.analyze_file_availability(notebooks)

        # 3. Generate reports
        print("📋 Step 3: Generating Reports...")
        self.generate_file_scan_results()
        self.generate_scan_summary()

        print("\n✅ File Scan Complete!")
        print(f"Duration: {datetime.now() - self.start_time}")

        return self.results

    def get_notebook_list(self):
        """Get list of all notebook directories"""

        if not self.items_path.exists():
            print(f"Error: Items directory not found: {self.items_path}")
            return []

        # Get all notebook directories
        notebooks = sorted([d.name for d in self.items_path.iterdir() if d.is_dir()])

        print(f"   Found {len(notebooks)} notebooks")

        return notebooks

    def analyze_notebook_files(self, notebook_path):
        """Analyze file structure and availability"""

        # Define expected file paths
        expected_files = {
            'metadata_xml': notebook_path / 'config/metadata/metadata.xml',
            'valid_text': notebook_path / 'transcription/text/valid',
            'tagged_text': notebook_path / 'transcription/text/tagged',
            'tei_doc': notebook_path / 'tei/doc',
            'zoo_files': notebook_path / 'transcription/text/zoo',  # Added zoo files
            'classifications': notebook_path / 'transcription/source/classifications'  # Added classifications file
        }

        file_analysis = {}

        for file_type, file_path in expected_files.items():
            if file_path.exists():
                try:
                    stat = file_path.stat()
                    file_analysis[file_type] = {
                        'exists': True,
                        'size': stat.st_size,
                        'path': str(file_path.relative_to(notebook_path))
                    }
                except Exception as e:
                    file_analysis[file_type] = {
                        'exists': True,
                        'size': 0,
                        'error': str(e),
                        'path': str(file_path.relative_to(notebook_path))
                    }
            else:
                file_analysis[file_type] = {
                    'exists': False,
                    'size': 0,
                    'path': str(file_path.relative_to(notebook_path))
                }

        # Count transcription data files (renamed to individual_transcriptions_files)
        transcription_data_path = notebook_path / 'transcription/data'
        individual_transcriptions_files = 0
        if transcription_data_path.exists():
            try:
                individual_transcriptions_files = len([f for f in transcription_data_path.iterdir() if f.is_file()])
            except:
                individual_transcriptions_files = 0

        file_analysis['individual_transcriptions_files'] = individual_transcriptions_files

        return file_analysis


    def analyze_file_availability(self, notebooks):
        """Analyze file availability across the collection"""

        availability_stats = defaultdict(int)
        total_notebooks = len(notebooks)
        notebook_file_info = {}

        print(f"   Analyzing file availability for {total_notebooks} notebooks...")

        for i, notebook_id in enumerate(notebooks, 1):
            if i % 10 == 0 or i == total_notebooks:
                print(f"   Processed {i}/{total_notebooks} notebooks...")

            notebook_path = self.items_path / notebook_id
            file_structure = self.analyze_notebook_files(notebook_path)
            notebook_file_info[notebook_id] = file_structure

            for file_type, file_info in file_structure.items():
                if file_type != 'individual_transcriptions_files':
                    if file_info['exists']:
                        availability_stats[file_type] += 1

        # Calculate percentages
        availability_percentages = {}
        for file_type, count in availability_stats.items():
            availability_percentages[file_type] = {
                'available': count,
                'total': total_notebooks,
                'percentage': round((count / total_notebooks) * 100, 1)
            }

        # Identify missing files
        missing_files = defaultdict(list)
        for notebook_id, file_structure in notebook_file_info.items():
            for file_type, file_info in file_structure.items():
                if file_type != 'individual_transcriptions_files':
                    if not file_info['exists']:
                        missing_files[file_type].append(notebook_id)

        return {
            'availability_stats': availability_percentages,
            'missing_files': dict(missing_files),
            'notebook_file_info': notebook_file_info,
            'total_notebooks': total_notebooks
        }

    def generate_file_scan_results(self):
        """Generate file scan results report with individual notebook status"""

        with open(self.repo_path / 'file_scan_output/file_scan_results.txt', 'w', encoding='utf-8') as f:
            f.write("DAVY NOTEBOOKS INDIVIDUAL NOTEBOOK FILE STATUS\n")
            f.write("=" * 48 + "\n\n")

            f.write(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Notebooks Scanned: {self.results['file_availability']['total_notebooks']}\n\n")

            file_types = [
                'metadata_xml', 'valid_text', 'tagged_text', 'tei_doc',
                'zoo_files', 'classifications'
            ]

            for notebook_id in sorted(self.results['file_availability']['notebook_file_info'].keys()):
                file_info = self.results['file_availability']['notebook_file_info'][notebook_id]

                f.write(f"{notebook_id}:\n")

                # File status
                present_files = []
                missing_files_list = []

                for file_type in file_types:
                    if file_type in file_info and file_info[file_type]['exists']:
                        present_files.append(file_type)
                    else:
                        missing_files_list.append(file_type)

                if present_files:
                    f.write(f"  Present: {', '.join(present_files)}\n")
                if missing_files_list:
                    f.write(f"  Missing: {', '.join(missing_files_list)}\n")

                # Individual transcriptions count
                individual_count = file_info['individual_transcriptions_files']
                if individual_count > 0:
                    f.write(f"  Individual transcriptions: {individual_count} files\n")

                f.write("\n")

    def generate_scan_summary(self):
        """Generate scan summary with summary statistics"""

        with open(self.repo_path / 'file_scan_output/scan_summary.txt', 'w', encoding='utf-8') as f:
            f.write("DAVY NOTEBOOKS FILE AVAILABILITY SCAN RESULTS\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Notebooks Scanned: {self.results['file_availability']['total_notebooks']}\n\n")

            # File availability summary
            f.write("FILE AVAILABILITY SUMMARY:\n")
            f.write("-" * 26 + "\n")
            availability = self.results['file_availability']['availability_stats']

            file_types = [
                'metadata_xml', 'valid_text', 'tagged_text', 'tei_doc',
                'zoo_files', 'classifications'
            ]

            for file_type in file_types:
                if file_type in availability:
                    stats = availability[file_type]
                    f.write(f"{file_type}: {stats['available']}/{stats['total']} ({stats['percentage']}% available)\n")

            # Individual transcriptions files count summary
            f.write("\nINDIVIDUAL_TRANSCRIPTIONS_FILES SUMMARY:\n")
            f.write("-" * 40 + "\n")

            total_individual_files = 0
            notebooks_with_individual_files = 0
            max_individual_files = 0
            max_individual_notebook = ""

            for notebook_id, file_info in self.results['file_availability']['notebook_file_info'].items():
                count = file_info['individual_transcriptions_files']
                total_individual_files += count
                if count > 0:
                    notebooks_with_individual_files += 1
                if count > max_individual_files:
                    max_individual_files = count
                    max_individual_notebook = notebook_id

            f.write(f"Total individual transcription files across all notebooks: {total_individual_files}\n")
            f.write(f"Notebooks with individual transcription files: {notebooks_with_individual_files}/{self.results['file_availability']['total_notebooks']}\n")
            f.write(f"Average files per notebook: {total_individual_files / self.results['file_availability']['total_notebooks']:.1f}\n")
            f.write(f"Maximum files in a single notebook: {max_individual_files} ({max_individual_notebook})\n\n")

            # Missing files by type
            f.write("MISSING FILES BY TYPE:\n")
            f.write("-" * 23 + "\n")

            missing_files = self.results['file_availability']['missing_files']
            for file_type in file_types:
                if file_type in missing_files and missing_files[file_type]:
                    missing_list = missing_files[file_type]
                    f.write(f"\n{file_type} - Missing from {len(missing_list)} notebooks:\n")

                    # Group notebooks for readability
                    for i in range(0, len(missing_list), 10):
                        batch = missing_list[i:i+10]
                        f.write(f"  {', '.join(batch)}\n")

def main():
    """Main execution function"""

    # Initialize scanner
    scanner = DavyNotebooksFileScanner()

    # Run file scan
    results = scanner.run_file_scan()

    print("\n" + "=" * 50)
    print("📋 FILE SCAN COMPLETE - FILES GENERATED IN file_scan_output/:")
    print("=" * 50)
    print("✅ file_scan_results.txt - Individual notebook status")
    print("✅ scan_summary.txt - File availability summary")
    print("\n🎯 Ready for file availability analysis!")

    return results


if __name__ == "__main__":
    main()

