import os
import argparse
import json
import logging
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

# Configure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GreedyStringTiling:
    """Implementation of Greedy String Tiling (GST) algorithm for text similarity detection."""

    def __init__(self, min_match_length=3):
        """Initialize GST with minimum match length.

        Args:
            min_match_length (int): Minimum length of matches to consider
        """
        self.min_match_length = min_match_length
        self.matches = []

    def compute_similarity(self, tokens1, tokens2):
        """Compute GST similarity between two token sequences.

        Args:
            tokens1, tokens2 (list): Lists of tokens to compare

        Returns:
            dict: Similarity metrics and match information
        """
        if not tokens1 or not tokens2:
            return self._empty_result()

        seq1, seq2 = tokens1.copy(), tokens2.copy()
        total_match_length = self._greedy_string_tiling(seq1, seq2)

        len1, len2 = len(tokens1), len(tokens2)
        coverage1 = total_match_length / len1 if len1 > 0 else 0.0
        coverage2 = total_match_length / len2 if len2 > 0 else 0.0

        return {
            'gst_similarity': (coverage1 + coverage2) / 2,
            'total_match_length': total_match_length,
            'matches_found': len(self.matches),
            'seq1_length': len1,
            'seq2_length': len2,
            'matches': self.matches.copy()
        }

    def _greedy_string_tiling(self, seq1, seq2):
        """Core GST algorithm implementation.

        Args:
            seq1, seq2 (list): Token sequences to compare

        Returns:
            int: Total length of all matches found
        """
        self.matches = []
        total_match_length = 0
        marked1 = [False] * len(seq1)
        marked2 = [False] * len(seq2)

        while True:
            max_match = self._find_longest_match(seq1, seq2, marked1, marked2)
            if max_match is None or max_match['length'] < self.min_match_length:
                break
            self._mark_match(marked1, marked2, max_match)
            self.matches.append({
                'length': max_match['length'],
                'pos1': max_match['pos1'],
                'pos2': max_match['pos2'],
                'tokens': seq1[max_match['pos1']:max_match['pos1'] + max_match['length']]
            })
            total_match_length += max_match['length']
        return total_match_length

    def _find_longest_match(self, seq1, seq2, marked1, marked2):
        """Find the longest unmarked common substring.

        Args:
            seq1, seq2 (list): Token sequences
            marked1, marked2 (list): Boolean arrays indicating marked tokens

        Returns:
            dict: Information about the longest match, or None if no match found
        """
        max_match = None
        max_length = 0

        for i in range(len(seq1)):
            if marked1[i]:
                continue
            for j in range(len(seq2)):
                if marked2[j]:
                    continue
                if seq1[i] == seq2[j]:
                    length = self._extend_match(seq1, seq2, marked1, marked2, i, j)
                    if length > max_length:
                        max_length = length
                        max_match = {'pos1': i, 'pos2': j, 'length': length}
        return max_match

    def _extend_match(self, seq1, seq2, marked1, marked2, start1, start2):
        """Extend a match from given starting positions.

        Args:
            seq1, seq2 (list): Token sequences
            marked1, marked2 (list): Boolean arrays indicating marked tokens
            start1, start2 (int): Starting positions for the match

        Returns:
            int: Length of the extended match
        """
        length = 0
        i, j = start1, start2
        while (i < len(seq1) and j < len(seq2) and
               not marked1[i] and not marked2[j] and seq1[i] == seq2[j]):
            length += 1
            i += 1
            j += 1
        return length

    def _mark_match(self, marked1, marked2, match):
        """Mark all tokens in a match as used.

        Args:
            marked1, marked2 (list): Boolean arrays to update
            match (dict): Match information containing position and length
        """
        start1, start2, length = match['pos1'], match['pos2'], match['length']
        for i in range(length):
            marked1[start1 + i] = True
            marked2[start2 + i] = True

    def _empty_result(self):
        """Return empty result for invalid inputs."""
        return {
            'gst_similarity': 0.0,
            'total_match_length': 0,
            'matches_found': 0,
            'seq1_length': 0,
            'seq2_length': 0,
            'matches': []
        }


class LibraryBasedGSTDetector:
    """GST-based text reuse detector for 18th-century historical texts."""

    def __init__(self, similarity_threshold=0.3, min_match_length=3,
                 use_stemming=False, remove_stopwords=True,
                 min_segment_length=20, min_words=3):
        """Initialize the GST detector with configurable parameters.

        Args:
            similarity_threshold (float): Minimum GST similarity for reuse
            min_match_length (int): Minimum length of GST matches
            use_stemming (bool): Whether to apply stemming
            remove_stopwords (bool): Whether to remove stopwords
            min_segment_length (int): Minimum character length for segments
            min_words (int): Minimum word count for segments
        """
        self.similarity_threshold = similarity_threshold
        self.min_match_length = min_match_length
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.min_segment_length = min_segment_length
        self.min_words = min_words

        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.gst = GreedyStringTiling(min_match_length=min_match_length)

        self.metrics = {
            'processing_time': 0,
            'total_comparisons': 0,
            'total_segments': 0,
            'gst_computation_time': 0,
            'similarity_calculation_time': 0
        }

    def load_texts(self, base_dir, notebooks, filenames):
        """Load texts and metadata with error handling.

        Args:
            base_dir (str): Base directory containing notebooks
            notebooks (list): List of notebook directories
            filenames (list): List of filenames to process

        Returns:
            tuple: (texts dictionary, metadata dictionary)
        """
        start_time = time.time()
        texts = {}
        all_metadata = {}

        for notebook in notebooks:
            notebook_dir = base_dir / notebook
            if not notebook_dir.exists():
                logger.warning(f"Directory {notebook_dir} not found, skipping.")
                continue

            metadata_file = notebook_dir / 'page_to_entities.json'
            all_metadata[notebook] = self._load_metadata(metadata_file)

            for filename in filenames:
                file_path = notebook_dir / filename
                if file_path.exists():
                    content = self._load_file_content(file_path)
                    if content:
                        texts[(notebook, str(file_path))] = content

        logger.info(f"Text loading completed in {time.time() - start_time:.2f} seconds")
        return texts, all_metadata

    def _load_metadata(self, metadata_file):
        """Load metadata from JSON file."""
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata from {metadata_file}: {e}")
        logger.warning(f"Metadata file not found: {metadata_file}")
        return {}

    def _load_file_content(self, file_path):
        """Load and process file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip() and str(file_path).endswith(".json"):
                    return {k: str(v) for k, v in json.loads(content).items() if str(v).strip()}
                logger.info(f"Skipping non-JSON file: {file_path}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
        return None

    def preprocess_text_advanced(self, text):
        """Preprocess text for 18th-century historical texts.

        Args:
            text (str): Input text to preprocess

        Returns:
            list: Preprocessed tokens for GST processing
        """
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\|&:|_|"|xxx', '', text)
        text = re.sub(r'\bf\b', 's', text)
        text = re.sub(r'\bye\b', 'the', text)
        text = re.sub(r'\bvs\b', 'us', text)
        text = text.lower()

        words = word_tokenize(text)

        if self.remove_stopwords:
            modern_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [word for word in words if word.lower() not in modern_stopwords]

        if self.use_stemming and self.stemmer:
            words = [self.stemmer.stem(word) for word in words]

        return [word for word in words if (word.isalpha() and len(word) > 1) or word in ['.', ',', ';', ':']]

    def segment_text_advanced(self, content_item, notebook_id, file_path):
        """Segment text using NLTK sentence tokenizer.

        Args:
            content_item (dict): Content to segment
            notebook_id (str): Notebook identifier
            file_path (str): File path

        Returns:
            list: List of segment dictionaries
        """
        segments_data = []
        filename = os.path.basename(file_path)

        if isinstance(content_item, dict):
            for page_key, page_content in content_item.items():
                sentences = sent_tokenize(page_content)
                current_segment = ""
                segment_index = 0

                for sentence in sentences:
                    if len(current_segment + " " + sentence) < 500:
                        current_segment += " " + sentence if current_segment else sentence
                    else:
                        if current_segment:
                            self._add_segment(segments_data, current_segment, notebook_id, file_path, page_key,
                                              segment_index)
                            segment_index += 1
                        current_segment = sentence

                if current_segment:
                    self._add_segment(segments_data, current_segment, notebook_id, file_path, page_key, segment_index)

        logger.info(f"Segmented {notebook_id}/{filename}: {len(segments_data)} segments")
        return segments_data

    def _add_segment(self, segments_data, raw_segment, notebook_id, file_path, page_key, segment_index):
        """Add a segment if it meets criteria.

        Args:
            segments_data (list): List to store segments
            raw_segment (str): Raw segment text
            notebook_id (str): Notebook identifier
            file_path (str): File path
            page_key (str): Page identifier
            segment_index (int): Segment index on page
        """
        preprocessed_tokens = self.preprocess_text_advanced(raw_segment)
        if len(preprocessed_tokens) >= self.min_words and len(' '.join(preprocessed_tokens)) >= self.min_segment_length:
            segments_data.append({
                'tokens': preprocessed_tokens,
                'original_text': raw_segment.strip(),
                'notebook': notebook_id,
                'file_path': str(file_path),
                'page_key': page_key,
                'segment_index_on_page': segment_index
            })

    def calculate_gst_similarity_metrics(self, tokens1, tokens2):
        """Calculate comprehensive GST-based similarity metrics.

        Args:
            tokens1, tokens2 (list): Token sequences to compare

        Returns:
            dict: Comprehensive similarity metrics
        """
        start_time = time.time()
        if not tokens1 or not tokens2:
            return self._empty_metrics()

        gst_start = time.time()
        gst_result = self.gst.compute_similarity(tokens1, tokens2)
        self.metrics['gst_computation_time'] += time.time() - gst_start

        matches = gst_result['matches']
        match_texts = [' '.join(match['tokens']) for match in matches if match['tokens']]

        result = {
            **gst_result,
            'match_texts': match_texts,
            'avg_match_length': np.mean([m['length'] for m in matches]) if matches else 0,
            'max_match_length': max([m['length'] for m in matches]) if matches else 0,
            'min_match_length_found': min([m['length'] for m in matches]) if matches else 0
        }

        self.metrics['similarity_calculation_time'] += time.time() - start_time
        return result

    def _empty_metrics(self):
        """Return empty metrics for invalid cases."""
        return {
            'gst_similarity': 0.0,
            'total_match_length': 0,
            'matches_found': 0,
            'seq1_length': 0,
            'seq2_length': 0,
            'matches': [],
            'match_texts': [],
            'avg_match_length': 0,
            'max_match_length': 0,
            'min_match_length_found': 0
        }

    def find_text_reuse_optimized(self, texts_data, all_metadata):
        """Optimized GST-based text reuse detection.

        Args:
            texts_data (dict): Dictionary of text content
            all_metadata (dict): Dictionary of metadata

        Returns:
            list: List of reuse instances
        """
        start_time = time.time()
        all_segments_data = []

        for (notebook, file_path), content_item in texts_data.items():
            all_segments_data.extend(self.segment_text_advanced(content_item, notebook, file_path))

        self.metrics['total_segments'] = len(all_segments_data)
        logger.info(f"Total segments to analyze: {len(all_segments_data)}")

        reuse_instances = []
        comparisons_made = 0

        for i, j in combinations(range(len(all_segments_data)), 2):
            segment_data1 = all_segments_data[i]
            segment_data2 = all_segments_data[j]

            if (segment_data1['notebook'] == segment_data2['notebook'] and
                    segment_data1['page_key'] == segment_data2['page_key']):
                continue

            comparisons_made += 1
            metrics = self.calculate_gst_similarity_metrics(segment_data1['tokens'], segment_data2['tokens'])

            if metrics['gst_similarity'] >= self.similarity_threshold:
                metadata1 = all_metadata.get(segment_data1['notebook'], {}).get(str(segment_data1['page_key']), {})
                metadata2 = all_metadata.get(segment_data2['notebook'], {}).get(str(segment_data2['page_key']), {})

                reuse_instances.append({
                    'notebook1': segment_data1['notebook'],
                    'file1': os.path.basename(segment_data1['file_path']),
                    'segment1_index_on_page': segment_data1['segment_index_on_page'],
                    'segment1_page_key': segment_data1['page_key'],
                    'segment1_text': segment_data1['original_text'],
                    'segment1_metadata': metadata1,
                    'notebook2': segment_data2['notebook'],
                    'file2': os.path.basename(segment_data2['file_path']),
                    'segment2_index_on_page': segment_data2['segment_index_on_page'],
                    'segment2_page_key': segment_data2['page_key'],
                    'segment2_text': segment_data2['original_text'],
                    'segment2_metadata': metadata2,
                    'gst_similarity': metrics['gst_similarity'],
                    'total_match_length': metrics['total_match_length'],
                    'matches_found': metrics['matches_found'],
                    'avg_match_length': metrics['avg_match_length'],
                    'max_match_length': metrics['max_match_length']
                })

            if comparisons_made % 1000 == 0:
                logger.info(f"Processed {comparisons_made} comparisons...")

        self.metrics['total_comparisons'] = comparisons_made
        self.metrics['processing_time'] = time.time() - start_time

        logger.info(f"GST analysis completed: {comparisons_made} comparisons, "
                    f"{len(reuse_instances)} reuse instances found")
        logger.info(f"Total processing time: {self.metrics['processing_time']:.2f} seconds")
        return reuse_instances


def get_available_notebooks(base_dir: Path):
    try:
        return sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    except Exception:
        return []


def parse_notebooks_arg(notebooks_arg: str, base_dir: Path):
    if notebooks_arg.strip() == '*':
        return get_available_notebooks(base_dir)
    return [nb.strip() for nb in notebooks_arg.split(',') if nb.strip()]

    def calculate_comprehensive_metrics(self, reuse_instances, all_segments_count):
        """Calculate comprehensive summary metrics."""
        if not reuse_instances:
            return self._empty_summary_metrics(all_segments_count)

        gst_similarity_scores = [inst['gst_similarity'] for inst in reuse_instances]
        match_counts = [inst['matches_found'] for inst in reuse_instances]
        match_lengths = [inst['total_match_length'] for inst in reuse_instances]

        # Convert all NumPy results to native Python types
        return {
            'total_instances': len(reuse_instances),
            'total_segments_analyzed': all_segments_count,
            'total_comparisons_made': self.metrics['total_comparisons'],
            'gst_similarity_mean': float(np.mean(gst_similarity_scores)),
            'gst_similarity_std': float(np.std(gst_similarity_scores)),
            'gst_similarity_median': float(np.median(gst_similarity_scores)),
            'gst_similarity_min': float(np.min(gst_similarity_scores)),
            'gst_similarity_max': float(np.max(gst_similarity_scores)),
            'avg_matches_per_instance': float(np.mean(match_counts)),
            'avg_match_length': float(np.mean(match_lengths)),
            'max_match_length': int(np.max(match_lengths)),
            'total_matches_found': int(np.sum(match_counts)),
            'high_similarity_count': int(sum(1 for score in gst_similarity_scores if score >= 0.8)),
            'medium_similarity_count': int(sum(1 for score in gst_similarity_scores if 0.5 <= score < 0.8)),
            'low_similarity_count': int(sum(1 for score in gst_similarity_scores if score < 0.5)),
            'processing_time_seconds': float(self.metrics['processing_time']),
            'gst_computation_time_seconds': float(self.metrics['gst_computation_time']),
            'similarity_calculation_time_seconds': float(self.metrics['similarity_calculation_time']),
            'comparisons_per_second': float(self.metrics['total_comparisons'] / self.metrics['processing_time']
                                            if self.metrics['processing_time'] > 0 else 0),
            'reuse_rate': float(len(reuse_instances) / (all_segments_count * (all_segments_count - 1) / 2)
                                if all_segments_count > 1 else 0.0),
            'min_match_length': self.min_match_length,
            'similarity_threshold': float(self.similarity_threshold),
            'stemming_used': self.use_stemming,
            'stopwords_removed': self.remove_stopwords
        }

    def _empty_summary_metrics(self, all_segments_count):
        """Return empty summary metrics."""
        return {
            'total_instances': 0,
            'total_segments_analyzed': all_segments_count,
            'gst_similarity_mean': 0.0,
            'processing_time_seconds': float(self.metrics['processing_time']),
            'reuse_rate': 0.0,
            'min_match_length': self.min_match_length,
            'similarity_threshold': float(self.similarity_threshold)
        }
    def save_results_for_experiment(self, reuse_instances, summary_metrics, results_dir, filename_base, config_id):
        """Save results in multiple formats for experimental comparison.

        Args:
            reuse_instances (list): List of reuse instances
            summary_metrics (dict): Summary metrics
            results_dir (str): Directory to save results
            filename_base (str): Base filename for results
        """
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        config_folder = results_path / f"config_{config_id}"
        config_folder.mkdir(exist_ok=True)

        results_file = config_folder / f"{filename_base}_gst_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'method': 'gst_greedy_string_tiling',
                'configuration': {
                    'min_match_length': self.min_match_length,
                    'similarity_threshold': self.similarity_threshold,
                    'stemming_used': self.use_stemming,
                    'stopwords_removed': self.remove_stopwords
                },
                'reuse_instances': reuse_instances,
                'summary_metrics': summary_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

        # Save CSV instances
        if reuse_instances:
            csv_instances = []
            for instance in reuse_instances:
                csv_instance = {k: v for k, v in instance.items() if
                                not isinstance(v, (list, dict)) or k == 'match_texts'}
                if 'match_texts' in csv_instance:
                    csv_instance['match_texts'] = '; '.join(csv_instance['match_texts'])
                csv_instances.append(csv_instance)
            instances_df = pd.DataFrame(csv_instances)
            instances_df.to_csv(config_folder / f"{filename_base}_gst_instances.csv", index=False)

        # Save TXT report
        self._save_detailed_txt_report(reuse_instances, summary_metrics, config_folder, filename_base)

        logger.info(f"GST results saved to {config_folder}")
        return str(results_file)

    def _save_detailed_txt_report(self, reuse_instances, summary_metrics, results_dir, filename_base):
        """Save a detailed, human-readable TXT report."""
        with open(os.path.join(results_dir, f"{filename_base}_detailed_report.txt"), 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"GST TEXT REUSE DETECTION RESULTS - {filename_base.upper()}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: GST (Greedy String Tiling) Algorithm\n")
            f.write(f"Minimum Match Length: {self.min_match_length}\n")
            f.write(f"Similarity Threshold: {self.similarity_threshold}\n")
            f.write(f"Stemming Used: {self.use_stemming}\n")
            f.write(f"Stopwords Removed: {self.remove_stopwords}\n")
            f.write("=" * 80 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total text reuse instances found: {len(reuse_instances)}\n")
            f.write(f"Processing time: {summary_metrics['processing_time_seconds']:.2f} seconds\n")
            f.write(f"Segments analyzed: {summary_metrics['total_segments_analyzed']}\n")
            f.write(f"Comparisons made: {summary_metrics['total_comparisons_made']}\n")
            if len(reuse_instances) > 0:
                f.write(f"Average GST similarity: {summary_metrics['gst_similarity_mean']:.3f}\n")
                f.write(
                    f"GST similarity range: {summary_metrics['gst_similarity_min']:.3f} - {summary_metrics['gst_similarity_max']:.3f}\n")
                f.write(f"Average matches per instance: {summary_metrics['avg_matches_per_instance']:.1f}\n")
                f.write(f"Average match length: {summary_metrics['avg_match_length']:.1f} tokens\n")
                f.write(f"Total matches found: {summary_metrics['total_matches_found']}\n")
                f.write(f"High similarity instances (≥0.8): {summary_metrics['high_similarity_count']}\n")
                f.write(f"Medium similarity instances (0.5-0.8): {summary_metrics['medium_similarity_count']}\n")
                f.write(f"Low similarity instances (<0.5): {summary_metrics['low_similarity_count']}\n")
                f.write(f"Reuse rate: {summary_metrics['reuse_rate']:.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            if not reuse_instances:
                f.write("No text reuse instances found above the similarity threshold.\n")
                return

            sorted_instances = sorted(reuse_instances, key=lambda x: x['gst_similarity'], reverse=True)
            f.write("DETAILED TEXT REUSE INSTANCES\n")
            f.write("=" * 80 + "\n")
            f.write(f"Showing all {len(sorted_instances)} instances (sorted by GST similarity):\n\n")

            for i, instance in enumerate(sorted_instances, 1):
                f.write(f"Instance {i}:\n")
                f.write(f" Notebooks: {instance['notebook1']} ↔ {instance['notebook2']}\n")
                f.write(f" Files: {instance['file1']} ↔ {instance['file2']}\n")
                f.write(
                    f" Segments (on page): {instance['segment1_index_on_page']} ↔ {instance['segment2_index_on_page']}\n")
                if instance.get('segment1_page_key') and instance.get('segment2_page_key'):
                    f.write(f" Pages: {instance['segment1_page_key']} ↔ {instance['segment2_page_key']}\n")
                f.write(f" GST Similarity: {instance['gst_similarity']:.3f}\n")
                f.write(f" Total Match Length: {instance['total_match_length']} tokens\n")
                f.write(f" Matches Found: {instance['matches_found']}\n")
                f.write(f" Average Match Length: {instance['avg_match_length']:.1f} tokens\n")
                f.write(f" Max Match Length: {instance['max_match_length']} tokens\n")
                f.write(
                    f" Text 1 ({instance['notebook1']}): {instance['segment1_text'][:150]}{'...' if len(instance['segment1_text']) > 150 else ''}\n")
                f.write(
                    f" Text 2 ({instance['notebook2']}): {instance['segment2_text'][:150]}{'...' if len(instance['segment2_text']) > 150 else ''}\n")
                if instance.get('segment1_metadata'):
                    f.write(f" Metadata Text 1 ({instance['notebook1']}):\n")
                    for key, value in instance['segment1_metadata'].items():
                        f.write(f"  {key}: {value}\n")
                if instance.get('segment2_metadata'):
                    f.write(f" Metadata Text 2 ({instance['notebook2']}):\n")
                    for key, value in instance['segment2_metadata'].items():
                        f.write(f"  {key}: {value}\n")

                if instance.get('match_texts') and len(instance['match_texts']) > 0:
                    f.write(f" Matching Segments Found by GST:\n")
                    for j, match_text in enumerate(instance['match_texts'][:3]):
                        f.write(f" Match {j + 1}: {match_text[:80]}{'...' if len(match_text) > 80 else ''}\n")
                    if len(instance['match_texts']) > 3:
                        f.write(f" ... and {len(instance['match_texts']) - 3} more matches\n")
                f.write("-" * 80 + "\n")
                if i % 5 == 0 and i < len(sorted_instances):
                    f.write("\n")
            f.write("\nEND OF DETAILED REPORT\n")
            f.write("=" * 80 + "\n")


def main():
    """Main function for GST-based text reuse analysis."""
    parser = argparse.ArgumentParser(description="GST text reuse analysis")
    parser.add_argument('--notebooks', type=str, default='*',
                        help="Comma-separated notebook IDs (e.g., 14e,14g) or * for all")
    parser.add_argument('--combo-size', type=str, default='2',
                        help="2,3,4 to run over combinations of that size, or 'all' to use all selected notebooks")
    parser.add_argument('--filenames', type=str, default='page_to_text.json',
                        help="Comma-separated filenames to process (default: page_to_text.json)")
    parser.add_argument('--config-id', type=int, default=None,
                        help="Specific config ID to run (optional - runs all configs if not specified)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    base_dir = project_root / "preprocessing"
    selected_notebooks = parse_notebooks_arg(args.notebooks, base_dir)
    if not selected_notebooks:
        logger.error("No notebooks selected. Check --notebooks argument or preprocessing directory.")
        return
    filenames = [f.strip() for f in args.filenames.split(',') if f.strip()]
    results_dir = project_root / "results_text_reuse" / "results_gst"

    all_configs = [
        (
            2,
            {'similarity_threshold': 0.3, 'min_match_length': 3, 'use_stemming': True, 'remove_stopwords': True,
             'min_segment_length': 25, 'min_words': 4}
        ),
        (
            3,
            {'similarity_threshold': 0.35, 'min_match_length': 4, 'use_stemming': False, 'remove_stopwords': False,
             'min_segment_length': 35, 'min_words': 5}
        ),
        (
            4,
            {'similarity_threshold': 0.25, 'min_match_length': 5, 'use_stemming': True, 'remove_stopwords': True,
             'min_segment_length': 30, 'min_words': 4}
        )
    ]

    # Filter configs if specific config_id is requested
    if args.config_id:
        configs = [c for c in all_configs if c[0] == args.config_id]
        if not configs:
            print(f"Config ID {args.config_id} not found. Available configs: {[c[0] for c in all_configs]}")
            return
    else:
        configs = all_configs

    all_experiment_results = []
    print("Starting GST Analysis for 18th-Century Historical Text (Sir David Humphry)")
    if args.config_id:
        print(f"Running configuration {args.config_id}")
    else:
        print("Running configurations 2, 3, and 4")
    print("=" * 80)

    # Build notebook groups
    combo_size_arg = args.combo_size.strip().lower()
    if combo_size_arg == 'all':
        notebook_groups = [tuple(selected_notebooks)]
    else:
        try:
            k = int(combo_size_arg)
            if k <= 0 or k > len(selected_notebooks):
                k = min(2, len(selected_notebooks))
            notebook_groups = list(combinations(selected_notebooks, k))
        except ValueError:
            notebook_groups = [tuple(selected_notebooks)]

    for group in notebook_groups:
        group_list = list(group)
        group_tag = "__nb_" + '-'.join(group_list)

        for config_id, config in configs:
            config_name = f"Config {config_id}: GST min-match-{config['min_match_length']}"
            if config['use_stemming']:
                config_name += " + stemming"
            if config['remove_stopwords']:
                config_name += " + stopword removal"
            else:
                config_name += " (preserving stopwords)"

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running {config_name} on notebooks: {group_list}")
            logger.info(f"Configuration: {config}")
            logger.info(f"{'=' * 60}")

            detector = LibraryBasedGSTDetector(**config)
            texts_data, all_metadata = detector.load_texts(base_dir, group_list, filenames)

            if not texts_data:
                logger.error("No texts loaded. Check your configuration.")
                continue

            for filename in filenames:
                current_texts = {(nb, fp): content for (nb, fp), content in texts_data.items()
                                 if Path(fp).name == filename}
                if not current_texts:
                    logger.warning(f"No texts found for {filename}")
                    continue

                reuse_instances = detector.find_text_reuse_optimized(current_texts, all_metadata)
                total_segments = detector.metrics['total_segments']
                summary_metrics = detector.calculate_comprehensive_metrics(reuse_instances, total_segments)

                match_length = config['min_match_length']
                preprocessing = "stemmed" if config['use_stemming'] else "unstemmed"
                stopwords = "no_stopwords" if config['remove_stopwords'] else "with_stopwords"
                base_name = f"{os.path.splitext(filename)[0]}_gst_match{match_length}_{preprocessing}_{stopwords}{group_tag}"

                detector.save_results_for_experiment(reuse_instances, summary_metrics, results_dir, base_name, config_id)

                experiment_result = {
                    'config_id': config_id,
                    'config_name': config_name,
                    'min_match_length': config['min_match_length'],
                    'configuration': config,
                    'filename': filename,
                    'summary_metrics': summary_metrics,
                    'instance_count': len(reuse_instances),
                    'notebooks': group_list
                }
                all_experiment_results.append(experiment_result)

                print(f"\n{config_name} Results for {filename} [notebooks: {group_list}]::")
                print(f" Text reuse instances found: {len(reuse_instances)}")
                print(f" Processing time: {summary_metrics['processing_time_seconds']:.2f}s")
                print(f" Mean GST similarity: {summary_metrics['gst_similarity_mean']:.3f}")
                print(f" Segments analyzed: {summary_metrics['total_segments_analyzed']}")
                print(f" Reuse rate: {summary_metrics['reuse_rate']:.4f}")
                if len(reuse_instances) > 0:
                    print(
                        f" GST similarity range: {summary_metrics['gst_similarity_min']:.3f} - {summary_metrics['gst_similarity_max']:.3f}")
                    print(f" Average matches per instance: {summary_metrics['avg_matches_per_instance']:.1f}")
                    print(f" Total matches found: {summary_metrics['total_matches_found']}")

    print(f"\n{'=' * 80}")
    print("GST EXPERIMENT COMPLETED (Configurations 2, 3 & 4 Only)")
    print(f"{'=' * 80}")
    for result in all_experiment_results:
        metrics = result['summary_metrics']
        print(f"Config {result['config_id']} - {result['config_name']}")
        print(f" Text reuse instances found: {result['instance_count']}")
        print(f" Mean GST similarity: {metrics['gst_similarity_mean']:.3f}")
        print(f" Processing time: {metrics['processing_time_seconds']:.2f}s")
        print(f" Comparisons made: {metrics['total_comparisons_made']}")


if __name__ == "__main__":
    main()