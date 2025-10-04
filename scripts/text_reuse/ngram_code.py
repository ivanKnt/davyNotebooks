import os
import argparse
import re
import json
import logging
import time
from collections import defaultdict, Counter
from datetime import datetime
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path

# NLP and text processing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# spaCy is optional - only import if available
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy not available - continuing without it")

# Ensure required NLTK data is downloaded
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
        pass  # punkt_tab might not be available in older NLTK versions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LibraryBasedNgramDetector:
    """
    Library-based N-gram text reuse detector using NLTK, scikit-learn, and spaCy.
    Designed for experimental comparison with LCS, TF-IDF, and GST methods.
    """

    def __init__(self, n_gram_size=3, similarity_threshold=0.2,
                 use_stemming=False, remove_stopwords=True,
                 min_segment_length=20, min_words=3):
        """
        Initialize the detector with configurable parameters.

        Args:
            n_gram_size (int): Size of n-grams to generate
            similarity_threshold (float): Minimum similarity to consider as reuse
            use_stemming (bool): Whether to apply stemming
            remove_stopwords (bool): Whether to remove stopwords
            min_segment_length (int): Minimum character length for segments
            min_words (int): Minimum word count for segments
        """
        self.n_gram_size = n_gram_size
        self.similarity_threshold = similarity_threshold
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.min_segment_length = min_segment_length
        self.min_words = min_words

        # Initialize NLP tools
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()

        # Metrics tracking
        self.metrics = {
            'processing_time': 0,
            'total_comparisons': 0,
            'total_segments': 0,
            'memory_usage': 0,
            'ngram_generation_time': 0,
            'similarity_calculation_time': 0
        }

    def load_texts(self, base_dir, notebooks, filenames):
        """Load texts and metadata using improved error handling."""
        start_time = time.time()
        texts = {}
        all_metadata = {}

        for notebook in notebooks:
            notebook_dir = base_dir / notebook
            if not notebook_dir.exists():
                logger.warning(f"Directory {notebook_dir} not found, skipping.")
                continue

            metadata_file_path = notebook_dir / 'page_to_entities.json'
            if metadata_file_path.exists():
                try:
                    with open(metadata_file_path, 'r', encoding='utf-8') as f:
                        all_metadata[notebook] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading metadata from {metadata_file_path}: {e}")
                    all_metadata[notebook] = {}
            else:
                logger.warning(f"Metadata file not found for {notebook}")
                all_metadata[notebook] = {}

            for filename in filenames:
                file_path = notebook_dir / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if content.strip():
                            if filename.endswith(".json"):
                                data = json.loads(content)
                                texts[(notebook, str(file_path))] = {k: str(v) for k, v in data.items() if str(v).strip()}
                            else:
                                logger.info(f"Skipping non-JSON file: {file_path}")
                                continue
                        else:
                            logger.warning(f"Empty file: {file_path}")

                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")

        loading_time = time.time() - start_time
        logger.info(f"Text loading completed in {loading_time:.2f} seconds")
        return texts, all_metadata

    def preprocess_text_advanced(self, text):
        """
        Advanced text preprocessing using NLTK and regex.
        Optimized for 18th-century historical texts.

        Args:
            text (str): Raw text to preprocess

        Returns:
            str: Preprocessed text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove specific unwanted characters
        text = re.sub(r'\|&:|_|"|xxx', '', text)

        # Handle historical text issues - normalize archaic spellings and characters
        # Common 18th-century text normalization
        text = re.sub(r'\bf\b', 's', text)  # Long s to regular s
        text = re.sub(r'ye\b', 'the', text)  # ye -> the
        text = re.sub(r'\bvs\b', 'us', text)  # v -> u in some cases

        # Convert to lowercase
        text = text.lower()

        # Tokenize using NLTK
        words = word_tokenize(text)

        # Remove stopwords if specified (but be careful with historical text)
        if self.remove_stopwords:
            # Use a more conservative approach for historical text
            modern_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [word for word in words if word.lower() not in modern_stopwords]

        # Apply stemming if specified
        if self.use_stemming and self.stemmer:
            words = [self.stemmer.stem(word) for word in words]

        # Remove non-alphabetic tokens and short words, but keep some punctuation context
        words = [word for word in words if (word.isalpha() and len(word) > 1) or word in ['.', ',', ';', ':']]

        return ' '.join(words)

    def generate_ngrams_nltk(self, text, n):
        """
        Generate n-grams using NLTK.

        Args:
            text (str): Preprocessed text
            n (int): N-gram size

        Returns:
            list: List of n-gram tuples
        """
        start_time = time.time()
        words = text.split()

        if len(words) < n:
            return []

        # Use NLTK's ngrams function
        ngram_list = list(ngrams(words, n))

        # Convert tuples to strings for compatibility
        ngram_strings = [' '.join(gram) for gram in ngram_list]

        self.metrics['ngram_generation_time'] += time.time() - start_time
        return ngram_strings

    def segment_text_advanced(self, content_item, notebook_id, file_path):
        """
        Advanced text segmentation using NLTK sentence tokenizer.

        Args:
            content_item: Text content (dict for JSON files)
            notebook_id (str): Notebook identifier
            file_path (str): Path to the file

        Returns:
            list: List of segment dictionaries
        """
        segments_data = []
        filename = os.path.basename(file_path)

        if isinstance(content_item, dict):
            for page_key, page_content in content_item.items():
                # Use NLTK sentence tokenizer for better segmentation
                sentences = sent_tokenize(page_content)

                # Group sentences into larger segments (e.g., paragraphs)
                current_segment = ""
                segment_index = 0

                for sentence in sentences:
                    if len(current_segment + " " + sentence) < 500:  # Adjust segment size
                        current_segment += " " + sentence if current_segment else sentence
                    else:
                        if current_segment:
                            self._add_segment(segments_data, current_segment, notebook_id,
                                              file_path, page_key, segment_index)
                            segment_index += 1
                        current_segment = sentence

                # Add the last segment
                if current_segment:
                    self._add_segment(segments_data, current_segment, notebook_id,
                                      file_path, page_key, segment_index)

        logger.info(f"Segmented {notebook_id}/{filename}: {len(segments_data)} segments")
        return segments_data

    def _add_segment(self, segments_data, raw_segment, notebook_id, file_path, page_key, segment_index):
        """Helper method to add a segment if it meets criteria."""
        preprocessed_segment = self.preprocess_text_advanced(raw_segment)

        if (len(preprocessed_segment) >= self.min_segment_length and
                len(preprocessed_segment.split()) >= self.min_words):
            segments_data.append({
                'text': preprocessed_segment,
                'original_text': raw_segment.strip(),
                'notebook': notebook_id,
                'file_path': file_path,
                'page_key': page_key,
                'segment_index_on_page': segment_index
            })

    def calculate_similarity_metrics(self, text1, text2):
        """
        Calculate comprehensive similarity metrics using library functions.

        Args:
            text1, text2 (str): Preprocessed texts to compare

        Returns:
            dict: Dictionary of similarity metrics
        """
        start_time = time.time()

        # Generate n-grams using NLTK
        ngrams1 = set(self.generate_ngrams_nltk(text1, self.n_gram_size))
        ngrams2 = set(self.generate_ngrams_nltk(text2, self.n_gram_size))

        if not ngrams1 or not ngrams2:
            return self._empty_metrics()

        # Calculate set-based similarities
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)

        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0

        # Containment similarities
        containment_1_in_2 = len(intersection) / len(ngrams1) if ngrams1 else 0.0
        containment_2_in_1 = len(intersection) / len(ngrams2) if ngrams2 else 0.0

        # Dice coefficient (alternative similarity measure)
        dice = (2 * len(intersection)) / (len(ngrams1) + len(ngrams2)) if (ngrams1 or ngrams2) else 0.0

        # Cosine similarity using TF-IDF (for additional comparison)
        vectorizer = TfidfVectorizer(ngram_range=(self.n_gram_size, self.n_gram_size),
                                     token_pattern=r'(?u)\b\w+\b')
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_sim = 0.0

        self.metrics['similarity_calculation_time'] += time.time() - start_time

        return {
            'jaccard': jaccard,
            'containment_1_in_2': containment_1_in_2,
            'containment_2_in_1': containment_2_in_1,
            'max_containment': max(containment_1_in_2, containment_2_in_1),
            'dice_coefficient': dice,
            'cosine_similarity': cosine_sim,
            'intersection_size': len(intersection),
            'union_size': len(union),
            'ngrams1_count': len(ngrams1),
            'ngrams2_count': len(ngrams2)
        }

    def _empty_metrics(self):
        """Return empty metrics for cases where n-grams can't be generated."""
        return {
            'jaccard': 0.0, 'containment_1_in_2': 0.0, 'containment_2_in_1': 0.0,
            'max_containment': 0.0, 'dice_coefficient': 0.0, 'cosine_similarity': 0.0,
            'intersection_size': 0, 'union_size': 0, 'ngrams1_count': 0, 'ngrams2_count': 0
        }

    def find_text_reuse_optimized(self, texts_data, all_metadata):
        """
        Optimized text reuse detection with comprehensive metrics tracking.

        Args:
            texts_data (dict): Dictionary of loaded texts
            all_metadata (dict): Dictionary of metadata

        Returns:
            list: List of reuse instances with detailed metrics
        """
        start_time = time.time()

        # Generate all segments
        all_segments_data = []
        for (notebook, file_path), content_item in texts_data.items():
            segments = self.segment_text_advanced(content_item, notebook, file_path)
            all_segments_data.extend(segments)

        self.metrics['total_segments'] = len(all_segments_data)
        logger.info(f"Total segments to analyze: {len(all_segments_data)}")

        reuse_instances = []
        comparisons_made = 0

        # Use itertools.combinations for efficient pairwise comparison
        for i, j in combinations(range(len(all_segments_data)), 2):
            segment_data1 = all_segments_data[i]
            segment_data2 = all_segments_data[j]

            # Skip same notebook + same page comparisons
            if (segment_data1['notebook'] == segment_data2['notebook'] and
                    segment_data1['page_key'] == segment_data2['page_key']):
                continue

            comparisons_made += 1

            # Calculate similarity metrics
            metrics = self.calculate_similarity_metrics(
                segment_data1['text'], segment_data2['text']
            )

            # Check if similarity meets threshold
            if metrics['jaccard'] >= self.similarity_threshold:
                # Get metadata
                metadata1 = all_metadata.get(segment_data1['notebook'], {}).get(
                    str(segment_data1['page_key']), {}) if segment_data1['page_key'] else {}
                metadata2 = all_metadata.get(segment_data2['notebook'], {}).get(
                    str(segment_data2['page_key']), {}) if segment_data2['page_key'] else {}

                # Create detailed reuse instance
                instance = {
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

                    # All similarity metrics
                    **metrics
                }

                reuse_instances.append(instance)

            # Progress reporting
            if comparisons_made % 1000 == 0:
                logger.info(f"Processed {comparisons_made} comparisons...")

        self.metrics['total_comparisons'] = comparisons_made
        self.metrics['processing_time'] = time.time() - start_time

        logger.info(f"Analysis completed: {comparisons_made} comparisons, "
                    f"{len(reuse_instances)} reuse instances found")
        logger.info(f"Total processing time: {self.metrics['processing_time']:.2f} seconds")

        return reuse_instances

    def calculate_comprehensive_metrics(self, reuse_instances, all_segments_count):
        """
        Calculate comprehensive summary metrics for experimental analysis.

        Args:
            reuse_instances (list): List of detected reuse instances
            all_segments_count (int): Total number of segments analyzed

        Returns:
            dict: Comprehensive metrics dictionary
        """
        if not reuse_instances:
            return self._empty_summary_metrics(all_segments_count)

        # Extract all similarity scores for analysis
        jaccard_scores = [inst['jaccard'] for inst in reuse_instances]
        containment_scores = [inst['max_containment'] for inst in reuse_instances]
        dice_scores = [inst['dice_coefficient'] for inst in reuse_instances]
        cosine_scores = [inst['cosine_similarity'] for inst in reuse_instances]

        # Statistical measures
        metrics = {
            'total_instances': len(reuse_instances),
            'total_segments_analyzed': all_segments_count,
            'total_comparisons_made': self.metrics['total_comparisons'],

            # Jaccard statistics
            'jaccard_mean': np.mean(jaccard_scores),
            'jaccard_std': np.std(jaccard_scores),
            'jaccard_median': np.median(jaccard_scores),
            'jaccard_min': np.min(jaccard_scores),
            'jaccard_max': np.max(jaccard_scores),

            # Containment statistics
            'max_containment_mean': np.mean(containment_scores),
            'max_containment_std': np.std(containment_scores),
            'max_containment_median': np.median(containment_scores),

            # Additional similarity measures
            'dice_coefficient_mean': np.mean(dice_scores),
            'cosine_similarity_mean': np.mean(cosine_scores),

            # Similarity categories
            'high_similarity_count': sum(1 for score in jaccard_scores if score >= 0.8),
            'medium_similarity_count': sum(1 for score in jaccard_scores if 0.5 <= score < 0.8),
            'low_similarity_count': sum(1 for score in jaccard_scores if score < 0.5),

            # Performance metrics
            'processing_time_seconds': self.metrics['processing_time'],
            'ngram_generation_time_seconds': self.metrics['ngram_generation_time'],
            'similarity_calculation_time_seconds': self.metrics['similarity_calculation_time'],
            'comparisons_per_second': (self.metrics['total_comparisons'] /
                                       self.metrics['processing_time'] if self.metrics['processing_time'] > 0 else 0),

            # Reuse rates
            'reuse_rate': (len(reuse_instances) /
                           (all_segments_count * (all_segments_count - 1) / 2)
                           if all_segments_count > 1 else 0.0),
            'segments_with_reuse': len(set([inst['segment1_page_key'] for inst in reuse_instances] +
                                           [inst['segment2_page_key'] for inst in reuse_instances])),

            # Configuration used
            'n_gram_size': self.n_gram_size,
            'similarity_threshold': self.similarity_threshold,
            'stemming_used': self.use_stemming,
            'stopwords_removed': self.remove_stopwords
        }

        return metrics

    def _empty_summary_metrics(self, all_segments_count):
        """Return empty summary metrics."""
        return {
            'total_instances': 0,
            'total_segments_analyzed': all_segments_count,
            'total_comparisons_made': self.metrics['total_comparisons'],
            'jaccard_mean': 0.0,
            'max_containment_mean': 0.0,
            'processing_time_seconds': self.metrics['processing_time'],
            'reuse_rate': 0.0,
            'n_gram_size': self.n_gram_size,
            'similarity_threshold': self.similarity_threshold,
            'stemming_used': self.use_stemming,
            'stopwords_removed': self.remove_stopwords,
            'high_similarity_count': 0,
            'medium_similarity_count': 0,
            'low_similarity_count': 0,
            'comparisons_per_second': (self.metrics['total_comparisons'] /
                                       self.metrics['processing_time'] if self.metrics['processing_time'] > 0 else 0),
            'segments_with_reuse': 0
        }

    def save_results_for_experiment(self, reuse_instances, summary_metrics,
                                    results_dir, filename_base, config_id):
        """
        Save results in formats suitable for experimental comparison.
        Creates JSON, CSV, and detailed TXT files.

        Args:
            reuse_instances (list): List of reuse instances
            summary_metrics (dict): Summary metrics
            results_dir (str): Directory to save results
            filename_base (str): Base filename for output files
        """
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        config_folder = results_path / f"config_{config_id}"
        config_folder.mkdir(exist_ok=True)

        results_file = config_folder / f"{filename_base}_ngram_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'method': 'ngram_library_based',
                'configuration': {
                    'n_gram_size': self.n_gram_size,
                    'similarity_threshold': self.similarity_threshold,
                    'stemming_used': self.use_stemming,
                    'stopwords_removed': self.remove_stopwords
                },
                'reuse_instances': reuse_instances,
                'summary_metrics': summary_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

        if reuse_instances:
            instances_df = pd.DataFrame(reuse_instances)
            instances_file = config_folder / f"{filename_base}_ngram_instances.csv"
            instances_df.to_csv(instances_file, index=False)

        self._save_detailed_txt_report(reuse_instances, summary_metrics, config_folder, filename_base)

        logger.info(f"N-gram results saved to {config_folder}")
        return str(results_file)

    def _save_detailed_txt_report(self, reuse_instances, summary_metrics, results_dir, filename_base):
        """
        Save a detailed, human-readable TXT report of text reuse instances.
        """
        txt_file = os.path.join(results_dir, f"{filename_base}_detailed_report.txt")

        with open(txt_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(f"TEXT REUSE DETECTION RESULTS - {filename_base.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Method: N-gram Library-Based Detection\n")
            f.write(f"N-gram Size: {self.n_gram_size}\n")
            f.write(f"Similarity Threshold: {self.similarity_threshold}\n")
            f.write(f"Stemming Used: {self.use_stemming}\n")
            f.write(f"Stopwords Removed: {self.remove_stopwords}\n")
            f.write("=" * 80 + "\n\n")

            # Summary Section
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total text reuse instances found: {len(reuse_instances)}\n")
            f.write(f"Processing time: {summary_metrics['processing_time_seconds']:.2f} seconds\n")
            f.write(f"Segments analyzed: {summary_metrics['total_segments_analyzed']}\n")
            f.write(f"Comparisons made: {summary_metrics['total_comparisons_made']}\n")

            if len(reuse_instances) > 0:
                f.write(f"Average Jaccard similarity: {summary_metrics['jaccard_mean']:.3f}\n")
                f.write(f"Average max containment: {summary_metrics['max_containment_mean']:.3f}\n")
                f.write(
                    f"Similarity range: {summary_metrics['jaccard_min']:.3f} - {summary_metrics['jaccard_max']:.3f}\n")
                f.write(f"High similarity instances (≥0.8): {summary_metrics['high_similarity_count']}\n")
                f.write(f"Medium similarity instances (0.5-0.8): {summary_metrics['medium_similarity_count']}\n")
                f.write(f"Low similarity instances (<0.5): {summary_metrics['low_similarity_count']}\n")

            f.write(f"Reuse rate: {summary_metrics['reuse_rate']:.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            if not reuse_instances:
                f.write("No text reuse instances found above the similarity threshold.\n")
                return

            # Sort instances by similarity for better readability
            sorted_instances = sorted(reuse_instances, key=lambda x: x['jaccard'], reverse=True)

            # Detailed instances section
            f.write("DETAILED TEXT REUSE INSTANCES\n")
            f.write("=" * 80 + "\n")
            f.write(f"Showing all {len(sorted_instances)} instances (sorted by Jaccard similarity):\n\n")

            for i, instance in enumerate(sorted_instances, 1):
                f.write(f"Instance {i}:\n")
                f.write(f"  Notebooks: {instance['notebook1']} ↔ {instance['notebook2']}\n")
                f.write(f"  Files: {instance['file1']} ↔ {instance['file2']}\n")
                f.write(
                    f"  Segments (on page): {instance['segment1_index_on_page']} ↔ {instance['segment2_index_on_page']}\n")

                if instance.get('segment1_page_key') and instance.get('segment2_page_key'):
                    f.write(f"  Pages: {instance['segment1_page_key']} ↔ {instance['segment2_page_key']}\n")

                f.write(f"  Jaccard Similarity: {instance['jaccard']:.3f}\n")
                f.write(f"  Max Containment: {instance['max_containment']:.3f}\n")
                f.write(
                    f"  Containment ({instance['notebook1']} in {instance['notebook2']}): {instance['containment_1_in_2']:.3f}\n")
                f.write(
                    f"  Containment ({instance['notebook2']} in {instance['notebook1']}): {instance['containment_2_in_1']:.3f}\n")
                f.write(f"  Dice Coefficient: {instance['dice_coefficient']:.3f}\n")
                f.write(f"  Cosine Similarity: {instance['cosine_similarity']:.3f}\n")
                f.write(f"  Intersection Size: {instance['intersection_size']} n-grams\n")
                f.write(f"  Union Size: {instance['union_size']} n-grams\n")
                f.write(
                    f"  Text 1 ({instance['notebook1']}): {instance['segment1_text'][:150]}{'...' if len(instance['segment1_text']) > 150 else ''}\n")
                f.write(
                    f"  Text 2 ({instance['notebook2']}): {instance['segment2_text'][:150]}{'...' if len(instance['segment2_text']) > 150 else ''}\n")

                # Add metadata if available
                if instance.get('segment1_metadata'):
                    f.write(f"  Metadata 1: {json.dumps(instance['segment1_metadata'], indent=4)}\n")
                if instance.get('segment2_metadata'):
                    f.write(f"  Metadata 2: {json.dumps(instance['segment2_metadata'], indent=4)}\n")

                f.write("-" * 80 + "\n")

                # Add spacing every 5 instances for better readability
                if i % 5 == 0 and i < len(sorted_instances):
                    f.write("\n")

            f.write("\nEND OF DETAILED REPORT\n")
            f.write("=" * 80 + "\n")


def get_available_notebooks(base_dir: Path):
    try:
        return sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    except Exception:
        return []


def parse_notebooks_arg(notebooks_arg: str, base_dir: Path):
    if notebooks_arg.strip() == '*':
        return get_available_notebooks(base_dir)
    return [nb.strip() for nb in notebooks_arg.split(',') if nb.strip()]

    def segment_text_advanced(self, content_item, notebook_id, file_path):
        """
        Advanced text segmentation using NLTK sentence tokenizer.

        Args:
            content_item: Text content (dict for JSON files)
            notebook_id (str): Notebook identifier
            file_path (str): Path to the file

        Returns:
            list: List of segment dictionaries
        """
        segments_data = []
        filename = os.path.basename(file_path)

        if isinstance(content_item, dict):
            for page_key, page_content in content_item.items():
                # Use NLTK sentence tokenizer for better segmentation
                sentences = sent_tokenize(page_content)

                # Group sentences into larger segments (e.g., paragraphs)
                current_segment = ""
                segment_index = 0

                for sentence in sentences:
                    if len(current_segment + " " + sentence) < 500:  # Adjust segment size
                        current_segment += " " + sentence if current_segment else sentence
                    else:
                        if current_segment:
                            self._add_segment(segments_data, current_segment, notebook_id,
                                              file_path, page_key, segment_index)
                            segment_index += 1
                        current_segment = sentence

                # Add the last segment
                if current_segment:
                    self._add_segment(segments_data, current_segment, notebook_id,
                                      file_path, page_key, segment_index)

        logger.info(f"Segmented {notebook_id}/{filename}: {len(segments_data)} segments")
        return segments_data

    def _add_segment(self, segments_data, raw_segment, notebook_id, file_path, page_key, segment_index):
        """Helper method to add a segment if it meets criteria."""
        preprocessed_segment = self.preprocess_text_advanced(raw_segment)

        if (len(preprocessed_segment) >= self.min_segment_length and
                len(preprocessed_segment.split()) >= self.min_words):
            segments_data.append({
                'text': preprocessed_segment,
                'original_text': raw_segment.strip(),
                'notebook': notebook_id,
                'file_path': file_path,
                'page_key': page_key,
                'segment_index_on_page': segment_index
            })

    def generate_ngrams_nltk(self, text, n):
        """
        Generate n-grams using NLTK.

        Args:
            text (str): Preprocessed text
            n (int): N-gram size

        Returns:
            list: List of n-gram tuples
        """
        start_time = time.time()
        words = text.split()

        if len(words) < n:
            return []

        # Use NLTK's ngrams function
        ngram_list = list(ngrams(words, n))

        # Convert tuples to strings for compatibility
        ngram_strings = [' '.join(gram) for gram in ngram_list]

        self.metrics['ngram_generation_time'] += time.time() - start_time
        return ngram_strings

    def calculate_similarity_metrics(self, text1, text2):
        """
        Calculate comprehensive similarity metrics using library functions.

        Args:
            text1, text2 (str): Preprocessed texts to compare

        Returns:
            dict: Dictionary of similarity metrics
        """
        start_time = time.time()

        # Generate n-grams using NLTK
        ngrams1 = set(self.generate_ngrams_nltk(text1, self.n_gram_size))
        ngrams2 = set(self.generate_ngrams_nltk(text2, self.n_gram_size))

        if not ngrams1 or not ngrams2:
            return self._empty_metrics()

        # Calculate set-based similarities
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)

        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0

        # Containment similarities
        containment_1_in_2 = len(intersection) / len(ngrams1) if ngrams1 else 0.0
        containment_2_in_1 = len(intersection) / len(ngrams2) if ngrams2 else 0.0

        # Dice coefficient (alternative similarity measure)
        dice = (2 * len(intersection)) / (len(ngrams1) + len(ngrams2)) if (ngrams1 or ngrams2) else 0.0

        # Cosine similarity using TF-IDF (for additional comparison)
        vectorizer = TfidfVectorizer(ngram_range=(self.n_gram_size, self.n_gram_size),
                                     token_pattern=r'(?u)\b\w+\b')
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_sim = 0.0

        self.metrics['similarity_calculation_time'] += time.time() - start_time

        return {
            'jaccard': jaccard,
            'containment_1_in_2': containment_1_in_2,
            'containment_2_in_1': containment_2_in_1,
            'max_containment': max(containment_1_in_2, containment_2_in_1),
            'dice_coefficient': dice,
            'cosine_similarity': cosine_sim,
            'intersection_size': len(intersection),
            'union_size': len(union),
            'ngrams1_count': len(ngrams1),
            'ngrams2_count': len(ngrams2)
        }

    def _empty_metrics(self):
        """Return empty metrics for cases where n-grams can't be generated."""
        return {
            'jaccard': 0.0, 'containment_1_in_2': 0.0, 'containment_2_in_1': 0.0,
            'max_containment': 0.0, 'dice_coefficient': 0.0, 'cosine_similarity': 0.0,
            'intersection_size': 0, 'union_size': 0, 'ngrams1_count': 0, 'ngrams2_count': 0
        }

    def find_text_reuse_optimized(self, texts_data, all_metadata):
        """
        Optimized text reuse detection with comprehensive metrics tracking.

        Args:
            texts_data (dict): Dictionary of loaded texts
            all_metadata (dict): Dictionary of metadata

        Returns:
            list: List of reuse instances with detailed metrics
        """
        start_time = time.time()

        # Generate all segments
        all_segments_data = []
        for (notebook, file_path), content_item in texts_data.items():
            segments = self.segment_text_advanced(content_item, notebook, file_path)
            all_segments_data.extend(segments)

        self.metrics['total_segments'] = len(all_segments_data)
        logger.info(f"Total segments to analyze: {len(all_segments_data)}")

        reuse_instances = []
        comparisons_made = 0

        # Use itertools.combinations for efficient pairwise comparison
        for i, j in combinations(range(len(all_segments_data)), 2):
            segment_data1 = all_segments_data[i]
            segment_data2 = all_segments_data[j]

            # Skip same notebook + same page comparisons
            if (segment_data1['notebook'] == segment_data2['notebook'] and
                    segment_data1['page_key'] == segment_data2['page_key']):
                continue

            comparisons_made += 1

            # Calculate similarity metrics
            metrics = self.calculate_similarity_metrics(
                segment_data1['text'], segment_data2['text']
            )

            # Check if similarity meets threshold
            if metrics['jaccard'] >= self.similarity_threshold:
                # Get metadata
                metadata1 = all_metadata.get(segment_data1['notebook'], {}).get(
                    str(segment_data1['page_key']), {}) if segment_data1['page_key'] else {}
                metadata2 = all_metadata.get(segment_data2['notebook'], {}).get(
                    str(segment_data2['page_key']), {}) if segment_data2['page_key'] else {}

                # Create detailed reuse instance
                instance = {
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

                    # All similarity metrics
                    **metrics
                }

                reuse_instances.append(instance)

            # Progress reporting
            if comparisons_made % 1000 == 0:
                logger.info(f"Processed {comparisons_made} comparisons...")

        self.metrics['total_comparisons'] = comparisons_made
        self.metrics['processing_time'] = time.time() - start_time

        logger.info(f"Analysis completed: {comparisons_made} comparisons, "
                    f"{len(reuse_instances)} reuse instances found")
        logger.info(f"Total processing time: {self.metrics['processing_time']:.2f} seconds")

        return reuse_instances

    def calculate_comprehensive_metrics(self, reuse_instances, all_segments_count):
        """
        Calculate comprehensive summary metrics for experimental analysis.

        Args:
            reuse_instances (list): List of detected reuse instances
            all_segments_count (int): Total number of segments analyzed

        Returns:
            dict: Comprehensive metrics dictionary
        """
        if not reuse_instances:
            return self._empty_summary_metrics(all_segments_count)

        # Extract all similarity scores for analysis
        jaccard_scores = [inst['jaccard'] for inst in reuse_instances]
        containment_scores = [inst['max_containment'] for inst in reuse_instances]
        dice_scores = [inst['dice_coefficient'] for inst in reuse_instances]
        cosine_scores = [inst['cosine_similarity'] for inst in reuse_instances]

        # Statistical measures
        metrics = {
            'total_instances': len(reuse_instances),
            'total_segments_analyzed': all_segments_count,
            'total_comparisons_made': self.metrics['total_comparisons'],

            # Jaccard statistics
            'jaccard_mean': np.mean(jaccard_scores),
            'jaccard_std': np.std(jaccard_scores),
            'jaccard_median': np.median(jaccard_scores),
            'jaccard_min': np.min(jaccard_scores),
            'jaccard_max': np.max(jaccard_scores),

            # Containment statistics
            'max_containment_mean': np.mean(containment_scores),
            'max_containment_std': np.std(containment_scores),
            'max_containment_median': np.median(containment_scores),

            # Additional similarity measures
            'dice_coefficient_mean': np.mean(dice_scores),
            'cosine_similarity_mean': np.mean(cosine_scores),

            # Similarity categories
            'high_similarity_count': sum(1 for score in jaccard_scores if score >= 0.8),
            'medium_similarity_count': sum(1 for score in jaccard_scores if 0.5 <= score < 0.8),
            'low_similarity_count': sum(1 for score in jaccard_scores if score < 0.5),

            # Performance metrics
            'processing_time_seconds': self.metrics['processing_time'],
            'ngram_generation_time_seconds': self.metrics['ngram_generation_time'],
            'similarity_calculation_time_seconds': self.metrics['similarity_calculation_time'],
            'comparisons_per_second': (self.metrics['total_comparisons'] /
                                    self.metrics['processing_time'] if self.metrics['processing_time'] > 0 else 0),

            # Reuse rates
            'reuse_rate': (len(reuse_instances) /
                        (all_segments_count * (all_segments_count - 1) / 2)
                        if all_segments_count > 1 else 0.0),
            'segments_with_reuse': len(set([inst['segment1_page_key'] for inst in reuse_instances] +
                                        [inst['segment2_page_key'] for inst in reuse_instances])),

            # Configuration used
            'n_gram_size': self.n_gram_size,
            'similarity_threshold': self.similarity_threshold,
            'stemming_used': self.use_stemming,
            'stopwords_removed': self.remove_stopwords
        }

        return metrics

    def _empty_summary_metrics(self, all_segments_count):
        """Return empty summary metrics."""
        return {
            'total_instances': 0,
            'total_segments_analyzed': all_segments_count,
            'total_comparisons_made': self.metrics['total_comparisons'],
            'jaccard_mean': 0.0,
            'max_containment_mean': 0.0,
            'processing_time_seconds': self.metrics['processing_time'],
            'reuse_rate': 0.0,
            'n_gram_size': self.n_gram_size,
            'similarity_threshold': self.similarity_threshold
        }
        def save_results_for_experiment(self, reuse_instances, summary_metrics,
                                        results_dir, filename_base, config_id):
            """
            Save results in formats suitable for experimental comparison.
            Creates JSON, CSV, and detailed TXT files.

            Args:
                reuse_instances (list): List of reuse instances
                summary_metrics (dict): Summary metrics
                results_dir (str): Directory to save results
                filename_base (str): Base filename for output files
            """
            results_path = Path(results_dir)
            results_path.mkdir(parents=True, exist_ok=True)

            config_folder = results_path / f"config_{config_id}"
            config_folder.mkdir(exist_ok=True)

            results_file = config_folder / f"{filename_base}_ngram_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'method': 'ngram_library_based',
                    'configuration': {
                        'n_gram_size': self.n_gram_size,
                        'similarity_threshold': self.similarity_threshold,
                        'stemming_used': self.use_stemming,
                        'stopwords_removed': self.remove_stopwords
                    },
                    'reuse_instances': reuse_instances,
                    'summary_metrics': summary_metrics,
                    'timestamp': datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)

            if reuse_instances:
                instances_df = pd.DataFrame(reuse_instances)
                instances_file = config_folder / f"{filename_base}_ngram_instances.csv"
                instances_df.to_csv(instances_file, index=False)

            self._save_detailed_txt_report(reuse_instances, summary_metrics, config_folder, filename_base)

            logger.info(f"N-gram results saved to {config_folder}")
            return str(results_file)

        def _save_detailed_txt_report(self, reuse_instances, summary_metrics, results_dir, filename_base):
            """
            Save a detailed, human-readable TXT report of text reuse instances.
            """
            txt_file = os.path.join(results_dir, f"{filename_base}_detailed_report.txt")

            with open(txt_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("=" * 80 + "\n")
                f.write(f"TEXT REUSE DETECTION RESULTS - {filename_base.upper()}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Method: N-gram Library-Based Detection\n")
                f.write(f"N-gram Size: {self.n_gram_size}\n")
                f.write(f"Similarity Threshold: {self.similarity_threshold}\n")
                f.write(f"Stemming Used: {self.use_stemming}\n")
                f.write(f"Stopwords Removed: {self.remove_stopwords}\n")
                f.write("=" * 80 + "\n\n")

                # Summary Section
                f.write("SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total text reuse instances found: {len(reuse_instances)}\n")
                f.write(f"Processing time: {summary_metrics['processing_time_seconds']:.2f} seconds\n")
                f.write(f"Segments analyzed: {summary_metrics['total_segments_analyzed']}\n")
                f.write(f"Comparisons made: {summary_metrics['total_comparisons_made']}\n")

                if len(reuse_instances) > 0:
                    f.write(f"Average Jaccard similarity: {summary_metrics['jaccard_mean']:.3f}\n")
                    f.write(f"Average max containment: {summary_metrics['max_containment_mean']:.3f}\n")
                    f.write(
                        f"Similarity range: {summary_metrics['jaccard_min']:.3f} - {summary_metrics['jaccard_max']:.3f}\n")
                    f.write(f"High similarity instances (≥0.8): {summary_metrics['high_similarity_count']}\n")
                    f.write(f"Medium similarity instances (0.5-0.8): {summary_metrics['medium_similarity_count']}\n")
                    f.write(f"Low similarity instances (<0.5): {summary_metrics['low_similarity_count']}\n")

                f.write(f"Reuse rate: {summary_metrics['reuse_rate']:.4f}\n")
                f.write("\n" + "=" * 80 + "\n\n")

                if not reuse_instances:
                    f.write("No text reuse instances found above the similarity threshold.\n")
                    return

                # Sort instances by similarity for better readability
                sorted_instances = sorted(reuse_instances, key=lambda x: x['jaccard'], reverse=True)

                # Detailed instances section
                f.write("DETAILED TEXT REUSE INSTANCES\n")
                f.write("=" * 80 + "\n")
                f.write(f"Showing all {len(sorted_instances)} instances (sorted by Jaccard similarity):\n\n")

                for i, instance in enumerate(sorted_instances, 1):
                    f.write(f"Instance {i}:\n")
                    f.write(f"  Notebooks: {instance['notebook1']} ↔ {instance['notebook2']}\n")
                    f.write(f"  Files: {instance['file1']} ↔ {instance['file2']}\n")
                    f.write(
                        f"  Segments (on page): {instance['segment1_index_on_page']} ↔ {instance['segment2_index_on_page']}\n")

                    if instance.get('segment1_page_key') and instance.get('segment2_page_key'):
                        f.write(f"  Pages: {instance['segment1_page_key']} ↔ {instance['segment2_page_key']}\n")

                    f.write(f"  Jaccard Similarity: {instance['jaccard']:.3f}\n")
                    f.write(f"  Max Containment: {instance['max_containment']:.3f}\n")
                    f.write(
                        f"  Containment ({instance['notebook1']} in {instance['notebook2']}): {instance['containment_1_in_2']:.3f}\n")
                    f.write(
                        f"  Containment ({instance['notebook2']} in {instance['notebook1']}): {instance['containment_2_in_1']:.3f}\n")
                    f.write(f"  Dice Coefficient: {instance['dice_coefficient']:.3f}\n")
                    f.write(f"  Cosine Similarity: {instance['cosine_similarity']:.3f}\n")
                    f.write(f"  Intersection Size: {instance['intersection_size']} n-grams\n")
                    f.write(f"  Union Size: {instance['union_size']} n-grams\n")
                    f.write(
                        f"  Text 1 ({instance['notebook1']}): {instance['segment1_text'][:150]}{'...' if len(instance['segment1_text']) > 150 else ''}\n")
                    f.write(
                        f"  Text 2 ({instance['notebook2']}): {instance['segment2_text'][:150]}{'...' if len(instance['segment2_text']) > 150 else ''}\n")

                    # Add metadata if available
                    if instance.get('segment1_metadata'):
                        f.write(f"  Metadata 1: {json.dumps(instance['segment1_metadata'], indent=4)}\n")
                    if instance.get('segment2_metadata'):
                        f.write(f"  Metadata 2: {json.dumps(instance['segment2_metadata'], indent=4)}\n")

                    f.write("-" * 80 + "\n")

                    # Add spacing every 5 instances for better readability
                    if i % 5 == 0 and i < len(sorted_instances):
                        f.write("\n")

                f.write("\nEND OF DETAILED REPORT\n")
                f.write("=" * 80 + "\n")


# Example usage function
def main():
    """
    Main function for Sir David Humphry's 18th-century text analysis.
    Configured specifically for 2-grams, 3-grams, and 4-grams comparison.
    """
    parser = argparse.ArgumentParser(description="N-gram text reuse analysis")
    parser.add_argument('--notebooks', type=str, default='*',
                        help="Comma-separated notebook IDs (e.g., 14e,14g) or * for all")
    parser.add_argument('--combo-size', type=str, default='2',
                        help="2,3,4 to run over combinations of that size, or 'all' to use all selected notebooks")
    parser.add_argument('--filenames', type=str, default='page_to_text.json',
                        help="Comma-separated filenames to process (default: page_to_text.json)")
    parser.add_argument('--config-id', type=int, default=None,
                        help="Specific config ID to run (optional - runs all configs if not specified)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    base_dir = project_root / "preprocessing"
    selected_notebooks = parse_notebooks_arg(args.notebooks, base_dir)
    if not selected_notebooks:
        logger.error("No notebooks selected. Check --notebooks argument or preprocessing directory.")
        return
    filenames = [f.strip() for f in args.filenames.split(',') if f.strip()]
    results_dir = project_root / "results_text_reuse" / "results_ngram"

    # Focused set of N-gram configurations (Experiment IDs 2 and 5)
    all_configs = [
        (
            2,
        {
            'n_gram_size': 2,
            'similarity_threshold': 0.25,
                'use_stemming': True,
                'remove_stopwords': True,
            'min_segment_length': 25,
            'min_words': 4
            }
        ),
        (
            5,
        {
            'n_gram_size': 4,
                'similarity_threshold': 0.15,
            'use_stemming': False,
            'remove_stopwords': False,
                'min_segment_length': 40,
            'min_words': 6
            }
        )
    ]

    # Filter configs if specific config_id is requested
    if args.config_id:
        configs = [c for c in all_configs if c[0] == args.config_id]
        if not configs:
            logger.error(f"Config ID {args.config_id} not found. Available configs: {[c[0] for c in all_configs]}")
            return
    else:
        configs = all_configs

    all_experiment_results = []

    print("Starting N-gram Analysis for 18th-Century Historical Text (Sir David Humphry)")
    if args.config_id:
        print(f"Running configuration {args.config_id}")
    else:
        print("Running configurations 2 and 5")
    print("=" * 80)
    # Determine notebook groups to run
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
        canonical_notebooks = sorted(group_list)
        group_tag = "__nb_" + '-'.join(canonical_notebooks)

        for config_id, config in configs:
            config_name = f"Config {config_id}: {config['n_gram_size']}-gram"
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

        detector = LibraryBasedNgramDetector(**config)

        # Load texts
        texts_data, all_metadata = detector.load_texts(base_dir, group_list, filenames)

        # Process each filename
        for filename in filenames:
            current_texts = {(nb, fp): content for (nb, fp), content in texts_data.items()
                             if os.path.basename(fp) == filename}

            if not current_texts:
                logger.warning(f"No texts found for {filename}")
                continue

            # If notebooks don't match requested pair, skip
            current_notebooks = sorted({nb for (nb, _) in current_texts.keys()})
            if canonical_notebooks != current_notebooks:
                logger.info(
                    f"Skipping results save for filename {filename} — notebooks in data {current_notebooks}"
                    f" do not match requested pair {canonical_notebooks}."
                )
                continue

            # Detect reuse
            reuse_instances = detector.find_text_reuse_optimized(current_texts, all_metadata)

            # Calculate metrics
            total_segments = detector.metrics['total_segments']
            summary_metrics = detector.calculate_comprehensive_metrics(reuse_instances, total_segments)

            # Save results with descriptive naming
            ngram_type = f"{config['n_gram_size']}gram"
            preprocessing = "stemmed" if config['use_stemming'] else "unstemmed"
            stopwords = "no_stopwords" if config['remove_stopwords'] else "with_stopwords"
            base_name = f"{os.path.splitext(filename)[0]}_{ngram_type}_{preprocessing}_{stopwords}{group_tag}"

            detector.save_results_for_experiment(reuse_instances, summary_metrics,
                                                 results_dir, base_name, config_id)

            # Store for comparison
            experiment_result = {
                'config_id': config_id,
                'config_name': config_name,
                'n_gram_size': config['n_gram_size'],
                'configuration': config,
                'filename': filename,
                'summary_metrics': summary_metrics,
                'instance_count': len(reuse_instances),
                'notebooks': group_list
            }
            all_experiment_results.append(experiment_result)

            # Print summary
            print(f"\n{config_name} Results for {filename} [notebooks: {group_list}]:")
            print(f"  Text reuse instances found: {len(reuse_instances)}")
            print(f"  Processing time: {summary_metrics['processing_time_seconds']:.2f}s")
            print(f"  Mean Jaccard similarity: {summary_metrics['jaccard_mean']:.3f}")
            print(f"  Segments analyzed: {summary_metrics['total_segments_analyzed']}")
            print(f"  Reuse rate: {summary_metrics['reuse_rate']:.4f}")
            if len(reuse_instances) > 0:
                print(
                    f"  Similarity range: {summary_metrics['jaccard_min']:.3f} - {summary_metrics['jaccard_max']:.3f}")

    print(f"\n{'=' * 80}")
    print("N-GRAM EXPERIMENT COMPLETED (Configurations 2 & 5 Only)")
    print(f"{'=' * 80}")
    for result in all_experiment_results:
        metrics = result['summary_metrics']
        print(f"Config {result['config_id']} - {result['config_name']}")
        print(f"  Instances found: {result['instance_count']}")
        print(f"  Mean Jaccard similarity: {metrics['jaccard_mean']:.3f}")
        print(f"  Processing time: {metrics['processing_time_seconds']:.2f}s")
        print(f"  Comparisons made: {metrics['total_comparisons_made']}")


if __name__ == "__main__":
    main()