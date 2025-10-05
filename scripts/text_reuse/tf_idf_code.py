import os
import argparse
import json
import logging
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import defaultdict
import re

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


class LibraryBasedTFIDFDetector:
    """TF-IDF-based text reuse detector for 18th-century historical texts."""

    def __init__(self, similarity_threshold=0.3, ngram_range=(1, 3), max_features=10000,
                 use_stemming=False, remove_stopwords=True, min_segment_length=20,
                 min_words=3, similarity_metric='cosine'):
        """Initialize the TF-IDF detector with configurable parameters.

        Args:
            similarity_threshold (float): Minimum TF-IDF similarity for reuse
            ngram_range (tuple): Range of n-grams to extract (min_n, max_n)
            max_features (int): Maximum number of features for TF-IDF vectorizer
            use_stemming (bool): Whether to apply stemming
            remove_stopwords (bool): Whether to remove stopwords
            min_segment_length (int): Minimum character length for segments
            min_words (int): Minimum word count for segments
            similarity_metric (str): 'cosine', 'euclidean', or 'manhattan'
        """
        self.similarity_threshold = similarity_threshold
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.min_segment_length = min_segment_length
        self.min_words = min_words
        self.similarity_metric = similarity_metric

        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else None
        self.vectorizer = None
        self.feature_names = None

        self.metrics = {
            'processing_time': 0,
            'total_comparisons': 0,
            'total_segments': 0,
            'vectorization_time': 0,
            'similarity_calculation_time': 0,
            'vocabulary_size': 0
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
            notebook_dir = os.path.join(base_dir, notebook)
            if not os.path.exists(notebook_dir):
                logger.warning(f"Directory {notebook_dir} not found, skipping.")
                continue

            metadata_file = os.path.join(notebook_dir, 'page_to_entities.json')
            all_metadata[notebook] = self._load_metadata(metadata_file)

            for filename in filenames:
                file_path = os.path.join(notebook_dir, filename)
                if os.path.exists(file_path):
                    content = self._load_file_content(file_path)
                    if content:
                        texts[(notebook, file_path)] = content

        logger.info(f"Text loading completed in {time.time() - start_time:.2f} seconds")
        return texts, all_metadata

    def _load_metadata(self, metadata_file):
        """Load metadata from JSON file."""
        if os.path.exists(metadata_file):
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
                if content.strip() and file_path.endswith(".json"):
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
            str: Cleaned text for TF-IDF processing
        """
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\|&:|_|"|xxx', '', text)
        text = re.sub(r'\bf\b', 's', text)
        text = re.sub(r'\bye\b', 'the', text)
        text = re.sub(r'\bvs\b', 'us', text)
        text = text.lower()
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', '', text)
        text = re.sub(r'\.{2,}', '.', text)
        return re.sub(r'\s+', ' ', text).strip()

    def create_custom_tokenizer(self):
        """Create a custom tokenizer for the TF-IDF vectorizer."""

        def tokenize_and_preprocess(text):
            words = word_tokenize(text)
            if self.use_stemming and self.stemmer:
                words = [self.stemmer.stem(word) for word in words]
            return [word for word in words if word.isalpha() and len(word) > 1]

        return tokenize_and_preprocess

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
        preprocessed_text = self.preprocess_text_advanced(raw_segment)
        word_count = len(preprocessed_text.split())
        if len(preprocessed_text) >= self.min_segment_length and word_count >= self.min_words:
            segments_data.append({
                'preprocessed_text': preprocessed_text,
                'original_text': raw_segment.strip(),
                'notebook': notebook_id,
                'file_path': file_path,
                'page_key': page_key,
                'segment_index_on_page': segment_index,
                'word_count': word_count
            })

    def create_tfidf_vectors(self, segments_data):
        """Create TF-IDF vectors for all segments.

        Args:
            segments_data (list): List of segment dictionaries

        Returns:
            tuple: (TF-IDF matrix, segments with vectors)
        """
        start_time = time.time()
        texts = [segment['preprocessed_text'] for segment in segments_data]

        vectorizer_params = {
            'ngram_range': self.ngram_range,
            'max_features': self.max_features,
            'lowercase': True,
            'token_pattern': r'(?u)\b\w\w+\b',
            'sublinear_tf': True,
            'norm': 'l2',
            'use_idf': True,
            'smooth_idf': True
        }

        if self.remove_stopwords:
            vectorizer_params['stop_words'] = 'english'
        if self.use_stemming:
            vectorizer_params['tokenizer'] = self.create_custom_tokenizer()

        self.vectorizer = TfidfVectorizer(**vectorizer_params)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.metrics['vocabulary_size'] = len(self.feature_names)
            logger.info(f"TF-IDF vectorization completed: {tfidf_matrix.shape[0]} documents, "
                        f"{tfidf_matrix.shape[1]} features")
        except Exception as e:
            logger.error(f"Error during TF-IDF vectorization: {e}")
            return None, None

        self.metrics['vectorization_time'] = time.time() - start_time
        return tfidf_matrix, segments_data

    def calculate_tfidf_similarity_metrics(self, vector1, vector2, segment1_data, segment2_data):
        """Calculate comprehensive TF-IDF-based similarity metrics.

        Args:
            vector1, vector2: TF-IDF vectors
            segment1_data, segment2_data: Segment metadata

        Returns:
            dict: Comprehensive similarity metrics
        """
        start_time = time.time()
        vector1 = vector1.reshape(1, -1) if vector1.ndim == 1 else vector1
        vector2 = vector2.reshape(1, -1) if vector2.ndim == 1 else vector2

        cosine_sim = cosine_similarity(vector1, vector2)[0, 0]
        euclidean_dist = euclidean_distances(vector1, vector2)[0, 0]
        euclidean_sim = 1 / (1 + euclidean_dist)
        manhattan_dist = np.sum(np.abs(vector1 - vector2))
        manhattan_sim = 1 / (1 + manhattan_dist)

        vector1_nonzero = set(np.nonzero(vector1)[1])
        vector2_nonzero = set(np.nonzero(vector2)[1])
        jaccard_sim = (len(vector1_nonzero.intersection(vector2_nonzero)) /
                       len(vector1_nonzero.union(vector2_nonzero))
                       if vector1_nonzero or vector2_nonzero else 0.0)
        overlap_coef = (len(vector1_nonzero.intersection(vector2_nonzero)) /
                        min(len(vector1_nonzero), len(vector2_nonzero))
                        if vector1_nonzero and vector2_nonzero else 0.0)

        combined_vector = vector1 + vector2
        top_feature_indices = np.argsort(combined_vector.toarray().flatten())[-10:][::-1]
        top_features = [self.feature_names[i] for i in top_feature_indices if combined_vector[0, i] > 0]

        primary_similarity = (cosine_sim if self.similarity_metric == 'cosine' else
                              euclidean_sim if self.similarity_metric == 'euclidean' else
                              manhattan_sim)

        self.metrics['similarity_calculation_time'] += time.time() - start_time

        return {
            'tfidf_similarity': primary_similarity,
            'cosine_similarity': cosine_sim,
            'euclidean_similarity': euclidean_sim,
            'manhattan_similarity': manhattan_sim,
            'jaccard_features': jaccard_sim,
            'overlap_coefficient': overlap_coef,
            'shared_features': len(vector1_nonzero.intersection(vector2_nonzero)),
            'total_features_1': len(vector1_nonzero),
            'total_features_2': len(vector2_nonzero),
            'top_shared_features': top_features[:5],
            'vector1_norm': np.linalg.norm(vector1.toarray()),
            'vector2_norm': np.linalg.norm(vector2.toarray())
        }

    def find_text_reuse_optimized(self, texts_data, all_metadata):
        """Optimized TF-IDF-based text reuse detection.

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

        tfidf_matrix, segments_with_vectors = self.create_tfidf_vectors(all_segments_data)
        if tfidf_matrix is None:
            logger.error("TF-IDF vectorization failed")
            return []

        reuse_instances = []
        comparisons_made = 0

        for i, j in combinations(range(len(segments_with_vectors)), 2):
            segment_data1 = segments_with_vectors[i]
            segment_data2 = segments_with_vectors[j]

            if (segment_data1['notebook'] == segment_data2['notebook'] and
                    segment_data1['page_key'] == segment_data2['page_key']):
                continue

            comparisons_made += 1
            metrics = self.calculate_tfidf_similarity_metrics(
                tfidf_matrix[i], tfidf_matrix[j], segment_data1, segment_data2)

            if metrics['tfidf_similarity'] >= self.similarity_threshold:
                metadata1 = all_metadata.get(segment_data1['notebook'], {}).get(str(segment_data1['page_key']), {})
                metadata2 = all_metadata.get(segment_data2['notebook'], {}).get(str(segment_data2['page_key']), {})

                reuse_instances.append({
                    'notebook1': segment_data1['notebook'],
                    'file1': os.path.basename(segment_data1['file_path']),
                    'segment1_index_on_page': segment_data1['segment_index_on_page'],
                    'segment1_page_key': segment_data1['page_key'],
                    'segment1_text': segment_data1['original_text'],
                    'segment1_metadata': metadata1,
                    'segment1_word_count': segment_data1['word_count'],
                    'notebook2': segment_data2['notebook'],
                    'file2': os.path.basename(segment_data2['file_path']),
                    'segment2_index_on_page': segment_data2['segment_index_on_page'],
                    'segment2_page_key': segment_data2['page_key'],
                    'segment2_text': segment_data2['original_text'],
                    'segment2_metadata': metadata2,
                    'segment2_word_count': segment_data2['word_count'],
                    **metrics
                })

            if comparisons_made % 1000 == 0:
                logger.info(f"Processed {comparisons_made} comparisons...")

        self.metrics['total_comparisons'] = comparisons_made
        self.metrics['processing_time'] = time.time() - start_time

        logger.info(f"TF-IDF analysis completed: {comparisons_made} comparisons, "
                    f"{len(reuse_instances)} reuse instances found")
        logger.info(f"Total processing time: {self.metrics['processing_time']:.2f} seconds")
        logger.info(f"Vocabulary size: {self.metrics['vocabulary_size']} features")
        return reuse_instances

    def calculate_comprehensive_metrics(self, reuse_instances, all_segments_count):
        """Calculate comprehensive summary metrics.

        Args:
            reuse_instances (list): List of reuse instances
            all_segments_count (int): Total number of segments analyzed

        Returns:
            dict: Comprehensive metrics
        """
        if not reuse_instances:
            return self._empty_summary_metrics(all_segments_count)

        tfidf_similarity_scores = [inst['tfidf_similarity'] for inst in reuse_instances]
        cosine_scores = [inst['cosine_similarity'] for inst in reuse_instances]
        euclidean_scores = [inst['euclidean_similarity'] for inst in reuse_instances]
        jaccard_scores = [inst['jaccard_features'] for inst in reuse_instances]
        shared_features = [inst['shared_features'] for inst in reuse_instances]

        # Convert NumPy types to native Python types for JSON serialization
        return {
            'total_instances': len(reuse_instances),
            'total_segments_analyzed': all_segments_count,
            'total_comparisons_made': self.metrics['total_comparisons'],
            'tfidf_similarity_mean': float(np.mean(tfidf_similarity_scores)),
            'tfidf_similarity_std': float(np.std(tfidf_similarity_scores)),
            'tfidf_similarity_median': float(np.median(tfidf_similarity_scores)),
            'tfidf_similarity_min': float(np.min(tfidf_similarity_scores)),
            'tfidf_similarity_max': float(np.max(tfidf_similarity_scores)),
            'cosine_similarity_mean': float(np.mean(cosine_scores)),
            'cosine_similarity_std': float(np.std(cosine_scores)),
            'euclidean_similarity_mean': float(np.mean(euclidean_scores)),
            'euclidean_similarity_std': float(np.std(euclidean_scores)),
            'jaccard_features_mean': float(np.mean(jaccard_scores)),
            'jaccard_features_std': float(np.std(jaccard_scores)),
            'avg_shared_features': float(np.mean(shared_features)),
            'max_shared_features': int(np.max(shared_features)),
            'min_shared_features': int(np.min(shared_features)),
            'high_similarity_count': int(sum(1 for score in tfidf_similarity_scores if score >= 0.8)),
            'medium_similarity_count': int(sum(1 for score in tfidf_similarity_scores if 0.5 <= score < 0.8)),
            'low_similarity_count': int(sum(1 for score in tfidf_similarity_scores if score < 0.5)),
            'processing_time_seconds': float(self.metrics['processing_time']),
            'vectorization_time_seconds': float(self.metrics['vectorization_time']),
            'similarity_calculation_time_seconds': float(self.metrics['similarity_calculation_time']),
            'comparisons_per_second': (float(self.metrics['total_comparisons'] / self.metrics['processing_time'])
                                       if self.metrics['processing_time'] > 0 else 0.0),
            'vocabulary_size': int(self.metrics['vocabulary_size']),
            'ngram_range_min': int(self.ngram_range[0]),
            'ngram_range_max': int(self.ngram_range[1]),
            'max_features_limit': int(self.max_features),
            'reuse_rate': (float(len(reuse_instances) / (all_segments_count * (all_segments_count - 1) / 2))
                           if all_segments_count > 1 else 0.0),
            'similarity_metric': self.similarity_metric,
            'similarity_threshold': float(self.similarity_threshold),
            'stemming_used': self.use_stemming,
            'stopwords_removed': self.remove_stopwords
        }

    def _empty_summary_metrics(self, all_segments_count):
        """Return empty summary metrics."""
        return {
            'total_instances': 0,
            'total_segments_analyzed': all_segments_count,
            'total_comparisons_made': self.metrics['total_comparisons'],
            'tfidf_similarity_mean': 0.0,
            'tfidf_similarity_std': 0.0,
            'tfidf_similarity_median': 0.0,
            'tfidf_similarity_min': 0.0,
            'tfidf_similarity_max': 0.0,
            'cosine_similarity_mean': 0.0,
            'cosine_similarity_std': 0.0,
            'euclidean_similarity_mean': 0.0,
            'euclidean_similarity_std': 0.0,
            'jaccard_features_mean': 0.0,
            'jaccard_features_std': 0.0,
            'avg_shared_features': 0.0,
            'max_shared_features': 0,
            'min_shared_features': 0,
            'high_similarity_count': 0,
            'medium_similarity_count': 0,
            'low_similarity_count': 0,
            'processing_time_seconds': float(self.metrics.get('processing_time', 0.0)),
            'vectorization_time_seconds': float(self.metrics.get('vectorization_time', 0.0)),
            'similarity_calculation_time_seconds': float(self.metrics.get('similarity_calculation_time', 0.0)),
            'comparisons_per_second': 0.0,
            'vocabulary_size': int(self.metrics.get('vocabulary_size', 0)),
            'ngram_range_min': int(self.ngram_range[0]),
            'ngram_range_max': int(self.ngram_range[1]),
            'max_features_limit': int(self.max_features),
            'reuse_rate': 0.0,
            'similarity_metric': self.similarity_metric,
            'similarity_threshold': float(self.similarity_threshold),
            'stemming_used': self.use_stemming,
            'stopwords_removed': self.remove_stopwords
        }

    def save_results_for_experiment(self, reuse_instances, summary_metrics,
                                    results_dir, filename_base, config_id):
        """Save experiment results in multiple formats."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Add config_id to filename
        if config_id is not None:
            json_filename = f"{filename_base}_tfidf_results.json"
        else:
            json_filename = f"{filename_base}_tfidf_results.json"

        json_path = results_dir / json_filename

        output_data = {
            'metadata': {
                'analysis_type': 'tfidf_text_reuse',
                'timestamp': datetime.now().isoformat(),
                'total_instances': len(reuse_instances),
                'config_id': config_id
            },
            'summary_metrics': summary_metrics,
            'reuse_instances': reuse_instances
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {json_path}")

        # Save detailed text report
        self._save_detailed_txt_report(reuse_instances, summary_metrics, results_dir, filename_base)
        # Save metrics summary
        self._save_metrics_txt_report(summary_metrics, results_dir, filename_base)

    def _save_detailed_txt_report(self, reuse_instances, summary_metrics, results_dir, filename_base):
        """Save a detailed text report of findings."""
        txt_path = results_dir / f"{filename_base}_detailed_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TF-IDF TEXT REUSE ANALYSIS - DETAILED REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Instances Found: {summary_metrics.get('total_instances', 0)}\n")
            f.write(f"Total Segments Analyzed: {summary_metrics.get('total_segments_analyzed', 0)}\n")
            f.write(f"Total Comparisons: {summary_metrics.get('total_comparisons_made', 0)}\n")
            f.write(f"Processing Time: {summary_metrics.get('processing_time_seconds', 0):.2f}s\n")
            f.write(f"Vocabulary Size: {summary_metrics.get('vocabulary_size', 0)}\n\n")

            if reuse_instances:
                f.write("INSTANCE DETAILS\n")
                f.write("-" * 80 + "\n")
                for i, inst in enumerate(reuse_instances[:50], 1):
                    f.write(f"\nInstance {i}:\n")
                    f.write(f"  Similarity: {inst.get('tfidf_similarity', 0):.4f}\n")
                    f.write(f"  Segments: {inst.get('segment1_page_key')} <-> {inst.get('segment2_page_key')}\n")
                    f.write(f"  Shared Features: {inst.get('shared_features', 0)}\n")
                if len(reuse_instances) > 50:
                    f.write(f"\n... and {len(reuse_instances) - 50} more instances\n")

        logger.info(f"Detailed report saved to {txt_path}")

    def _save_metrics_txt_report(self, summary_metrics, results_dir, filename_base):
        """Save metrics summary as plain text."""
        metrics_path = results_dir / f"{filename_base}_metrics_summary.txt"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("TF-IDF TEXT REUSE METRICS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            for key, value in summary_metrics.items():
                f.write(f"{key}: {value}\n")
        logger.info(f"Metrics summary saved to {metrics_path}")


def get_available_notebooks(base_dir: Path):
    try:
        return sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    except Exception:
        return []


def parse_notebooks_arg(notebooks_arg: str, base_dir: Path):
    if notebooks_arg.strip() == '*':
        return get_available_notebooks(base_dir)
    return [nb.strip() for nb in notebooks_arg.split(',') if nb.strip()]


def main():
    """Main function for TF-IDF-based text reuse analysis."""
    parser = argparse.ArgumentParser(description="TF-IDF text reuse analysis")
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
        logger.error("No notebooks found or specified")
        return

    filenames = [f.strip() for f in args.filenames.split(',') if f.strip()]
    combo_size_arg = args.combo_size.strip().lower()
    if combo_size_arg == 'all':
        combo_sizes = [len(selected_notebooks)]
    else:
        try:
            combo_sizes = [int(x.strip()) for x in combo_size_arg.split(',') if x.strip()]
        except ValueError:
            logger.error("Invalid combo-size format")
            return

    results_base_dir = project_root / "results_text_reuse" / "results_tfidf"

    configs = [
        (1, {'similarity_threshold': 0.3, 'ngram_range': (1, 1), 'max_features': 5000, 'use_stemming': False,
         'remove_stopwords': False, 'similarity_metric': 'cosine', 'min_segment_length': 30, 'min_words': 5}),
        (2, {'similarity_threshold': 0.25, 'ngram_range': (1, 1), 'max_features': 5000, 'use_stemming': True,
         'remove_stopwords': True, 'similarity_metric': 'cosine', 'min_segment_length': 25, 'min_words': 4}),
        (3, {'similarity_threshold': 0.3, 'ngram_range': (1, 2), 'max_features': 8000, 'use_stemming': False,
         'remove_stopwords': False, 'similarity_metric': 'cosine', 'min_segment_length': 35, 'min_words': 5}),
        (4, {'similarity_threshold': 0.25, 'ngram_range': (1, 2), 'max_features': 8000, 'use_stemming': True,
         'remove_stopwords': True, 'similarity_metric': 'cosine', 'min_segment_length': 30, 'min_words': 4}),
        (5, {'similarity_threshold': 0.35, 'ngram_range': (1, 3), 'max_features': 10000, 'use_stemming': False,
         'remove_stopwords': False, 'similarity_metric': 'cosine', 'min_segment_length': 40, 'min_words': 6}),
        (6, {'similarity_threshold': 0.2, 'ngram_range': (1, 3), 'max_features': 10000, 'use_stemming': True,
         'remove_stopwords': True, 'similarity_metric': 'cosine', 'min_segment_length': 35, 'min_words': 5}),
        (7, {'similarity_threshold': 0.4, 'ngram_range': (1, 2), 'max_features': 8000, 'use_stemming': False,
         'remove_stopwords': True, 'similarity_metric': 'euclidean', 'min_segment_length': 30, 'min_words': 5}),
        (8, {'similarity_threshold': 0.35, 'ngram_range': (1, 2), 'max_features': 8000, 'use_stemming': True,
         'remove_stopwords': False, 'similarity_metric': 'manhattan', 'min_segment_length': 35, 'min_words': 5})
    ]

    # Filter configs if config_id specified
    if args.config_id is not None:
        configs = [(cid, cfg) for cid, cfg in configs if cid == args.config_id]
        if not configs:
            logger.error(f"Config ID {args.config_id} not found")
            return

    print("Starting TF-IDF Analysis for 18th-Century Historical Text (Sir David Humphry)")
    print("=" * 80)

    for combo_size in combo_sizes:
        if combo_size > len(selected_notebooks):
            logger.warning(f"Combo size {combo_size} > available notebooks {len(selected_notebooks)}, skipping")
            continue

        notebook_groups = list(combinations(selected_notebooks, combo_size))
        logger.info(f"Running {len(notebook_groups)} notebook combination(s) of size {combo_size}")

        for group in notebook_groups:
            group_list = list(group)
            group_tag = "__nb_" + '-'.join(group_list)

            for config_id, config in configs:
                ngram_desc = f"{config['ngram_range'][0]}to{config['ngram_range'][1]}gram"
                preprocessing = "stemmed" if config['use_stemming'] else "unstemmed"
                stopwords_str = "no_stopwords" if config['remove_stopwords'] else "with_stopwords"
                config_name = f"Config {config_id}: TF-IDF {ngram_desc} {config['similarity_metric']}"
                if config['use_stemming']:
                    config_name += " + stemming"
                if config['remove_stopwords']:
                    config_name += " + stopword removal"

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Running {config_name} on notebooks: {group_list}")
                logger.info(f"Configuration: {config}")
                logger.info(f"{'=' * 80}")

                detector = LibraryBasedTFIDFDetector(**config)
                texts_data, all_metadata = detector.load_texts(str(base_dir), group_list, filenames)

                if not texts_data:
                    logger.error("No texts loaded")
                    continue

                for filename in filenames:
                    current_texts = {(nb, fp): content for (nb, fp), content in texts_data.items()
                                     if os.path.basename(fp) == filename}
                    if not current_texts:
                        continue

                    reuse_instances = detector.find_text_reuse_optimized(current_texts, all_metadata)
                    total_segments = detector.metrics['total_segments']
                    summary_metrics = detector.calculate_comprehensive_metrics(reuse_instances, total_segments)

                    results_dir = results_base_dir / f"config_{config_id}"
                    base_name = f"{os.path.splitext(filename)[0]}_tfidf_{ngram_desc}_{config['similarity_metric']}_{preprocessing}_{stopwords_str}{group_tag}"

                    detector.save_results_for_experiment(reuse_instances, summary_metrics, results_dir, base_name, config_id)

                    print(f"\n{config_name} Results for {filename}:")
                    print(f" Instances found: {len(reuse_instances)}")
                    print(f" Processing time: {summary_metrics['processing_time_seconds']:.2f}s")
                    if len(reuse_instances) > 0:
                        print(f" Mean similarity: {summary_metrics['tfidf_similarity_mean']:.3f}")

    print(f"\n{'=' * 80}")
    print("TF-IDF ANALYSIS COMPLETED!")
    print(f"Results saved to: {results_base_dir}")


if __name__ == "__main__":
    main()
