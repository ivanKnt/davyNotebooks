import os
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
        """Load texts and metadata with error handling."""
        start_time = time.time()
        texts = {}
        all_metadata = {}

        for notebook in notebooks:
            notebook_dir = base_dir / notebook
            if not notebook_dir.exists():
                logger.warning(f"Directory {notebook_dir} not found, skipping.")
                continue

            metadata_file = notebook_dir / 'page_to_entities.json'
            all_metadata[notebook] = {}
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        all_metadata[notebook] = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading metadata from {metadata_file}: {e}")

            for filename in filenames:
                file_path = notebook_dir / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if content.strip():
                            if filename.endswith('.json'):
                                data = json.loads(content)
                                texts[(notebook, str(file_path))] = {k: str(v) for k, v in data.items() if str(v).strip()}
                            else:
                                logger.info(f"Skipping non-JSON file: {file_path}")
                        else:
                            logger.warning(f"Empty file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error reading {file_path}: {e}")

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
            'processing_time_seconds': float(self.metrics['processing_time']),
            'vectorization_time_seconds': float(self.metrics['vectorization_time']),
            'similarity_calculation_time_seconds': float(self.metrics['similarity_calculation_time']),
            'comparisons_per_second': 0.0,
            'vocabulary_size': int(self.metrics['vocabulary_size']),
            'ngram_range_min': int(self.ngram_range[0]),
            'ngram_range_max': int(self.ngram_range[1]),
            'max_features_limit': int(self.max_features),
            'reuse_rate': 0.0,
            'similarity_metric': self.similarity_metric,
            'similarity_threshold': float(self.similarity_threshold),
            'stemming_used': self.use_stemming,
            'stopwords_removed': self.remove_stopwords
        }
    def get_feature_importance_analysis(self, reuse_instances, top_n=20):
        """Analyze important features for detecting reuse.

        Args:
            reuse_instances (list): List of reuse instances
            top_n (int): Number of top features to return

        Returns:
            dict: Feature importance analysis
        """
        # Fixed the condition to properly check if feature_names array is empty
        if not reuse_instances or self.feature_names is None or self.feature_names.size == 0:
            return {}

        feature_counts = defaultdict(int)
        for instance in reuse_instances:
            if instance.get('top_shared_features'):
                for feature in instance['top_shared_features']:
                    feature_counts[feature] += 1

        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return {
            'top_discriminative_features': top_features,
            'total_unique_features_used': len(feature_counts),
            'feature_usage_distribution': dict(feature_counts)
        }
    def save_results_for_experiment(self, reuse_instances, summary_metrics, results_dir, filename_base, config_id):
        os.makedirs(results_dir, exist_ok=True)

        config_folder = Path(results_dir) / f"config_{config_id}"
        config_folder.mkdir(parents=True, exist_ok=True)

        feature_analysis = self.get_feature_importance_analysis(reuse_instances)

        results_file = config_folder / f"{filename_base}_tfidf_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'method': 'tfidf_vector_similarity',
                'configuration': {
                    'ngram_range': self.ngram_range,
                    'max_features': self.max_features,
                    'similarity_metric': self.similarity_metric,
                    'similarity_threshold': self.similarity_threshold,
                    'stemming_used': self.use_stemming,
                    'stopwords_removed': self.remove_stopwords
                },
                'reuse_instances': reuse_instances,
                'summary_metrics': summary_metrics,
                'feature_analysis': feature_analysis,
                'timestamp': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

        if reuse_instances:
            csv_instances = []
            for instance in reuse_instances:
                csv_instance = {k: v for k, v in instance.items() if not isinstance(v, (list, dict)) or k == 'top_shared_features'}
                if 'top_shared_features' in csv_instance:
                    csv_instance['top_shared_features'] = '; '.join(csv_instance['top_shared_features'])
                csv_instances.append(csv_instance)
            instances_df = pd.DataFrame(csv_instances)
            instances_df.to_csv(config_folder / f"{filename_base}_tfidf_instances.csv", index=False)

        self._save_detailed_txt_report(reuse_instances, summary_metrics, feature_analysis, config_folder, filename_base)

        logger.info(f"TF-IDF results saved to {config_folder}")
        return str(results_file)

    def _save_detailed_txt_report(self, reuse_instances, summary_metrics, feature_analysis, results_dir, filename_base):
        """Save a detailed, human-readable TXT report."""
        txt_file = results_dir / f"{filename_base}_detailed_report.txt"

        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TF-IDF TEXT REUSE INSTANCES - {filename_base.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: ngram_range={self.ngram_range}, max_features={self.max_features}, "
                    f"similarity_metric={self.similarity_metric}, threshold={self.similarity_threshold}, "
                    f"stemming={self.use_stemming}, stopwords_removed={self.remove_stopwords}\n")
            f.write("=" * 80 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total text reuse instances found: {len(reuse_instances)}\n")
            f.write(f"Processing time: {summary_metrics['processing_time_seconds']:.2f} seconds\n")
            f.write(f"Segments analyzed: {summary_metrics['total_segments_analyzed']}\n")
            f.write(f"Comparisons made: {summary_metrics['total_comparisons_made']}\n")
            f.write(f"Vocabulary size: {summary_metrics['vocabulary_size']} features\n")
            if len(reuse_instances) > 0:
                f.write(f"Average TF-IDF similarity: {summary_metrics['tfidf_similarity_mean']:.3f}\n")
                f.write(f"Average cosine similarity: {summary_metrics['cosine_similarity_mean']:.3f}\n")
                f.write(
                    f"TF-IDF similarity range: {summary_metrics['tfidf_similarity_min']:.3f} - {summary_metrics['tfidf_similarity_max']:.3f}\n")
                f.write(f"Average shared features: {summary_metrics['avg_shared_features']:.1f}\n")
                f.write(f"High similarity instances (≥0.8): {summary_metrics['high_similarity_count']}\n")
                f.write(f"Medium similarity instances (0.5-0.8): {summary_metrics['medium_similarity_count']}\n")
                f.write(f"Low similarity instances (<0.5): {summary_metrics['low_similarity_count']}\n")
                f.write(f"Reuse rate: {summary_metrics['reuse_rate']:.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            if feature_analysis.get('top_discriminative_features'):
                f.write("TOP DISCRIMINATIVE FEATURES\n")
                f.write("-" * 40 + "\n")
                f.write("Features most frequently shared between similar segments:\n")
                for i, (feature, count) in enumerate(feature_analysis['top_discriminative_features'][:10], 1):
                    f.write(f" {i:2d}. {feature:<20} (appears in {count} comparisons)\n")
                f.write(f"\nTotal unique features used: {feature_analysis['total_unique_features_used']}\n")
                f.write("\n" + "=" * 80 + "\n\n")

            if not reuse_instances:
                f.write("No text reuse instances found above the similarity threshold.\n")
                return

            sorted_instances = sorted(reuse_instances, key=lambda x: x['tfidf_similarity'], reverse=True)
            f.write("DETAILED TEXT REUSE INSTANCES\n")
            f.write("=" * 80 + "\n")
            f.write(f"Showing all {len(sorted_instances)} instances (sorted by TF-IDF similarity):\n\n")

            for i, instance in enumerate(sorted_instances, 1):
                f.write(f"Instance {i}:\n")
                f.write(f" Notebooks: {instance['notebook1']} ↔ {instance['notebook2']}\n")
                f.write(f" Files: {instance['file1']} ↔ {instance['file2']}\n")
                f.write(
                    f" Segments (on page): {instance['segment1_index_on_page']} ↔ {instance['segment2_index_on_page']}\n")
                if instance.get('segment1_page_key') and instance.get('segment2_page_key'):
                    f.write(f" Pages: {instance['segment1_page_key']} ↔ {instance['segment2_page_key']}\n")
                f.write(f" TF-IDF Similarity: {instance['tfidf_similarity']:.3f}\n")
                f.write(f" Shared Features: {instance['shared_features']}\n")
                f.write(f" Features in Text 1: {instance['total_features_1']}\n")
                f.write(f" Features in Text 2: {instance['total_features_2']}\n")
                f.write(f" Word Counts: {instance['segment1_word_count']} ↔ {instance['segment2_word_count']}\n")
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
                if instance.get('top_shared_features') and len(instance['top_shared_features']) > 0:
                    f.write(f" Top Shared Features: {', '.join(instance['top_shared_features'])}\n")
                f.write("-" * 80 + "\n")
                if i % 5 == 0 and i < len(sorted_instances):
                    f.write("\n")
            f.write("\nEND OF DETAILED REPORT\n")
            f.write("=" * 80 + "\n")

    def _save_metrics_txt_report(self, summary_metrics, feature_analysis, results_dir, filename_base):
        """Save a comprehensive metrics report in TXT format."""
        with open(os.path.join(results_dir, f"{filename_base}_metrics_report.txt"), 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"COMPREHENSIVE TF-IDF METRICS REPORT\n")
            f.write(f"{filename_base.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("CONFIGURATION PARAMETERS\n")
            f.write("-" * 30 + "\n")
            f.write(f"N-gram range: {summary_metrics['ngram_range_min']}-{summary_metrics['ngram_range_max']} grams\n")
            f.write(f"Max features: {summary_metrics['max_features_limit']}\n")
            f.write(f"Similarity metric: {summary_metrics['similarity_metric']}\n")
            f.write(f"Similarity threshold: {summary_metrics['similarity_threshold']}\n")
            f.write(f"Stemming used: {summary_metrics['stemming_used']}\n")
            f.write(f"Stopwords removed: {summary_metrics['stopwords_removed']}\n\n")

            f.write("BASIC STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total instances found: {summary_metrics['total_instances']}\n")
            f.write(f"Total segments analyzed: {summary_metrics['total_segments_analyzed']}\n")
            f.write(f"Total comparisons made: {summary_metrics['total_comparisons_made']}\n")
            f.write(f"Vocabulary size: {summary_metrics['vocabulary_size']} features\n")
            f.write(f"Reuse rate: {summary_metrics['reuse_rate']:.6f}\n\n")

            if summary_metrics['total_instances'] > 0:
                f.write("SIMILARITY STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write("Primary TF-IDF Similarity:\n")
                f.write(f" Mean: {summary_metrics['tfidf_similarity_mean']:.4f}\n")
                f.write(f" Standard deviation: {summary_metrics['tfidf_similarity_std']:.4f}\n")
                f.write(f" Median: {summary_metrics['tfidf_similarity_median']:.4f}\n")
                f.write(f" Minimum: {summary_metrics['tfidf_similarity_min']:.4f}\n")
                f.write(f" Maximum: {summary_metrics['tfidf_similarity_max']:.4f}\n\n")
                f.write("Alternative Similarity Measures:\n")
                f.write(f" Cosine similarity mean: {summary_metrics['cosine_similarity_mean']:.4f}\n")
                f.write(f" Cosine similarity std: {summary_metrics['cosine_similarity_std']:.4f}\n")
                f.write(f" Euclidean similarity mean: {summary_metrics['euclidean_similarity_mean']:.4f}\n")
                f.write(f" Euclidean similarity std: {summary_metrics['euclidean_similarity_std']:.4f}\n")
                f.write(f" Jaccard features mean: {summary_metrics['jaccard_features_mean']:.4f}\n")
                f.write(f" Jaccard features std: {summary_metrics['jaccard_features_std']:.4f}\n\n")
                f.write("Feature Sharing Statistics:\n")
                f.write(f" Average shared features: {summary_metrics['avg_shared_features']:.2f}\n")
                f.write(f" Maximum shared features: {summary_metrics['max_shared_features']}\n")
                f.write(f" Minimum shared features: {summary_metrics['min_shared_features']}\n\n")

                f.write("SIMILARITY CATEGORIES\n")
                f.write("-" * 30 + "\n")
                f.write(f"High similarity (≥0.8): {summary_metrics['high_similarity_count']} instances\n")
                f.write(f"Medium similarity (0.5-0.8): {summary_metrics['medium_similarity_count']} instances\n")
                f.write(f"Low similarity (<0.5): {summary_metrics['low_similarity_count']} instances\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total processing time: {summary_metrics['processing_time_seconds']:.2f} seconds\n")
            f.write(f"Vectorization time: {summary_metrics['vectorization_time_seconds']:.2f} seconds\n")
            f.write(
                f"Similarity calculation time: {summary_metrics['similarity_calculation_time_seconds']:.2f} seconds\n")
            f.write(f"Comparisons per second: {summary_metrics['comparisons_per_second']:.1f}\n\n")

            if feature_analysis.get('top_discriminative_features'):
                f.write("FEATURE ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total unique features in similarities: {feature_analysis['total_unique_features_used']}\n")
                f.write("Most discriminative features:\n")
                for i, (feature, count) in enumerate(feature_analysis['top_discriminative_features'][:15], 1):
                    f.write(f" {i:2d}. {feature:<25} ({count:3d} occurrences)\n")
                f.write("\n")

            f.write("ANALYSIS SUMMARY\n")
            f.write("-" * 30 + "\n")
            if summary_metrics['total_instances'] > 0:
                effectiveness = ("High" if summary_metrics['reuse_rate'] > 0.01 else
                                 "Medium" if summary_metrics['reuse_rate'] > 0.001 else "Low")
                f.write(f"Detection effectiveness: {effectiveness}\n")
                f.write(f"Average similarity quality: {summary_metrics['tfidf_similarity_mean']:.3f}\n")
                if summary_metrics['high_similarity_count'] > 0:
                    f.write(
                        f"Strong matches found: {summary_metrics['high_similarity_count']} high-confidence instances\n")
                processing_efficiency = ("Fast" if summary_metrics['comparisons_per_second'] > 1000 else
                                         "Moderate" if summary_metrics['comparisons_per_second'] > 100 else "Slow")
                f.write(
                    f"Processing efficiency: {processing_efficiency} ({summary_metrics['comparisons_per_second']:.0f} comp/sec)\n")
                f.write(f"TF-IDF advantages: Semantic similarity, vocabulary weighting, scalable\n")
                f.write(f"Feature space: {summary_metrics['vocabulary_size']} dimensions\n")
                avg_features_per_doc = summary_metrics['avg_shared_features'] if summary_metrics[
                                                                                     'avg_shared_features'] > 0 else 0
                vocab_density = (avg_features_per_doc / summary_metrics['vocabulary_size'] * 100
                                 if summary_metrics['vocabulary_size'] > 0 else 0)
                f.write(f"Average vocabulary density: {vocab_density:.2f}%\n")
            else:
                f.write("No text reuse detected above the specified threshold.\n")
                f.write("Consider lowering the similarity threshold or adjusting TF-IDF parameters.\n")
                f.write("For historical texts, try different n-gram ranges or preprocessing options.\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("END OF TF-IDF METRICS REPORT\n")
            f.write("=" * 60 + "\n")


def main():
    """Main function for TF-IDF-based text reuse analysis."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    base_dir = project_root / "preprocessing"
    notebooks = ['14e', '14g']
    filenames = ['page_to_text.json']
    results_dir = project_root / "results_text_reuse" / "results_tfidf"

    configs = [
        (
            2,
            {'similarity_threshold': 0.25, 'ngram_range': (1, 1), 'max_features': 5000, 'use_stemming': True,
             'remove_stopwords': True, 'similarity_metric': 'cosine', 'min_segment_length': 25, 'min_words': 4}
        ),
        (
            6,
            {'similarity_threshold': 0.2, 'ngram_range': (1, 3), 'max_features': 10000, 'use_stemming': True,
         'remove_stopwords': True, 'similarity_metric': 'cosine', 'min_segment_length': 35, 'min_words': 5},
        )
    ]

    all_experiment_results = []

    print("Starting TF-IDF Analysis for 18th-Century Historical Text (Sir David Humphry)")
    print("Running configurations 2 and 112")
    print("=" * 80)

    for config_id, config in configs:
        ngram_desc = f"{config['ngram_range'][0]}-{config['ngram_range'][1]}gram" if config['ngram_range'][0] != config['ngram_range'][1] else f"{config['ngram_range'][0]}gram"
        config_name = f"Config {config_id}: TF-IDF {ngram_desc} {config['similarity_metric']}"
        if config['use_stemming']:
            config_name += " + stemming"
        if config['remove_stopwords']:
            config_name += " + stopword removal"
        else:
            config_name += " (preserving stopwords)"

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running {config_name}")
        logger.info(f"Configuration: {config}")
        logger.info(f"{'=' * 60}")

        detector = LibraryBasedTFIDFDetector(**config)
        texts_data, all_metadata = detector.load_texts(base_dir, notebooks, filenames)

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

            ngram_str = f"{config['ngram_range'][0]}to{config['ngram_range'][1]}gram"
            preprocessing = "stemmed" if config['use_stemming'] else "unstemmed"
            stopwords = "no_stopwords" if config['remove_stopwords'] else "with_stopwords"
            similarity_metric = config['similarity_metric']
            base_name = f"{os.path.splitext(filename)[0]}_tfidf_{ngram_str}_{similarity_metric}_{preprocessing}_{stopwords}"

            detector.save_results_for_experiment(reuse_instances, summary_metrics,
                                                results_dir, base_name, config_id)

            experiment_result = {
                'config_id': config_id,
                'config_name': config_name,
                'ngram_range': config['ngram_range'],
                'similarity_metric': config['similarity_metric'],
                'configuration': config,
                'filename': filename,
                'summary_metrics': summary_metrics,
                'instance_count': len(reuse_instances)
            }
            all_experiment_results.append(experiment_result)

            print(f"\n{config_name} Results for {filename}:")
            print(f" Text reuse instances found: {len(reuse_instances)}")
            print(f" Processing time: {summary_metrics['processing_time_seconds']:.2f}s")
            print(f" Mean TF-IDF similarity: {summary_metrics['tfidf_similarity_mean']:.3f}")
            print(f" Vocabulary size: {summary_metrics['vocabulary_size']}")
            print(f" Segments analyzed: {summary_metrics['total_segments_analyzed']}")
            print(f" Reuse rate: {summary_metrics['reuse_rate']:.4f}")
            if len(reuse_instances) > 0:
                print(
                    f" TF-IDF similarity range: {summary_metrics['tfidf_similarity_min']:.3f} - {summary_metrics['tfidf_similarity_max']:.3f}")
                print(f" Average shared features: {summary_metrics['avg_shared_features']:.1f}")

    print(f"\n{'=' * 80}")
    print("TF-IDF EXPERIMENT COMPLETED (Configurations 2 & 112 Only)")
    print(f"{'=' * 80}")
    for result in all_experiment_results:
        metrics = result['summary_metrics']
        print(f"Config {result['config_id']} - {result['config_name']}")
        print(f" Text reuse instances found: {result['instance_count']}")
        print(f" Mean TF-IDF similarity: {metrics['tfidf_similarity_mean']:.3f}")
        print(f" Vocabulary size: {metrics['vocabulary_size']}")
        print(f" Processing time: {metrics['processing_time_seconds']:.2f}s")
        print(f" Comparisons made: {metrics['total_comparisons_made']}")


if __name__ == "__main__":
    main()