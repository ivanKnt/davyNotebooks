
import os
import re
import json
import logging
import csv
from collections import defaultdict
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deduplicate_successive_words(text):
    """Remove successive duplicate words while preserving case and other text."""
    words = text.split()
    if not words:
        return text
    result = [words[0]]
    for i in range(1, len(words)):
        if words[i] != words[i - 1]:
            result.append(words[i])
    return ' '.join(result)

def extract_text_from_tei(notebook_id):
    xml_file_path = os.path.join('../..', 'items', notebook_id, 'tei', 'doc')
    classifications_path = os.path.join('../..', 'items', notebook_id, 'transcription', 'source', 'classifications')
    try:
        if not os.path.exists(xml_file_path):
            logger.error(f"TEI file not found at {xml_file_path}")
            return

        output_dir = os.path.join('../..', 'preprocessing', notebook_id)
        os.makedirs(output_dir, exist_ok=True)

        with open(xml_file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'xml')

        body_element = soup.find('body')
        standoff_element = soup.find('standOff')

        if body_element is None:
            logger.error(f"<body> element not found in {notebook_id}.")
            return

        # Remove <note> elements entirely before processing
        for note in body_element.find_all('note'):
            note.extract()

        # Process page by page using a static list of descendants
        descendants = list(body_element.descendants)
        pages = []
        current_page_text = []
        current_page_num = None
        page_to_annotations = {}  # Map page numbers to annotation IDs

        for element in descendants:
            if element is None:
                logger.debug("Encountered None element, skipping.")
                continue

            if element.name is None:  # NavigableString (text node)
                if element.string and element.string.strip():
                    current_page_text.append(element.string.strip())
                continue

            try:
                if element.name == 'pb':
                    if current_page_text and current_page_num:
                        pages.append((current_page_num, '\n'.join(current_page_text)))
                    current_page_num = element.get('n', 'unknown')
                    current_page_text = []
                    page_to_annotations[current_page_num] = set()
                    element.extract()  # Remove pb tag after processing
                elif element.name == 'lb':
                    if current_page_text:
                        current_page_text.append('\n')
                    element.extract()  # Remove lb tag after processing
                elif element.name == 'rs' and 'ref' in element.attrs:
                    if current_page_num:
                        annotation_id = element['ref'].lstrip('#')
                        page_to_annotations[current_page_num].add(annotation_id)
                    text_content = element.get_text(separator=' ', strip=True) if element.get_text(separator=' ', strip=True) else ''
                    if text_content:
                        current_page_text.append(text_content)
                elif element.name not in ['note', 'pb', 'lb', 'rs']:
                    text_content = element.get_text(separator=' ', strip=True) if element.get_text(separator=' ', strip=True) else ''
                    if text_content:
                        current_page_text.append(text_content)
            except AttributeError as e:
                logger.error(f"AttributeError on element {element}: {e}")
                continue

        if current_page_text and current_page_num:
            pages.append((current_page_num, '\n'.join(current_page_text)))

        # Extract all entities metadata from <standOff>
        all_entities_metadata = {
            'persons': {},
            'places': {},
            'chemicals': {},
            'events': {},
            'orgs': {},
            'works': {}
        }

        if standoff_element:
            for person in standoff_element.find_all('person'):
                person_id = person.get('xml:id')
                pers_name_elem = person.find('persName')
                pers_name = pers_name_elem.get_text(strip=True) if pers_name_elem else 'Unknown'
                birth_elem = person.find('birth')
                birth = birth_elem.get_text(strip=True) if birth_elem else None
                death_elem = person.find('death')
                death = death_elem.get_text(strip=True) if death_elem else None
                note_elem = person.find('note')
                description = note_elem.get_text(strip=True) if note_elem else ''
                all_entities_metadata['persons'][person_id] = {
                    'name': pers_name,
                    'birth': birth,
                    'death': death,
                    'description': description
                }

            for place in standoff_element.find_all('place'):
                place_id = place.get('xml:id')
                place_name_elem = place.find('placeName')
                place_name = place_name_elem.get_text(strip=True) if place_name_elem else 'Unknown'
                note_elem = place.find('note')
                description = note_elem.get_text(strip=True) if note_elem else ''
                all_entities_metadata['places'][place_id] = {
                    'name': place_name,
                    'description': description
                }

            for term in standoff_element.find_all('term', {'type': 'chemical'}):
                term_id = term.get('xml:id')
                name_elem = term.find('name')
                chem_name = name_elem.get_text(strip=True) if name_elem else 'Unknown'
                note_elem = term.find('note')
                description = note_elem.get_text(strip=True) if note_elem else ''
                all_entities_metadata['chemicals'][term_id] = {
                    'name': chem_name,
                    'description': description
                }

            for event in standoff_element.find_all('event'):
                event_id = event.get('xml:id')
                label_elem = event.find('label')
                event_name = label_elem.get_text(strip=True) if label_elem else 'Unknown'
                note_elem = event.find('note')
                description = note_elem.get_text(strip=True) if note_elem else ''
                all_entities_metadata['events'][event_id] = {
                    'name': event_name,
                    'description': description
                }

            for org in standoff_element.find_all('org'):
                org_id = org.get('xml:id')
                org_name_elem = org.find('orgName')
                org_name = org_name_elem.get_text(strip=True) if org_name_elem else 'Unknown'
                note_elem = org.find('note')
                description = note_elem.get_text(strip=True) if note_elem else ''
                all_entities_metadata['orgs'][org_id] = {
                    'name': org_name,
                    'description': description
                }

            for work in standoff_element.find_all('bibl'):
                work_id = work.get('xml:id')
                title_elem = work.find('title')
                title = title_elem.get_text(strip=True) if title_elem else 'Unknown'
                author_elem = work.find('author')
                author = author_elem.get_text(strip=True) if author_elem else None
                all_entities_metadata['works'][work_id] = {
                    'name': title,
                    'author': author
                }

        # Process each page into text and collect metadata
        page_to_text = {}
        page_to_entities = {}

        for page_num, page_text in pages:
            # Clean text with spaces, deduplicate successive words
            cleaned_text = re.sub(r'\n\n\n', ' ', page_text)
            cleaned_text = deduplicate_successive_words(cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            page_to_text[page_num] = cleaned_text

            # Collect and resolve entities for this page
            if page_num in page_to_annotations:
                entities_by_type = {
                    'persons': [],
                    'places': [],
                    'chemicals': [],
                    'events': [],
                    'orgs': [],
                    'works': []
                }
                for annotation_id in page_to_annotations[page_num]:
                    for entity_type, entities_map in all_entities_metadata.items():
                        if annotation_id in entities_map:
                            entity_details = entities_map[annotation_id].copy()
                            entity_details['id'] = annotation_id
                            entities_by_type[entity_type].append(entity_details)
                            break
                page_to_entities[page_num] = {k: v for k, v in entities_by_type.items() if v}
            else:
                page_to_entities[page_num] = {}

        # Save outputs
        with open(f'{output_dir}/page_to_text.json', 'w', encoding='utf-8') as f:
            json.dump(page_to_text, f, ensure_ascii=False, indent=2)

        with open(f'{output_dir}/all_entities_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(all_entities_metadata, f, ensure_ascii=False, indent=2)

        with open(f'{output_dir}/page_to_entities.json', 'w', encoding='utf-8') as f:
            json.dump(page_to_entities, f, ensure_ascii=False, indent=2)

        return f"Text extraction completed for {notebook_id}. Check {output_dir}/ for output files."

    except Exception as e:
        return f"An error occurred for {notebook_id}: {e}"

def get_page_mapping_from_tagged(notebook_id):
    """Extract page mapping from tagged file to get proper page numbers."""
    tagged_file = os.path.join('../..', 'items', notebook_id, 'transcription', 'text', 'tagged')
    if not os.path.exists(tagged_file):
        return {}

    page_mapping = {}
    try:
        with open(tagged_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all [page] tags
        page_pattern = r'\[page\]([^|]+)\|([^|]+)\|([^|]+)\[/page\]'
        matches = re.findall(page_pattern, content)

        for match in matches:
            platform_page_id = match[0]  # field1: LDC platform page ID (should be like "1", "2", etc.)
            internal_id = match[1]       # field2: internal_id (like "MS-DAVY-11405-000-00001")
            subject_id = match[2]        # field3: subject_id (numeric like "000192")

            # Map subject_id to platform page number
            try:
                page_num = int(platform_page_id)
                page_mapping[str(subject_id)] = page_num
            except ValueError:
                continue

    except Exception as e:
        logger.error(f"Error reading tagged file for page mapping: {e}")

    return page_mapping


def process_classifications(classifications_path, output_dir, notebook_id):
    overall_classification = defaultdict(int)
    per_page_classifications = defaultdict(list)
    workflow_name = None
    
    try:
        with open(classifications_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            annotation_idx = headers.index('annotations')
            subject_data_idx = headers.index('subject_data')
            workflow_name_idx = headers.index('workflow_name')
            
            for row in reader:
                if len(row) < max(annotation_idx, subject_data_idx, workflow_name_idx) + 1:
                    continue
                    
                try:
                    annotations = json.loads(row[annotation_idx])
                    subject_data = json.loads(row[subject_data_idx])
                    workflow_name = row[workflow_name_idx] if not workflow_name else workflow_name
                    
                    # Extract classifications from T2 or T3 task
                    for task in annotations:
                        if task['task'] in ['T2', 'T3']:  # Support both T2 and T3 tasks
                            classifications = task['value']
                            
                            # Skip if value is None or not a list
                            if not classifications or not isinstance(classifications, list):
                                continue
                            
                            # Count classifications
                            for cls in classifications:
                                overall_classification[cls] += 1
                            
                            # Get page information from subject_data
                            subject_key = list(subject_data.keys())[0]
                            subject_info = subject_data[subject_key]
                            
                            # Try different ways to get the page number
                            page_num = None
                            
                            # Method 1: From filename (for notebooks like 14e)
                            filename = subject_info.get('Filename', '')
                            if filename:
                                filename_match = re.search(r'_(\d+)\.jpg$', filename)
                                if filename_match:
                                    page_num = int(filename_match.group(1))
                            
                            # Method 2: From image field (for notebooks like 01a1)
                            if page_num is None:
                                image = subject_info.get('image', '')
                                if image:
                                    # Extract from patterns like "HD01a1_06.jpg" -> 6
                                    image_match = re.search(r'_(\d+)\.jpg$', image)
                                    if image_match:
                                        page_num = int(image_match.group(1))
                            
                            # Method 3: From internal_id (for notebooks like 01a1)
                            if page_num is None:
                                internal_id = subject_info.get('internal_id', '')
                                if internal_id:
                                    # Extract from patterns like "dnp01a106pp" -> 6
                                    internal_match = re.search(r'(\d+)pp$', internal_id)
                                    if internal_match:
                                        page_num = int(internal_match.group(1))
                            
                            if page_num is not None:
                                per_page_classifications[str(page_num)].extend(classifications)
                            
                except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                    logger.debug(f"Skipping row due to error: {e}")
                    continue
                    
    except FileNotFoundError:
        logger.warning(f"Classifications CSV not found at {classifications_path}")
        workflow_name = "Unknown"
    except Exception as e:
        logger.error(f"Error processing classifications for {classifications_path}: {e}")
        workflow_name = "Unknown"

    classifications_data = {
        'overall_classification': workflow_name if workflow_name else f"Notebook {notebook_id}",
        'per_page_classifications': dict(per_page_classifications)
    }

    with open(f'{output_dir}/classifications.json', 'w', encoding='utf-8') as f:
        json.dump(classifications_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Classifications saved to {output_dir}/classifications.json")

    return classifications_data

def process_all_notebooks():
    items_dir = os.path.join('../..', 'items')
    if not os.path.exists(items_dir):
        logger.error(f"Items directory {items_dir} not found.")
        return

    for notebook_id in os.listdir(items_dir):
        notebook_path = os.path.join(items_dir, notebook_id)
        if os.path.isdir(notebook_path):
            tei_path = os.path.join(notebook_path, 'tei', 'doc')
            classifications_path = os.path.join(notebook_path, 'transcription', 'source', 'classifications')
            if os.path.exists(tei_path):
                logger.info(f"Processing notebook: {notebook_id}")
                # Extract text and entities
                result = extract_text_from_tei(notebook_id)
                print(result)
                # Process classifications if file exists
                if os.path.exists(classifications_path):
                    process_classifications(classifications_path, os.path.join('../..', 'preprocessing', notebook_id), notebook_id)
                else:
                    logger.warning(f"Classifications file not found for {notebook_id}, skipping classification step.")

if __name__ == "__main__":
    process_all_notebooks()
