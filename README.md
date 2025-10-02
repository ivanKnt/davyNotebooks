# Davy Notebooks

The notebooks of Sir Humphry Davy, transcribed by the [Davy Notebooks Project](https://wp.lancs.ac.uk/davynotebooks) [team](https://wp.lancs.ac.uk/davynotebooks/the-project-team/) and volunteers on [Zooniverse](https://www.zooniverse.org).

The majority of the source material is held by the [Royal Institution](https://www.rigb.org/), with the remaining notebooks held by [Kresen Kernow](https://kresenkernow.org/) (the Cornwall Centre).

<a id="toc"></a>

## Contents

- [**Components**](#comp)
    - [Images](#comp_images)
    - [HTML](#comp_html)
    - [Metadata](#comp_metadata)
    - [Transcriptions](#comp_trans)
    - [Annotations](#comp_anno)
    - [TEI](#comp_tei)
- [**Notebook Naming Conventions**](#naming)
    - [Source ID](#naming_source)
        - [Royal Institution](#naming_source_ri)
        - [Kresen Kernow](#naming_source_kk)
    - [Platform ID](#naming_platform)
- [**GitHub Repository Structure**](#github)
    - [config/](#github_config)
        - [config/autotags/](#github_config_auto)
        - [config/collection/](#github_config_coll)
        - [config/tags/](#github_config_tag)
        - [config/transcription/](#github_config_trans)
        - [config/xsl](#github_config_xsl)
    - [html/](#github_html)
        - [html/images/](#github_html_images)
        - [html/pages/](#github_html_pages)
            - [html/pages/summary/](#github_html_pages_summary)
            - [html/pages/sponsorts/](#github_html_pages_sponsors)
    - [manifest/](#github_manifest)
    - [standoff/](#github_standoff)
        - [standoff/*\<type\>*/source/](#github_standoff_source)
        - [standoff/*\<type\>*/data/](#github_standoff_data)
        - [standoff/*\<type\>*/tei](#github_standoff_tei)
    - [items/](#github_item)
        - [items/*\<item\>*/config/](#github_item_config)
            - [items/*\<item\>*/config/item](#github_item_config_item)
            - [items/*\<item\>*/config/metadata](#github_item_config_metadata)
        - [items/*\<item\>*/manifest](#github_item_manifest)
            - [items/*\<item\>*/manifest/image/](#github_item_manifest_image)
            - [items/*\<item\>*/manifest/transcription/](#github_item_manifest_trans)
            - [items/*\<item\>*/manifest/collection](#github_item_manifest_coll)
        - [items/*\<item\>*/tei/](#github_item_tei)
        - [items/*\<item\>*/transcription/](#github_item_trans)
- [**Transcription Data**](#trans)
    - [Zooniverse](#trans_zoo)
        - [Manifest](#trans_zoo_manifest)
        - [Page Data](#trans_zoo_data)
        - [Conversion To Plaintext](#trans_zoo_conv)
    - [Word](#trans_word)
        - [Conversion To Plaintext](#trans_word_conv)
    - [Intermediate Plaintext](#trans_text)
    - [Tagging](#trans_tag)
        - [Simple Tags](#trans_tag_simple)
        - [Structured Tags](#trans_tag_structured)
        - [Nesting](#trans_tag_nesting)
        - [Automated Annotation (Auto-Tagging)](#trans_tag_auto)
        - [Troubleshooting](#trans_tag_issues)
    - [Tag Dictionary](#trans_tag_dict)
        - [abbrev](#trans_tag_dict_abbrev)
        - [blockdeletion](#trans_tag_dict_blockdel)
        - [chemical](#trans_tag_dict_chemical)
        - [deletion](#trans_tag_dict_del)
        - [insertion](#trans_tag_dict_ins)
        - [line](#trans_tag_dict_line)
        - [misc](#trans_tag_dict_misc)
        - [nonenglish](#trans_tag_dict_nonenglish)
        - [nonstandard](#trans_tag_dict_nonstandard)
        - [otherwork](#trans_tag_dict_otherwork)
        - [otherwriter](#trans_tag_dict_otherwriter)
        - [overwrite](#trans_tag_dict_overwrite)
        - [page](#trans_tag_dict_page)
        - [person](#trans_tag_dict_person)
        - [place](#trans_tag_dict_place)
        - [sidebar](#trans_tag_dict_sidebar)
        - [superscript](#trans_tag_dict_superscript)
        - [term](#trans_tag_dict_term)
        - [unclear](#trans_tag_dict_unclear)
        - [underline](#trans_tag_dict_underline)        
        - [Rules](#trans_tag_dict_rules)
        - [Tables](#trans_tag_dict_tables)
            - [Tables with Column Headings](#trans_tag_dict_tables_col)
            - [Tables with Row Headings](#trans_tag_dict_tables_row)
- [**TEI Data**](#data)
    - [Metadata](#data_metadata)
    - [Images](#data_image)
    - [Annotations](#data_anno)
        - [Chemicals](#data_anno_chemical)
        - [Miscellaneous](#data_anno_misc)
        - [Non-English](#data_anno_nonenglish)
        - [Bibliographic (Other Work) References](#data_anno_otherwork)
        - [People](#data_anno_person)
        - [Places](#data_anno_place)
    - [Transcriptions](#data_trans)
        - [Page Breaks](#data_trans_page)
        - [Line Breaks](#data_trans_line)
        - [Typography](#data_trans_type)
        - [Annotations](#data_trans_anno)
- [**Build Process**](#build)
    - [Images and Manifests](#build_image)
    - [HTML](#build_html)
    - [Annotations](#build_anno)
    - [Transcriptions](#build_trans)
    - [TEI](#build_tei)

<a id="comp"></a>

## Components

The Davy Notebooks source material is composed of several elements.

<a id="comp_images"></a>

### Images and Manifests

Source images (usually TIFFs) for each notebook page are converted into tiled JPEG2000 suitable for the LDC zooming image viewer. This repository doesn't include the images themselves, but it does contain manifests containing image dimensions, which are incorporated into the notebooks' [TEI](#comp_tei) representation.

<a id="comp_html"></a>

### HTML

The collection homepage is defined by XML data which is assembled into the static page HTML.

<a id="comp_metadata"></a>

### Metadata

Each notebook's metadata (manuscript properties etc.) is defined in an XML document which is converted into the TEI header section during the build process.

<a id="comp_trans"></a>

### Transcriptions

Notebook transcriptions are supplied either as data exports from the Zooniverse ALiCE editor or as Word documents. They are split into pages and aligned with the corresponding page images using parallel image and transcription page manifests.

The source data is converted into an intermediate plaintext format which contains markup tags to denote typography (*[deletion]...[/deletion]*, *[superscript]..[/superscript]*) and links to [annotations](#comp_anno) (*[chemical]chemical_id\|Transcribed text[/chemical]*).
The markup is applied by hand and/or by an automated text-replacement process (automatic annotation, or auto-tagging).

The plaintext is then converted to [TEI](#comp_tei) for loading into the LDC platform.

<a id="comp_anno"></a>

### Annotations

The transcription text is annotated with definitions of people, places, chemicals and other details. The annotation data consists of:

- *Annotation definitions* held in Excel spreadsheets
- *Automated annotation definitions* (auto-tagging) mapping common words/phrases to annotated equivalents, applied globally across all notebooks
- *Tag definitions* mapping the plaintext markup tags (e.g. *[person]id|Name[/person]*) to their [TEI](#comp_tei) equivalents

<a id="comp_tei"></a>

### TEI

The image manifest, metadata, transcription and annotation components are used to build [Text Encoding Initiative (TEI)](https://tei-c.org) XML representations of each notebook for deployment to the [Lancaster Digital Collections (LDC)](https://digitalcollections.lancaster.ac.uk/collections/davy/1) platform.

Annotation definitions are converted into equivalent TEI records. Each notebook selectively copies in the definitions it refers to.

Each TEI document contains the following XML sections:

- `<teiHeader>` - metadata about the physical notebook and its digital representation; used to generate the "About" metadata panel in LDC
- `<facsimile>` - metadata about the page images (orientation, dimensions) used to generate [IIIF](https://iiif.io) for the LDC zooming image viewer
- `<standOff>` - data records for [annotation definitions](#comp_anno); used to generate the annotation panels in transcriptions
- `<text>` - paginated transcription text, including references to annotation data records (using TEI [referencing strings](https://tei-c.org/release/doc/tei-p5-doc/en/html/ref-rs.html)); used to generate the "Transcription" panel in LDC

TEI documents are self-contained and don't refer to any external documents. Annotation records are copied into each notebook's TEI `<standOff>` section from the TEI files generated from the annotation spreadsheets.

See below for details of the [TEI data structure](#data)

<a id="naming"></a>

## Notebook Naming Conventions

<a id="naming_source"></a>

### Source ID

Notebook subfolders are named using an internal project naming convention based on the original classmarks in the Royal Institution and Kresen Kernow catalogues.

<a id="naming_source_ri"></a>

#### Royal Institution

`RI/HD/<NN>/<SUFFIX><SUFFIX2>` (<SUFFIX> and <SUFFIX2> optional)

Examples:
- RI/HD/03/B10 -> 03b10
- RI/HD/06 -> 06
- RI/HD/21/A -> 21a

<a id="naming_source_kk"></a>

#### Kresent Kernow

`GS-<NN>-<SUFFIX>`

Examples:
- GS-06-01 -> gs61

<a id="naming_platform"></a>

### Platform ID

Lancaster Digital Collections uses a standard collection/item/page ID format:

`<PREFIX>-<NNNNN>-000-<PPPPP>`

- `<PREFIX>` = collection ID (`MS-DAVY` for this collection)
- `<NNNNNN>` = item/notebook ID (5-digit zero-padded number)
- `000` = fixed string delimiting item/page IDs
- `<PPPPP>` = page ID within item (5-digit zero-padded number)

The internal project names are converted to LDC item IDs (the `<NNNNN>` part of the full item/page ID) as follows:

* Royal Institution: `1<II><SS>`
* Kresen Kernow: `0<II><SS>`

where:

- `II` = 2-digit notebook number (01-22)
- `SS` = 2-digit suffix calculated as follows:
    - `01-04`: (alphabetic position of suffix letter - 1) * 20 + (numeric suffix)
    - `06-22`: alphabetic position of suffix letter, or `00` if no suffix

Examples:

- `01a1` = MS-DAVY-10101
- `03b1` = MS-DAVY-10321 (`SS` = (2 - 1) * 20 + 1 = 21)
- `03b10` = MS-DAVY-10330 (`SS` = (2 - 1) * 10 + 10 = 30)
- `06` = MS-DAVY-10600
- `21a` = MS-DAVY-12101`
- `gs64` = MS-DAVY-00604

<a id="github"></a>

## GitHub Repository Structure

<a id="github_config"></a>

### config/

This folder contains configuration, data and XSL transformations used in the data processing and building of the notebook TEI.

<a id="github_config_auto"></a>

#### config/autotags

Configuration for the automated annotation (auto-tagging) process.

`config/autotags/source` contains the original Excel spreadsheet containing the text-to-annotation mapping. This is unpacked into `config/autotags/data/`.

The spreadsheet data is extracted to a tab-delimited text file, `config/autotags/text`

<a id="github_config_coll"></a>

#### config/collection/

`config/collection/collection.xml` contains basic details about the collection, used to construct the collection metadata for the LDC platform.

<a id="github_config_tag"></a>

#### config/tags/

`config/tags/tags.xml` defines the mapping from the plaintext markup tags to TEI.

`config/tags/tags-name.xml` defines a specialised tag vocabulary for names of people and places, mainly used in the auto-tagging configuration.

<a id="github_config_trans"></a>

#### config/transcription/

`config/transcription/davy.xsl` configures the default LDC content processing XSL for project-specific features.

<a id="github_config_xsl"></a>

#### config/xsl/

This folder contains the project-specific XSL for:
- creating the TEI header from an item's metadata
- creating the TEI facsimile from an item's page manifest
- creating the TEI standoff metadata from the annotation definitions spreadsheets
- creating the TEI transcription from an item's plaintext transcription
- assembling a complete TEI document from the component parts
- extracting auto-tagging configuration from its Excel spreadsheet
- creating HTML transcription pages from the TEI transcription

<a id="github_html"></a>

### html/

Images, source files and generated HTML for static webpage content (collection landing-page).

<a id="github_html_images"></a>

#### html/images/

Images embedded in the home page. This should always include a JPEG image *`collection-slug.jpg`* (e.g. `davy.jpg`) All other images should be prefixed with *`collection-slug-`* (e.g. `davy-image1.jpg`)

Older collections may also have `html/images/collectionsView/collection-`*`collection-slug.jpg`* (e.g. `collection-davy.jpg`) as a 150x150px collection thumbnail. The LDC platform no longer uses these images.

<a id="github_html_pages"></a>

#### html/pages/

This folder contains data used to build HTML fragments which are assembled by the LDC platform into the full homepage for the collection.

<a id="github_html_pages_summary"></a>

`html/pages/summary/data` is an XML document definining the main properties of the homepage, including the text content, header image and quote.

<a id="github_html_pages_sponsors"></a>

`html/pages/sponsors/data` is an XML document listing the project sponsors (name, logo, link and contribution statement). This content is rendered after the main homepage text.

<a id="github_manifest"></a>

### manifest/

`manifest/collection` is an XML document listing the items in the collection, in the order they are to appear on the platform. This file should be edited to define the correct order of items in the collection.

`manifest/json` is a JSON document derived from `manifest/collection` in the platform's native metadata format.

<a id="github_standoff"></a>

### standoff/

"Standoff" is the TEI terminology for data embedded in a TEI document which is not part of the header (e.g. technical metadata about the document or the physical entity it represents) and is also not part of the transcription text.

In the Davy Notebooks, it's used to contain the annotation definitions for various types of data (chemicals, people, places etc.). The standoff data is used to create the HTML versions of the annotations which are displayed in the transcription panel on the platform.

The annotation types used in the Davy Notebooks are:
- `chemical` - elements, compounds etc
- `miscellaneous` - general notes on the transcription text
- `nonenglish` - definitions of non-English words and phrases
- `otherwork` - bibliographic references to books and other publications
- `person` - people referenced in the notebooks
- `place` - placenames and other geographical features

Each annotation type is defined in an Excel spreadsheet with a specific column structure defining the fields for each record type. has the following folder structure:

<a id="github_standoff_source"></a>

#### standoff/*\<type\>*/source/

The source spreadsheet. The project's Microsoft Teams site contains the original copies of these files, and all updates should be applied to those copies. The files in this repository should be updated from the Teams copies.

<a id="github_standoff_data"></a>

#### standoff/*\<type\>*/data/

The unpacked spreadsheet data.

<a id="github_standoff_tei"></a>

#### standoff/*\<type\>*/tei

A TEI document containing all of the annotation records defined in the spreadsheet.

When an individual notebook is built, the records referenced in the transcription are copied into the notebook's TEI so that each notebook is self-contained.

<a id="github_item"></a>

### items/

One subfolder per item (notebook). Subfolders are named using the [source ID](#naming_source).

<a id="github_item_config"></a>

#### items/*\<item\>*/config/

<a id="github_item_config_item"></a>

`items/<item>/config/item` contains the item's internal project ID and the LDC platform ID.

<a id="github_item_config_metadata"></a>

`items/<item>/config/metadata/metadata.xml` is an XML document containing metadata about the physical notebook.

<a id="github_item_manifest"></a>

#### items/*\<item\>*/manifest/

<a id="github_item_manifest_image"></a>

##### items/*\<item\>*/manifest/image/

`items/<item>/manifest/image/source` is a text file listing the source image filenames (one image per page) in document page order.

`items/<item>/manifest/image/conv` is a corresponding list of images after conversion to tiled JPEG2000. File format:

`JP2k-image-filename \t height-in-pixels \t width-in-pixels`

<a id="github_item_manifest_trans"></a>

##### items/*\<item\>*/manifest/transcription/

`items/<item>/manifest/transcription/source` is a list of page transcription files from the transcription source data (usually a Zooniverse data export). Note that this does not necessarily correspond directly with the list of images - for example, blank pages may not appear in the transcription manifest.

`items/<item>/manifest/transcription/data` is an edited version of the source transcription manifest with placeholders (`# filename` lines) for blank pages. This file should directly correspond to the `manifest/image/conv` list to provide a mapping between a page image and its transcription.

##### items/*\<item\>*/manifest/collection

<a id="github_item_manifest_coll"></a>

`items/<item>/manifest/collection` is an XML document derived from the manifest text files which links page images to their LDC platform ID. This is used to assemble the complete TEI document for the notebook.

<a id="github_item_tei"></a>

#### items/*\<item\>*/tei/

This folder contains the TEI generated from the various data sources for the notebook. The `facsimile`, `header`, `standoff` and `text` files respectively contain the image details, metadata, annotation records and transcription text. The `doc` file is the complete document assembled from these components.

`items/<item>/tei/standoff` is created by copying only the annotation records referenced in the transcription text from the [`standoff/<type>/tei`](#github_standoff_tei) files.

<a id="github_item_trans"></a>

#### items/*\<item\>*/transcription/

The `items/*\<item\>*/transcription/source/` folder contains the raw data for the transcriptions, usually a zip file exported from the Zooniverse ALiCE editor or a Word docx file.

The source data is unpacked into the `items/<item>/transcription/data/` folder. The data is intentionally left unedited so that it can be replaced by subsequent versions.

The `items/<item>/transcription/text/` folder contains the basic plaintext of the transcriptions, usually generated from the Zooniverse data exports, or extracted from the Word docx file (for the 8 pilot notebooks). The filenames reflect stages in a processing workflow (to be implemented in the future):

- `invalid` is the raw data extracted from the source
- `valid` is a hand-edited version of the raw data (with corrections, additional tagging etc.) which has passed an automated validation stage (not yet implemented)
- `tagged` is the result of applying the automated annotation process to the `valid` file; this is the final plaintext stage before conversion to TEI.

Any files with names other than these are temporary or backup files and can be ignored.

<a id="trans"></a>

## Transcription Data

<a id="trans_zoo"></a>

### Zooniverse

For notebooks transcribed in Zooniverse, the source data in `items/<item>/transcription/source/` is a zip file, unpacked into `items/<item>/transcription/data`.

The zip contents a manifest file `transcriptions_metadata.csv` and a set of folders, one folder per page, `transcription_<transid>` where *<transid>* is the numeric transcription ID from the first column of the `transcriptions_metadata.csv`.

The Zooniverse transcription workflow collects multiple transcription attempts for each text line (for the Davy Notebooks, 3 attempts) and generates a "consensus" transcription by comparing each attempt. This can be overridden by the project editors, so the consensus text should be considered definitive.

<a id="trans_zoo_manifest"></a>

#### Manifest

The `transcriptions_metadata.csv` manifest includes a Zooniverse numeric page ID (used in each page's folder and file names) and a project-defined internal ID of the form:

```
dnp<item-number><page-number>pp
```

Item-number is the [source ID](#naming_source).

Page-number is usually a zero-padded number (occasionally inconsistently within a single manifest) but sometimes includes a suffix where extra pages have been inserted. It's therefore best to ignore this as a reliable source of page numbering. 

`transcriptions_metadata.csv` is **not** sorted into page order and does **not** necessarily include every page in a notebook (for example, it usually omits blank pages, as these weren't submitted to Zooniverse for transcription), The unedited manifest is copied to  `items/<item>/manifest/transcription/source`

Th `items/<item>/manifest/transcription/data` manifest is a simplified, correctly-ordered and correctly-padded list that includes placeholders for blank pages and corresponds line-for-line with the `items/<item>/manifest/images/conv` image manifest. This file should be used for any analysis which requires correct page ordering.

The page numbers that appear in the LDC platform are the line numbers of the pages in `items/<item>/manifest/transcription/data`.

<a id="trans_zoo_data"></a>

#### Page Data

The folder for each page contains 4 files:

- `consensus_text_<transid>.txt` is the plaintext of the whole page, aggregated from the consensus text of each line. Currently this is the only file used by the text processing workflow.
- `raw_data_<transid>.json` is a JSON file containing the complete data for the page transcription. Lines are grouped into frames (usually a single one, "frame0"). The data for each line includes its orientation ("line_slope"), the individual transcription attempts for each word ("clusters_text") and the "consensus_text" and "edited_consensus_text" of the whole line. "edited_consensus_text" is the definitive line text.
- `transcription_line_metadata_<transid>.csv` is a CSV version of the `raw_data_<transid>.json` file.
- `transcription_metadata_<transid>.csv` is the single line of transcription parameter metadata for this page. The manifest file `transcriptions_metadata.csv` is an aggregate of these lines for every page.

<a id="trans_zoo_conv"></a>

#### Conversion To Plaintext

The [intermediate plaintext](#trans_text) transcription for a notebook is the concatenation of all pages' `consensus_text_<transid>.txt` with page breaks added 

<a id="trans_word"></a>

### Word

The pilot notebooks initially transcribed as Microsoft Word documents are stored in `items/<item>/transcription/source/` and unpacked into `items/<item>/transcription/data/`.

The document contents basically follow the [intermediate plaintext](#trans_text) format. Page breaks are denoted by text lines (e.g. "RI MS HD/13/F, front endpaper"). and page order is assumed from the document content ordering.

<a id="trans_word_conv"></a>

#### Conversion To Plaintext

Page breaks are identified from the document content. 

Page content is taken literally from the document. Limited detection of document styles (e.g. strikethrough) may be used to infer some tags (e.g. \[deletion\] for strikethrough).

Each page is extracted to a plaintext file in `items/<item>/transcription/data`. These are concatenated with page breaks added to form the intermediate plaintext.

<a id="trans_text"></a>

### Intermediate Plaintext

The intermediate plaintext transcription of a notebook is a simple text version of the source content. It's initially created in `items/<item>/transcription/text/invalid` to denote that it's unchecked from source and should be validated before conversion to TEI.

Each page is preceded by a [\[page\]](#trans_tag_dict_page) tag containing pagination details.

Each line is exactly as transcribed, with line breaks occurring as they appear on the page.

<a id="trans_tag"></a>

### Tagging

Intermediate plaintext may contain markup (tags) added by Zooniverse and project editors. Most tags have an opener (e.g. `[underline]`) and a closer (e.g. `[/underline]`), although a few are empty and have only openers.

Tags are defined in the `config/tags/tags.xml` configuration file, which specifies the opening and closing text of the tag, and the corresponding TEI it's converted into. The syntax of this file is described in the file itself.

Tags are either *simple* or *structured*.

<a id="trans_tag_simple"></a>

#### Simple Tags

Simple tags enclose a piece of text:

```
Plain text [underline]underlined text[/underline] More plain text
```

Some tags are inserted by [Zooniverse](#trans_tag_zoo) as a result of actions by transcribers in the Zooniverse user interface.

<a id="trans_tag_structured"></a>

#### Structured Tags

Structured tags enclose a set of data fields, delimited by | (pipe symbol/vertical bar), 

The most common usage of structured tags is to link transcription text to annotation records:

```
[person]person_001|Sir Humphry Davy[/person]
```

The general form is:

```
[entity]entity_id|text as it appears in transcription[/entity]
```

*entity_id* is the value in the ID column of the related annotation spreadsheet.

More complex forms are supported, but this simple form is mostly used throughout the transcriptions.

The text in a structured tag may contain simple tags but cannot contain other structured tags.

<a id="trans_tag_nesting"></a>

#### Nesting

Simple tags can enclose other simple and structured tags.

Nested tags must be correctly closed within their enclosing tag. This follows the same principle as HTML and XML. The intermediate plaintext is converted into XML (TEI) so it must result in well-formed XML content.

Correct:
```
[underline]Underlined [superscript]superscript[/superscript] text[/underline]
```
(The `[superscript]` tag is opened and closed within the `[underline]` tag.)

Incorrect:
```
[underline]Underlined [superscript]superscript[/underline] text[/superscript]
```
(This is not correctly nested - `[superscript]` is opened within `[underline]` so it must be closed before the `[/underline]`).

<a id="trans_tag_auto"></a>

#### Automated Annotation (Auto-tagging)

The auto-tagging process is a simple text-replacement system that replaces frequently-occurring words/phrases with tagged equivalents.

Due to its limitations, care is required in choosing text for auto-tagging. Words/phrases, names etc. with any ambiguity are not good candidates.

The source spreadsheet defines the text to be matched and the tagged replacement. It uses an abbreviated tagging syntax to avoid repetition of the matched text. A definition such as:

```
Humphry Davy        [person]person_001[/person]
```

is expanded in the intermediate plaintext to:

```
[person]person_001|Humphry Davy[/person]
```

(i.e. with the matched text included in the full tag syntax).

<a id="trans_tag_issues"></a>

#### Troubleshooting Issues

Tagging issues can arise because of errors in the transcription text, the auto-tagging data or the annotation data spreadsheets.

Tags must be correctly closed in the right places. Failure to correctly close a tag can lead to text formatting errors (e.g. subsequent page text being struck-through or superscripted) and sometimes missing page transcriptions. Common typos to watch out for are incomplete or incorrect brackets (`[underline[`), missing / characters (e.g. `[underline]Underlined[underline] non-underlined` instead of `[underlined]Underlined[/underlined] non-underlined`)

- **Overunning formatting**: e.g. extended deletion strikethrough, superscript - check that the tag introducing the formatting is correctly closed at the right point. The close tag may be omitted, mistyped, or incorrectly nested.

- **Non-functioning annotations**: check that structured annotation tags reference genuine IDs from the annotation data spreadsheets, and that IDs are correctly delimited by | characters.

- **Overrunning annotation links**: check that structured annotation tags are correctly closed. 

- **Auto-tagging**: If the transcription plaintext looks fine but there are formatting errors in the final HTML transcription, check the auto-tagging spreadsheet for errors. For example, if text formatting breaks after the text "Humphry Davy" but there's no obvious error in the transcribed text, check the auto-tagging spreadsheet for entries with the text "Humphry Davy" and verify the tags in the entry. Bear in mind how auto-tagging expands text definitions, and if the matched text introduces unbalanced tags when exoanded, this can cause formatting issues.

<a id="trans_tag_dict"></a>

### Tag Dictionary

The `[deletion]`, `[insertion]`, `[superscript]`, `[unclear]` and `[underline]` tags are used by Zooniverse to indicate text features highlighted by transcribers.

All other tags are defined for use in LDC projects and for the Davy Notebooks project specifically.

For each tag, the plaintext form and the equivalent [TEI](#data) is given. Where the TEI requires a link between multiple elements (such as a term and its gloss, or a referencing string and its standoff metadata) an item-specific xml:id value is generated by the build process; this is unrelated to the IDs used to link annotated text to annotation data records.

Structured tags are implicitly surrounded by TEI [referencing strings](#data_trans_anno) which link the text to the annotation standoff record using the ID field. These are omitted for brevity.

<a id="trans_tag_dict_abbrev"></a>

#### abbrev

A general abbreviation.

Do not use for abbreviations in complex tags - use the correction field of the complex tag to specify the expansion.

```
[abbrev]id|text|correction[/abbrev]
```

```xml
<choice>
  <abbr>text</abbr>
  <expan>correction</expan>
</choice>
```

```
[abbrev]id|text|correction|lang[/abbrev]
```

```xml
<choice>
  <abbr xml:lang="lang">text</abbr>
  <expan xml:lang="lang">correction</expan>
</choice>
```

<a id="trans_tag_dict_blockdel"></a>

#### blockdeletion

Multi-line text section deleted by the manuscript author.

```
[blockdeletion]
Deleted line
Deleted line
[/blockdeletion]
```

```xml
<del type="blockStrikethrough">
  Deleted line<lb/>
  Deleted line<lb/>
</del>
```

<a id="trans_tag_dict_chemical"></a>

#### chemical

A chemical element, compound or related term.

```
[chemical]id|text[/chemical]
```

```xml
<term xml:id="id123">text</term>
```

```
[chemical]id|text|||lang[/chemical]
```

```xml
<term xml:id="id123" xml:lang="lang">text</term>
```

```
[chemical]id|text||gloss[/chemical]
```

```xml
<term xml:id="id123">text</term>
<gloss target="#id123">gloss</gloss>
```

```
[chemical]id|text||gloss|lang[/chemical]
```

```xml
<term xml:id="id123" xml:lang="lang">text</term>
<gloss target="#id123">gloss</gloss>
```

```
[chemical]id|text|correction[/chemical]
```

```xml
<term xml:id="id123">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</term>
```

```
[chemical]id|text|correction||lang[/chemical]
```

```xml
<term xml:id="id123" xml:lang="lang">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</term>
```

```
[chemical]id|text|correction|gloss[/chemical]
```

```xml
<term xml:id="id123">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</term>
<gloss target="#id123">gloss</gloss>
```

```
[chemical]id|text|correction|gloss|lang[/chemical]
```

```xml
<term xml:id="id123" xml:lang="lang">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</term>
<gloss target="#id123">gloss</gloss>
```

<a id="trans_tag_dict_del"></a>

#### deletion

Text deleted by the manuscript author.

```xml
<del rend="strikethrough">deleted text</del>
```

<a id="trans_tag_dict_ins"></a>

#### insertion

Text inserted above the line.

```
[insertion]inserted text[/insertion]
```

```xml
<add>inserted text</add>
```

<a id="trans_tag_dict_line"></a>

#### line

A line break.

```
Line 1[line]Line 2
```

```xml
Line 1<lb/>
Line 2
```

<a id="trans_tag_dict_misc"></a>

#### misc

A miscellaneous note annotation.

```
[misc]id|text[/misc]
```

```xml
<term xml:id="id123">text</term>
```

<a id="trans_tag_dict_nonenglish"></a>

#### nonenglish

A non-English word or phrase.

For non-English chemicals, bibliographic citations etc. use the lang field of the complex tag instead of [nonenglish].

```
[nonenglish]id|lang|text[nonenglish]
```

```xml
<foreign xml:id="id123" xml:lang="lang">text</foreign>
```

<a id="trans_tag_dict_nonstandard"></a>

#### nonstandard

A non-standard spelling or use of a word or phrase.

For non-standard usage of chemicals and other terms, use the correction field of the complex tag to specify the correction instead of [nonstandard].

```
[nonstandard]id|text[/nonstandard]
```

```xml
<distinct xml:id="id123">text</distinct>
```

```
[nonstandard]id|text||gloss[/nonstandard]
```

```xml
<distinct xml:id="id123">text</distinct>
<gloss target="#id123">gloss</gloss>
```

```
[nonstandard]id|text|correction[/nonstandard]
```

```xml
<distinct xml:id="id123">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</distinct>
```

```
[nonstandard]id|text|correction|gloss[/nonstandard]
```

```xml
<distinct xml:id="id123">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</distinct>
<gloss target="#id123">gloss</gloss>
```

<a id="trans_tag_dict_otherwork"></a>

#### otherwork

A reference to an external work (e.g. bibliographic citation)

```
[otherwork]id|text[/otherwork]
```

```xml
<bibl xml:id="id123">text</ref>
```

```
[otherwork]id|text|lang[/otherwork]
```

```xml
<bibl xml:id="id123" xml:lang="lamg">text</bibl>
```

<a id="trans_tag_dict_otherwriter"></a>

#### otherwriter

A reference to an external author.

```
[otherwriter]id|text[/otherwriter]
```

```xml
<author xml:id="id123">
  <name role="aut" type="person">text</name>
</author>
```

<a id="trans_tag_dict_overwrite"></a>

#### overwrite

Text replaced by overwriting with new text.

```
[overwritten]old text[overwritten][overwrite]new text[/overwrite]
```

```xml
<del type=”over”>old text</del> <add type=”over”>new text</add>
```

<a id="trans_tag_dict_page"></a>

#### page

A page break. These are generated automatically during the conversion from the transcription source data to the intermediate plaintext.

```
[page]<number>|<platform-id>|<project-id>|<zooniverse-id>[/page]
[page]<number>|<platform-id>|<placeholder-text>|[/page]
```

- `<number>` is the page number (position of the entry in the item manifest).
- `<platform-id>`* is the full platform page ID (e.g. MS-DAVY-11305-000-00123).
- `<project-id>`* is the value from the first column of the `items/<item>/manifest/transcription/data` manifest (e.g. the Zooniverse project-specific ID)
- `<placeholder-text>` will appear if there's no corresponding ID for the page (e.g. a blank page omitted from a Zooniverse manifest) - this will usually begin with #.
- `<zooniverse-id>` is the numeric Zooniverse page ID from the second column of the `items/<item>/manifest/transcription/data` manifest, if available

```xml
<pb n="<number>" xml:id="pb-<number>" facs="'i<number>"/>
```

<a id="trans_tag_dict_person"></a>

#### person

A reference to a person.

```
[person]id|text[/person]
```

```xml
<name xml:id="id123" type="person">text</name>
```

```
[person]id|text|correction[/person]
```

```xml
<name xml:id="id123" type="person">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</name>
```

<a id="trans_tag_dict_place"></a>

#### place

A reference to a geographic location.

```
[place]id|text[/person]
```

```xml
<name xml:id="id123" type="place">text</name>
```

```
[place]id|text|correction[/person]
```

```xml
<name xml:id="id123" type="place">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</name>
```

<a id="trans_tag_dict_sidebar"></a>

#### sidebar

A line or block of text highlighted by a vertical bar in the margin or left side of the text.

```
[sidebar]
highlighted line
highlighted line
[/sidebar]
```

```xml
<hi rend=”sidebar-left”>
  highlighted line
  highlighted line
</hi>
```

<a id="trans_tag_dict_superscript"></a>

#### superscript

Superscript

```
[superscript]Superscript text[/superscript]
```

```xml
<hi rend="superscript">Superscript text</h1>
```

<a id="trans_tag_dict_term"></a>

#### term

A general (scientific or other) externally-defined term.    

```
[term]id|text[/term]
```

```xml
<term xml:id="id123">text</term>
```

```
[term]id|text|||lang[/term]
```

```xml
<term xml:id="id123" xml:lang="lang">text</term>
```

```
[term]id|text||gloss[/term]
```

```xml
<term xml:id="id123">text</term>
<gloss target="#id123">gloss</gloss>
```

```
[term]id|text||gloss|lang[/term]
```

```xml
<term xml:id="id123" xml:lang="lang">text</term>
<gloss target="#id123">gloss</gloss>
```

```
[term]id|text|correction[/term]
```

```xml
<term xml:id="id123">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</term>
```

```
[term]id|text|correction||lang[/term]
```

```xml
<term xml:id="id123" xml:lang="lang">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</term>
```

```
[term]id|text|correction|gloss[/term]
```

```xml
    <term xml:id="id123">
        <choice>
            <sic>text</sic>
            <corr>correction</corr>
        </choice>
    </term>
    <gloss target="#id123">gloss</gloss>
```

```
[term]id|text|correction|gloss|lang[/term]
```

```xml
<term xml:id="id123" xml:lang="lang">
  <choice>
    <sic>text</sic>
    <corr>correction</corr>
  </choice>
</term>
<gloss target="#id123">gloss</gloss>
```

<a id="trans_tag_dict_unclear"></a>

#### unclear

Unclear or illegible text

```
[unclear]Illegible text[/unclear]
```

```xml
<unclear>Illegible text</unclear>
```

<a id="trans_tag_dicgt_underline"></a>

#### underline

Underlined text

```
[underline]Underlined text[/underline]
```

```xml
<hi rend="underline">Underlined text</hi>
```

<a id="trans_tag_dict_rules"></a>

#### Rules

Lines/rules denoting section breaks

Several tags have been used to indicate a rule or line breaking up sections of text:

```
[Horizontal rule]
[Horizontal dashed rule]
[Short horizontal rule]
[Short dashed horizontal rule]
```

```xml
<milestone unit="section" rend="line"/>
<milestone unit="section" rend="line-dashed"/>
<milestone unit="section" rend="short-line"/>
<milestone unit="section" rend="short-line-dashed"/>
```

<a id="trans_tag_dict_tables"></a>

#### Tables

Tables can be represented with a set of simple tags which follow the TEI (and HTML) table structure, where a table is a set of rows, and a row is a set of cells representing the columns in that row.

The table tags are:
- `[thead]...[/thead]` defines a table title or caption.
- `[trh]...[/trh]` defines a table heading row, where each cell is a column heading.
- `[tr]...[/tr]` defines a regular table row containing cells for each column.
- `[tdh]...[/tdh]` defines a table cell functioning as a heading for its row
- `[td]...[/td]` defines a regular table cell in a row.

The two most common layouts are a table with an initial row of column headings and a table with row headings in the first column.

A table without column or row headings should follow the table with column headings pattern but omit the `[trh]...[/trh]` tag and its contents.

<a id="trans_tag_dict_tables_col"></a>

##### Table with Column Headings

This is a table with an initial header row containing the column headings [trh]…[/trh], followed by any number of regular rows.

```
[table]
  [thead]Table heading caption[/thead]
  [trh]
    [td]Heading 1[/td]
    [td]Heading 2[/td]
    [td]Heading 3[/td]
  [/trh]
  [tr]
    [td]Row 1 column 1[/td]
    [td]Row 1 column 2[/td]
    [td]Row 1 column 3[/td]
  [/tr]
  [tr]
    [td]Row 2 column 1[/td]
    [td]Row 2 column 2[/td]
    [td]Row 2 column 3[/td]
  [/tr]
[/table]
```

```xml
<table>
  <head>Table heading caption</head>
  <row role=”label”>
    <cell>Heading 1</cell>
    <cell>Heading 2</cell>
    <cell>Heading 3</cell>
  </row>
  <row>
    <cell>Row 1 column 1</cell>
    <cell>Row 1 column 2</cell>
    <cell>Row 1 column 3</cell>
  </row>
  <row>
    <cell>Row 2 column 1</cell>
    <cell>Row 2 column 2</cell>
    <cell>Row 3 column 3</cell>
  </row>    
</table>
```

<a id="trans_tag_dict_tables_row"></a>

##### Table with Row Headings

This is a table of regular rows, where the first cell of each row is a row heading `[th]...[/th]` and the remaining cells are regular `[td]...[/td]`

```
[table]
  [thead]Table heading caption[/thead]
  [tr]
    [th]Row 1 heading[/th]
    [td]Row 1 column 2[/td]
    [td]Row 1 column 3[/td]
  [/tr]
  [tr]
    [th]Row 2 heading[/th]
    [td]Row 2 column 2[/td]
    [td]Row 2 column 3[/td]
  [/tr]
[/table]
```

```xml
<table>
  <head>Table heading caption</head>
  <row>
    <cell role=”label”>Row 1 heading</cell>
    <cell>Row 1 column 2</cell>
    <cell>Row 1 column 3</cell>
  </row>
  <row>
    <cell role=”label”>Row 2 heading</cell>
    <cell>Row 2 column 2</cell>
    <cell>Row 2 column 3</cell>
  </row>
</table>
```

<a id="data"></a>

## TEI Data

The source data components are combined into a single TEI document for each notebook

All references to TEI and other XML documents use XPath notation. Understanding of XML namespaces is assumed. Note that all TEI elements are defined in the namespace "

<a id="data_metadata"></a>

### Metadata

Item mmetadata specified in [items/*\<item\>*/config/metadata/metadata.xml](#github_item_config_metadata) appears in `/TEi/teiHeader`.

Some elements of interest:
- `TEI/teiHeader/fileDesc/title`:\
   the collection title
- `TEI/teiHeader/fileDesc/publicationStmt/idno[@type="platform-coll"]`:\
   the collection slug (collection name in URLs)
- `TEI/teiHeader/fileDesc/publicationStmt/idno[@type="platform-item"]`:\
   the collection platform ID for the item (e.g. MS-DAVY-10600)
- `TEI/teiHeader/fileDesc/sourceDesc/msDesc/msIdentifier/idno`:\
   the official LDC platform ID (same as the platform ID above)
- `TEI/teiHeader/fileDesc/sourceDesc/msDesc/msIdentifier/altIdentifiers/idno`:\
   the source ID (classmark at the Royal Institution/Kresen Kernow)
- `TEI/teiHeader/fileDesc/sourceDesc/history/origin/origDate/@when`\
   or (for date ranges) `TEI/teiHeader/fileDesc/sourceDesc/history/origin/origDate/@notBefore|@notAfter`:\
   the origination date or date range - dates usually in `YYYY-MM-DD` format

<a id="data_image"></a>

### Images

Image information and pagination is encoded in `/TEI/facsimile`.

The thumbnail image for an item is specified in `/TEI/facsimile/graphic[@decls="#document-thumbnail"]`

The page images are specified in `/TEI/facsimile/surface` elements. The `graphic` element of the `surface` defines the image dimensions and platform page ID. The `zone` element defies visible areas of the page. At present, LDC only supports a single zone representing the whole page. This may develop in future.

The `xml:id` attribute of the surface (e.g. `xml:id="i1"`) is referenced by the `facs` attribute of the `<pb>` (page break) element in the `/TEI/text` transcription (e.g. `<pb facs="#i1"...`/>)

<a id="data_anno"></a>

### Annotations

Annotation records from the spreadsheets is contained in `/TEI/standOff`. Each record has an `xml:id` attribute containing the record ID from the spreadsheet (e.g. `<person xml:id="person_371">`)

The annotation records are grouped into lists by type. Where necessary,the containing list element's `type` attribute indicates the entry type.

<a id="data_anno_chemical"></a>

#### Chemicals

```xml
<listNym type="chemical">
  <nym xml:id="chemical_001">
    <form>amalgam of ammonium</form>
    <def>
      <p>Symbol/formula: 2NH3-Hg-H</p>
      <p>A grey-coloured compound, discovered by Davy and Berzelius in 1808.</p>
    </def>
  </nym>
  :
</listNym>
```

<a id="data_anno_misc"></a>

#### Miscellaneous

```xml
<listAnnotation type="miscellaneous">
  <note xml:id="misc_001" type="description">
    <term id="misc_001-term">Cryptogamia</term>
    <gloss id="misc_001-gloss" target="#misc_001-term">
      <p>A plant or a plant-like organism that reproduces by spores, without flowers or seeds</p>
    </gloss>
  </note>
  :
</listAnnotation>
```

<a id="data_anno_nonenglish"></a>

#### Non-English

<listAnnotation type="nonenglish">
  <note xml:id="nonenglish_001" type="description">
    <term id="nonenglish_001-term">acide carbonique</term>
    <gloss id="nonenglish_001-gloss" target="#nonenglish_001-term">
        <p>carbonic acid (French)</p>
    </gloss>
  </note>
  :
</listAnnotation>

<a id="data_anno_otherwork"></a>

#### Bibliographic (Other Work) References

```xml
<listBibl>

  <!-- Journal article -->
  <bibl xml:id="otherwork_003">
    <author>John Dalton</author>
    <title level="j">Memoirs of the Literary and Philosophical Society of Manchester</title>
    <title level="a" type="main">Experimental Essays, on the Constitution of Mixed Gases; on the Force of Steam or Vapour from Water and Other Liquids in Different Temperatures, both in a Torricellian Vacuum and in Air; on Evaporation; and on the Expansion of Gases by Heat</title>
    <date when="1802">1802</date>
    <biblScope unit="volume">5</biblScope>
    <biblScope unit="issue">2</biblScope>
    <biblScope unit="page" from="535" to="602">535-602</biblScope>
  </bibl>

  <!-- Book -->
  <bibl xml:id="otherwork_020">
    <author>Sometimes attributed to Cassius Longinus</author>
    <title level="a" type="main">On the Sublime</title>
    <date when="10th century">10th century</date>
    <note>In a draft poem “On the Morning” in notebook 13h, Davy’s footnote explains that his depiction of the sublime is influenced from Longinus through Sappho (74).</note>
  </bibl>
  :
```

<a id="data_anno_person"></a>

#### People

<listPerson>
  <person xml:id="person_002">
    <persName>
      <forename type="first">Anna</forename>
      <surname>Beddoes</surname>
    </persName>
    <birth when="1773"/>
    <death when="1824"/>
    <note>Anna Beddoes, née Edgeworth, the wife of Thomas Beddoes who befriended Davy when he arrived in Bristol. Critics have speculated that they had an affair because Davy wrote poems to Anna and she wrote poems for him that he copied out in his notebooks.  </note>
  </person>
  :
</listPerson>

<a id="data_anno_place"></a>

#### Places

<listPlace>
  <place xml:id="place_001">
    <placeName>Cotopaxi</placeName>
    <location>
        <geo decls="#WGS84">-0.67706501157719901 -78.4384024330882</geo>
    </location>
    <note>Cotopaxi is an active stratovolcano in the Andes Mountains, in Ecuador. The first European to climb it was Alexander Von Humboldt in 1802.</note>
  </place>
  :
</listPlace>

<a id="data_trans"></a>

### Transcriptions

Transcription text is contained in the `/TEI/text` element.

Text is unstructured except for line and page breaks - there are no sections, chapters, paragraphs etc. This makes it simpler to generate individual HTML pages, where structures spanning pages would complicate the page generation.

The whole transcription is contained in `/TEI/text/body/div`:

```xml
<TEI>
  :
  <text>
    <body> xml:lang="en">
      <div>
        <pb/>
        page text
        <pb/>
        page text
        :
      </div>
    </body>
  </text>
</TEI>
```

<a id="data_trans_page"></a>

#### Page Breaks

Pages are delimited by `pb` (page break) elements 
```xml
<pb n="*page-number*" xml:id="pb-*page-number*" facs="#i*page-number*"/>
```
*n* is the numeric page-number.

*xml:id* uniquely identifies the `pb` element.

`faces` references the `xml:id` of the corresponding `/TEI/facsimile/surface` element for the page image.

<a id="data_trans_line"></a>

#### Line Breaks

The `lb` element denotes the end of a text line:

```xml
<lb/>
```

<a id="data_trans_type"></a>

#### Typography

The `hi` (highlight) element is used to denote some typographical features:

```xml
<hi rend="superscript">Superscript text</hi>
<hi rend="underline">Underlined text</hi>
```

`unclear` denotes illegible text:

```xml
<unclear>Illegible text</unclear>
```

Deleted text is indicated by `del`:

```xml
<del rend="strikethrough">Deleted text</del>
```
A block of deleted text is also indicated by `del`:

```xml
<del type="blockStrikethrough">
Deleted text
Deleted text
</del>
```

Inserted text is denoted by `add`:
<add>Inserted text</add>

Overwritten text is represented by a combination of `add` and `del`:
<del type="over">Overwritten (deleted) text</del><add type="over">Replacement text</add>

Horizonal rules are indicated by `milestone`:

```xml
<milestone unit="section" rend="line"/>
<milestone unit="section" rend="line-dashed"/>
<milestone unit="section" rend="short-line"/>
<milestone unit="section" rend="short-line-dashed"/>
```

<a id="data_trans_anno"></a>

#### Annotations

Annotated text is linked to the corresponding annotation record in `/TEI/standOff` using a referencing string:

```xml
<rs ref="#person_371" type="person" xml:id="ttag1">
  <name xml:id="tag1" type="person">text</name>
</rs>
```

`ref` contains the *xml:id* of the standoff record.
`type` indicates the data type of the annotation.
`xml:id` uniquely identifies this referencing string element.

Within the `rs` element, the text will be marked up differently depending on the 
[type of annotation](#trans_tag_dict). The `xml:id` is related to the `xml:id` of its child element by prefixing the child ID with "t" (e.g. child `xml:id` = "tag123", rs `xml:id` = "ttag123")

In the examples below, `tagN` denotes an ID of the form "tag123".

Chemicals, miscellaneous and non-English phrases appear as `term` elements:

```xml
<term xml:id="tagN">Text</term>
```

Non-standard usages (misspellings etc.) appear as `distinct` elements:

```xml
<distinct xml:id="tagN">Text</distinct>
```

Bibliographic (other work) references appear as `bibl` elements:

```xml
<bibl xml:id="tagN">Text</bibl>
```

Authors may appear as regular person references or in the form:

```xml
<author xml:id="tagN">
  <name role="aut" type="person">Text</name>
</author>
```

Personal names appear in `name` elements:

```xml
<name xml:id="tagN" type="person">Text</name>
```

Place names also appear in `name` elements:

```xml
<name xml:id="tagN" type="place">Text</name>
```

<a id="build"></a>

## Build Process

The manual build process involves several bash scripts and XSL transformations in the [ldc-content-builder](https://github.com/lulibrary/ldc-content-builder/).

Clone the ldc-content-builder repository at the same level as this repository:
```
$ ldc_path=/path/to/ldc-content
$ cd ${ldc_path}
$ git clone https://github.com/lulibrary/ldc-content-builder ldc-content-builder
$ git clone https://github.com/lulibrary/ldc-content-davy ldc-content-davy
```

Set your path to include specific projects before ldc-content-builder:
```
PATH=${ldc_path}/ldc-content-davy/bin:${ldc_path}/ldc-content-builder/bin:${PATH}
```

### Manifest

The collection manifest defines the ordering of items. It exists in two forms:
- the primary XML form: `manifest/collection`
- an LDC-specific JSON form derived from the primary XML: `manifest/json`

All changes should be made to the primary XML `manifest/collection`. Regenerate the LDC-specific JSON whenever changes to the primary copy are made.

1. Build the initial primary collection manifest `manifest/collection`. **This will overwrite any changes to `manifest/collection`!**
    ```
    $ manifest -c davy
    ```
2. Arrange `manifest/collection` in the required order
3. Rebuild the LDC-specific JSON manifest each time `manifest/collection` is edited
    ```
    $ manifest -j davy
    ```

<a id="build_anno"></a>

### Annotations

#### Auto-Tagging

1. Download auto-tagging spreadsheet from Davy Notebooks Teams site to `config/autotags/source`
2. Run `$ autotags davy`

#### Annotation Types

1. Download all spreadsheets from Davy Notebooks Teams site to `standoff/<type>/source/`>
2. Run `$ standoff davy`

<a id="build_html"></a>

### HTML


### Items

<a id="build_item_image"></a>

### Images and Manifests

1. Add images to `items/<item>/images/source`

2. Create the image source manifest in `items/<item>/manifest/image/source`  
The order and line count of the manifest must match that of the transcription manifest. Use "# image-filename" as a placeholder for transcribed pages which have no corresponding image.

3. Convert images to tiled JPEG2000
    ```
    $ image-convert davy <item>
    ```

4. Create item manifest
    ```
    $ manifest davy <item>

5. Rename images
    ```
    image-rename davy <item>
    ```

<a id="build_item_trans"></a>

### Transcriptions

<a id="build_item_tei"></a>

### TEI