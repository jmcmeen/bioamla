
## dataset_uploader.py

**Dataset Loading:**
- Public and private dataset support
- Organization dataset support
- Authentication handling with tokens
- Multiple configuration and split options
- Streaming mode support
- Revision/branch selection

**Information & Inspection:**
- Dataset metadata retrieval
- Summary statistics display
- Example data preview
- Column and split information

**Export Options:**
- JSON, CSV, and Parquet export formats
- Automatic split handling
- Directory structure creation

## Installation Requirements:
```bash
pip install datasets huggingface_hub
```

## Usage Examples:

**Load a public dataset:**
```bash
python hf_dataset_loader.py glue --config cola
```

**Load a private individual dataset:**
```bash
python hf_dataset_loader.py stevhliu/demo --private --token
```

**Load a private organization dataset:**
```bash
python hf_dataset_loader.py organization/dataset_name --private --token
```

**Load specific split with examples:**
```bash
python hf_dataset_loader.py imdb --split train --examples 3
```

**Get dataset info only:**
```bash
python hf_dataset_loader.py wikitext --info-only --save-info dataset_info.json
```

**Load and export dataset:**
```bash
python hf_dataset_loader.py squad --export-dir ./squad_data --export-format json
```

**Load with streaming mode:**
```bash
python hf_dataset_loader.py openwebtext --streaming --examples 1
```

**Load specific configuration and revision:**
```bash
python hf_dataset_loader.py glue --config mrpc --revision main --cache-dir ./cache
```

## Authentication:

The script handles authentication in several ways:
1. **Automatic**: Uses existing Hugging Face CLI login
2. **Token flag**: Use `--token` for private datasets
3. **Private flag**: Use `--private` to indicate private dataset
4. **No auth**: Use `--no-auth` for public datasets only

## Sample Output:

```
Fetching information for dataset: imdb
✅ Already authenticated as: username

============================================================
DATASET INFORMATION
============================================================
Dataset ID: imdb
Author: None
Private: False
Downloads: 500,000
Likes: 1,200
License: Apache-2.0
Tags: sentiment-classification, text-classification
Languages: en
Created: 2020-05-02 15:30:45
Last Modified: 2023-10-15 09:22:33

Loading dataset: imdb
  Split: train

============================================================
DATASET SUMMARY
============================================================
Dataset type: Dataset
Number of rows: 25,000
Number of columns: 2
Column names: ['text', 'label']
First 2 example(s):
  Example 1: {'text': 'This movie is amazing...', 'label': 1}
  Example 2: {'text': 'Terrible film, would not...', 'label': 0}

✅ Dataset loading completed successfully!
```

## Command-Line Options:

**Required:**
- `dataset_name`: Dataset identifier (e.g., 'glue', 'stevhliu/demo')

**Optional:**
- `--config`: Dataset configuration name
- `--split`: Specific split to load
- `--token`: Use authentication token
- `--private`: Indicate private dataset
- `--streaming`: Enable streaming mode
- `--cache-dir`: Custom cache directory
- `--revision`: Specific revision/branch
- `--trust-remote-code`: Trust remote code execution
- `--examples N`: Show N example entries
- `--info-only`: Only fetch metadata
- `--save-info FILE`: Save metadata to JSON
- `--export-dir DIR`: Export dataset to directory
- `--export-format FORMAT`: Export format (json/csv/parquet)
- `--no-auth`: Disable authentication

The script provides comprehensive error handling, clear progress indicators, and detailed help information to make working with Hugging Face datasets straightforward from the command line.