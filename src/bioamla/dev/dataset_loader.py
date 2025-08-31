#!/usr/bin/env python3
"""
Hugging Face Dataset Loader CLI
Command-line wrapper for loading datasets from Hugging Face Hub
with support for public/private datasets, organizations, and various configurations.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from datasets import load_dataset, DatasetDict, Dataset
    from huggingface_hub import HfApi, login
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Install with: pip install datasets huggingface_hub")
    print(f"Missing package: {str(e)}")
    sys.exit(1)

class HuggingFaceDatasetLoader:
    def __init__(self):
        self.api = HfApi()
        self.dataset = None
        
    def authenticate(self, token: Optional[str] = None, use_auth_token: bool = True):
        """
        Authenticate with Hugging Face Hub.
        
        Args:
            token (str, optional): Explicit token to use
            use_auth_token (bool): Whether to use authentication
        """
        if not use_auth_token:
            return True
        
        try:
            if token:
                login(token=token)
                print("✅ Successfully authenticated with provided token")
            else:
                # Try to use existing token or prompt for login
                whoami_info = self.api.whoami()
                if whoami_info:
                    print(f"✅ Already authenticated as: {whoami_info['name']}")
                else:
                    print("🔐 Authentication required. Please run 'huggingface-cli login' first")
                    return False
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {str(e)}")
            return False
    
    def load_dataset_info(self, dataset_name: str, use_auth_token: bool = True):
        """
        Get information about a dataset without loading it.
        
        Args:
            dataset_name (str): Name of the dataset
            use_auth_token (bool): Whether to use authentication
            
        Returns:
            dict: Dataset information or None if error
        """
        try:
            token = True if use_auth_token else None
            dataset_info = self.api.dataset_info(dataset_name, token=token)
            return {
                'id': dataset_info.id,
                'author': dataset_info.author,
                'downloads': dataset_info.downloads,
                'likes': dataset_info.likes,
                'tags': dataset_info.tags,
                'task_categories': dataset_info.task_categories,
                'size_categories': dataset_info.size_categories,
                'language': dataset_info.language,
                'license': getattr(dataset_info, 'license', 'Unknown'),
                'private': dataset_info.private,
                'created_at': str(dataset_info.created_at) if dataset_info.created_at else None,
                'last_modified': str(dataset_info.last_modified) if dataset_info.last_modified else None
            }
        except Exception as e:
            print(f"Warning: Could not fetch dataset info: {str(e)}")
            return None
    
    def load_dataset_with_config(self, 
                                dataset_name: str,
                                config_name: Optional[str] = None,
                                split: Optional[str] = None,
                                cache_dir: Optional[str] = None,
                                use_auth_token: bool = True,
                                streaming: bool = False,
                                revision: Optional[str] = None,
                                trust_remote_code: bool = False,
                                **kwargs):
        """
        Load a dataset with specified configuration.
        
        Args:
            dataset_name (str): Name of the dataset
            config_name (str, optional): Dataset configuration name
            split (str, optional): Which split to load
            cache_dir (str, optional): Directory to cache the dataset
            use_auth_token (bool): Whether to use authentication token
            streaming (bool): Whether to load in streaming mode
            revision (str, optional): Specific revision/branch to load
            trust_remote_code (bool): Whether to trust remote code
            **kwargs: Additional arguments to pass to load_dataset
            
        Returns:
            Dataset or DatasetDict: Loaded dataset
        """
        try:
            # Prepare arguments
            load_args = {
                'path': dataset_name,
                'token': True if use_auth_token else None,
                'streaming': streaming,
                'trust_remote_code': trust_remote_code
            }
            
            # Add optional arguments
            if config_name:
                load_args['name'] = config_name
            if split:
                load_args['split'] = split
            if cache_dir:
                load_args['cache_dir'] = cache_dir
            if revision:
                load_args['revision'] = revision
            
            # Add any additional kwargs
            load_args.update(kwargs)
            
            print(f"Loading dataset: {dataset_name}")
            if config_name:
                print(f"  Configuration: {config_name}")
            if split:
                print(f"  Split: {split}")
            if streaming:
                print("  Streaming mode: enabled")
            
            # Load the dataset
            self.dataset = load_dataset(**load_args)
            
            return self.dataset
            
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                print("❌ Authentication required. The dataset might be private.")
                print("   Try: huggingface-cli login")
            elif e.response.status_code == 404:
                print("❌ Dataset not found. Check the dataset name and spelling.")
            else:
                print(f"❌ HTTP Error {e.response.status_code}: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return None
    
    def print_dataset_summary(self, dataset, show_examples: int = 0):
        """
        Print a summary of the loaded dataset.
        
        Args:
            dataset: The loaded dataset
            show_examples (int): Number of examples to show
        """
        if dataset is None:
            print("No dataset loaded.")
            return
        
        print(f"\n{'='*60}")
        print("DATASET SUMMARY")
        print(f"{'='*60}")
        
        if isinstance(dataset, DatasetDict):
            print("Dataset type: DatasetDict")
            print(f"Available splits: {list(dataset.keys())}")
            
            for split_name, split_dataset in dataset.items():
                print(f"\nSplit '{split_name}':")
                print(f"  Number of rows: {len(split_dataset):,}")
                print(f"  Number of columns: {len(split_dataset.column_names)}")
                print(f"  Column names: {split_dataset.column_names}")
                
                if show_examples > 0 and len(split_dataset) > 0:
                    print(f"  First {min(show_examples, len(split_dataset))} example(s):")
                    for i in range(min(show_examples, len(split_dataset))):
                        print(f"    Example {i+1}: {split_dataset[i]}")
        
        elif isinstance(dataset, Dataset):
            print("Dataset type: Dataset")
            print(f"Number of rows: {len(dataset):,}")
            print(f"Number of columns: {len(dataset.column_names)}")
            print(f"Column names: {dataset.column_names}")
            
            if show_examples > 0 and len(dataset) > 0:
                print(f"First {min(show_examples, len(dataset))} example(s):")
                for i in range(min(show_examples, len(dataset))):
                    print(f"  Example {i+1}: {dataset[i]}")
        
        else:
            print(f"Dataset type: {type(dataset)}")
            print(f"Dataset: {dataset}")
    
    def save_dataset_info(self, dataset_info: Dict[Any, Any], output_file: str):
        """
        Save dataset information to a JSON file.
        
        Args:
            dataset_info (dict): Dataset information
            output_file (str): Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            print(f"✅ Dataset info saved to: {output_file}")
        except Exception as e:
            print(f"❌ Error saving dataset info: {str(e)}")
    
    def export_dataset(self, dataset, output_dir: str, format_type: str = "json"):
        """
        Export dataset to files.
        
        Args:
            dataset: The dataset to export
            output_dir (str): Output directory
            format_type (str): Export format (json, csv, parquet)
        """
        if dataset is None:
            print("No dataset to export.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(dataset, DatasetDict):
                for split_name, split_dataset in dataset.items():
                    filename = f"{split_name}.{format_type}"
                    filepath = output_path / filename
                    
                    if format_type.lower() == "json":
                        split_dataset.to_json(str(filepath))
                    elif format_type.lower() == "csv":
                        split_dataset.to_csv(str(filepath))
                    elif format_type.lower() == "parquet":
                        split_dataset.to_parquet(str(filepath))
                    
                    print(f"✅ Exported {split_name} split to: {filepath}")
            
            elif isinstance(dataset, Dataset):
                filename = f"dataset.{format_type}"
                filepath = output_path / filename
                
                if format_type.lower() == "json":
                    dataset.to_json(str(filepath))
                elif format_type.lower() == "csv":
                    dataset.to_csv(str(filepath))
                elif format_type.lower() == "parquet":
                    dataset.to_parquet(str(filepath))
                
                print(f"✅ Exported dataset to: {filepath}")
        
        except Exception as e:
            print(f"❌ Error exporting dataset: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect Hugging Face datasets from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load a public dataset
  python hf_dataset_loader.py glue --config cola
  
  # Load a private dataset with authentication
  python hf_dataset_loader.py stevhliu/demo --private --token
  
  # Load organization dataset
  python hf_dataset_loader.py organization/dataset_name --private --token
  
  # Load specific split and show examples
  python hf_dataset_loader.py imdb --split train --examples 2
  
  # Load and export dataset
  python hf_dataset_loader.py squad --export-dir ./squad_data --export-format json
  
  # Get dataset info only
  python hf_dataset_loader.py wikitext --info-only --save-info dataset_info.json
        """
    )
    
    parser.add_argument(
        "dataset_name",
        help="Name of the dataset (e.g., 'glue', 'stevhliu/demo', 'organization/dataset_name')"
    )
    
    parser.add_argument(
        "--config", "--name",
        help="Dataset configuration name"
    )
    
    parser.add_argument(
        "--split",
        help="Specific split to load (e.g., 'train', 'test', 'validation')"
    )
    
    parser.add_argument(
        "--token",
        action="store_true",
        help="Use authentication token (required for private datasets)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Indicate this is a private dataset (enables token usage)"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Load dataset in streaming mode"
    )
    
    parser.add_argument(
        "--cache-dir",
        help="Directory to cache the dataset"
    )
    
    parser.add_argument(
        "--revision",
        help="Specific revision/branch to load"
    )
    
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code (use with caution)"
    )
    
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Number of examples to display (default: 0)"
    )
    
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only fetch and display dataset information without loading data"
    )
    
    parser.add_argument(
        "--save-info",
        help="Save dataset information to JSON file"
    )
    
    parser.add_argument(
        "--export-dir",
        help="Directory to export the dataset"
    )
    
    parser.add_argument(
        "--export-format",
        choices=["json", "csv", "parquet"],
        default="json",
        help="Export format (default: json)"
    )
    
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable authentication (for public datasets only)"
    )
    
    args = parser.parse_args()
    
    # Initialize loader
    loader = HuggingFaceDatasetLoader()
    
    # Determine if authentication should be used
    use_auth = not args.no_auth and (args.token or args.private)
    
    # Authenticate if needed
    if use_auth:
        if not loader.authenticate(use_auth_token=True):
            print("Authentication failed. Exiting.")
            sys.exit(1)
    
    # Get dataset info
    print(f"Fetching information for dataset: {args.dataset_name}")
    dataset_info = loader.load_dataset_info(args.dataset_name, use_auth_token=use_auth)
    
    if dataset_info:
        print(f"\n{'='*60}")
        print("DATASET INFORMATION")
        print(f"{'='*60}")
        print(f"Dataset ID: {dataset_info['id']}")
        print(f"Author: {dataset_info['author']}")
        print(f"Private: {dataset_info['private']}")
        print(f"Downloads: {dataset_info['downloads']:,}")
        print(f"Likes: {dataset_info['likes']:,}")
        print(f"License: {dataset_info['license']}")
        if dataset_info['tags']:
            print(f"Tags: {', '.join(dataset_info['tags'])}")
        if dataset_info['language']:
            print(f"Languages: {', '.join(dataset_info['language'])}")
        print(f"Created: {dataset_info['created_at']}")
        print(f"Last Modified: {dataset_info['last_modified']}")
        
        # Save dataset info if requested
        if args.save_info:
            loader.save_dataset_info(dataset_info, args.save_info)
    
    # If info-only mode, exit here
    if args.info_only:
        print("\nInfo-only mode. Exiting without loading dataset.")
        return
    
    # Load the dataset
    dataset = loader.load_dataset_with_config(
        dataset_name=args.dataset_name,
        config_name=args.config,
        split=args.split,
        cache_dir=args.cache_dir,
        use_auth_token=use_auth,
        streaming=args.streaming,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code
    )
    
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        sys.exit(1)
    
    # Print dataset summary
    loader.print_dataset_summary(dataset, show_examples=args.examples)
    
    # Export dataset if requested
    if args.export_dir:
        print(f"\nExporting dataset to: {args.export_dir}")
        loader.export_dataset(dataset, args.export_dir, args.export_format)
    
    print("\n✅ Dataset loading completed successfully!")

if __name__ == "__main__":
    main()