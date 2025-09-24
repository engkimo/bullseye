"""
Dataset registry for managing dataset paths and download hooks.
Users must register actual paths or provide download functions.
"""

import os
from typing import Dict, Callable, Optional, Any
from pathlib import Path
import logging
import json


logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Registry for dataset paths and download hooks."""
    
    def __init__(self):
        self._paths: Dict[str, Path] = {}
        self._download_hooks: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load from environment variables
        self._load_from_env()
        
        # Register default metadata
        self._register_default_metadata()
    
    def register_path(self, dataset_name: str, path: str):
        """Register a dataset path."""
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning(f"Dataset path does not exist: {path}")
        
        self._paths[dataset_name] = path_obj
        logger.info(f"Registered dataset '{dataset_name}' at {path}")
    
    def register_download_hook(self, dataset_name: str, download_fn: Callable[[str], None]):
        """Register a download function for a dataset."""
        self._download_hooks[dataset_name] = download_fn
        logger.info(f"Registered download hook for '{dataset_name}'")
    
    def get_path(self, dataset_name: str) -> Path:
        """Get dataset path, downloading if necessary."""
        
        # Check if path is registered
        if dataset_name in self._paths:
            return self._paths[dataset_name]
        
        # Check environment variable
        env_var = f"DATASET_{dataset_name.upper()}_PATH"
        if env_var in os.environ:
            path = Path(os.environ[env_var])
            self.register_path(dataset_name, str(path))
            return path
        
        # Try to download
        if dataset_name in self._download_hooks:
            logger.info(f"Dataset '{dataset_name}' not found, attempting download...")
            
            # Default download location
            download_dir = Path("data/datasets") / dataset_name
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Call download hook
            self._download_hooks[dataset_name](str(download_dir))
            
            # Register the path
            self.register_path(dataset_name, str(download_dir))
            return download_dir
        
        # Dataset not found
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Please either:\n"
            f"1. Set environment variable {env_var}\n"
            f"2. Call registry.register_path('{dataset_name}', '/path/to/data')\n"
            f"3. Register a download hook"
        )
    
    def list_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all registered datasets."""
        datasets = {}
        
        for name in set(list(self._paths.keys()) + list(self._download_hooks.keys())):
            datasets[name] = {
                'registered': name in self._paths,
                'path': str(self._paths.get(name, '')),
                'has_download_hook': name in self._download_hooks,
                'metadata': self._metadata.get(name, {})
            }
        
        return datasets
    
    def _load_from_env(self):
        """Load dataset paths from environment variables."""
        for key, value in os.environ.items():
            if key.startswith('DATASET_') and key.endswith('_PATH'):
                dataset_name = key[8:-5].lower()  # Remove DATASET_ and _PATH
                self.register_path(dataset_name, value)
    
    def _register_default_metadata(self):
        """Register metadata for known datasets."""
        
        self._metadata.update({
            'synthdog_ja': {
                'description': 'Synthetic document images for Japanese text detection/recognition',
                'tasks': ['text_detection', 'text_recognition'],
                'format': 'images + COCO annotations',
                'license': 'Check original SynthDoG license',
                'size': '~10GB'
            },
            'doclaynet': {
                'description': 'Document layout analysis dataset',
                'tasks': ['layout_detection'],
                'format': 'COCO format',
                'license': 'CC BY 4.0',
                'size': '~30GB'
            },
            'pubtabnet': {
                'description': 'Table detection and structure recognition',
                'tasks': ['table_detection', 'table_structure'],
                'format': 'Custom JSON',
                'license': 'CC BY-NC-SA 4.0',
                'size': '~90GB'
            },
            'jsquad': {
                'description': 'Japanese question answering dataset',
                'tasks': ['question_answering'],
                'format': 'SQuAD format',
                'license': 'CC BY-SA 4.0',
                'size': '~50MB'
            },
            'jaquad': {
                'description': 'Japanese question answering dataset (alternative)',
                'tasks': ['question_answering'],
                'format': 'SQuAD format',
                'license': 'CC BY-SA 3.0',
                'size': '~30MB'
            },
            'jdocqa': {
                'description': 'Japanese document QA dataset',
                'tasks': ['document_qa'],
                'format': 'JSONL',
                'license': 'Research use only',
                'size': '~200MB'
            }
        })


# Global registry instance
registry = DatasetRegistry()


# Example download hooks (users must implement actual download logic)
def _synthdog_ja_download_hook(download_dir: str):
    """Download hook for SynthDoG-JA dataset."""
    instructions = """
    SynthDoG-JA Download Instructions:
    
    1. Visit the SynthDoG repository
    2. Follow instructions to generate Japanese synthetic data
    3. Place the generated data in: {download_dir}
    
    Expected structure:
    {download_dir}/
    ├── images/
    │   ├── train/
    │   └── val/
    └── annotations/
        ├── train.json
        └── val.json
    """.format(download_dir=download_dir)
    
    logger.info(instructions)
    raise NotImplementedError(
        "Automatic download not implemented. Please follow the instructions above."
    )


def _doclaynet_download_hook(download_dir: str):
    """Download hook for DocLayNet dataset."""
    instructions = """
    DocLayNet Download Instructions:
    
    1. Visit https://github.com/DS4SD/DocLayNet
    2. Follow the download instructions
    3. Extract the dataset to: {download_dir}
    
    Expected structure:
    {download_dir}/
    ├── COCO/
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    └── PNG/
        ├── train/
        ├── val/
        └── test/
    """.format(download_dir=download_dir)
    
    logger.info(instructions)
    raise NotImplementedError(
        "Automatic download not implemented. Please follow the instructions above."
    )


def _jsquad_download_hook(download_dir: str):
    """Download hook for JSQuAD dataset."""
    # This could potentially use Hugging Face datasets
    instructions = """
    JSQuAD Download Instructions:
    
    Option 1 - Using Hugging Face:
    ```python
    from datasets import load_dataset
    dataset = load_dataset("shunk031/JGLUE", "JSQuAD")
    dataset.save_to_disk("{download_dir}")
    ```
    
    Option 2 - Manual download:
    1. Visit the JGLUE repository
    2. Download JSQuAD data
    3. Place in: {download_dir}
    """.format(download_dir=download_dir)
    
    logger.info(instructions)
    
    # Try to use HF datasets if available
    try:
        from datasets import load_dataset
        logger.info("Attempting to download JSQuAD via Hugging Face...")
        dataset = load_dataset("shunk031/JGLUE", "JSQuAD")
        dataset.save_to_disk(download_dir)
        logger.info(f"Successfully downloaded JSQuAD to {download_dir}")
    except Exception as e:
        logger.error(f"Failed to download via HF: {e}")
        raise NotImplementedError(
            "Automatic download failed. Please follow the manual instructions above."
        )


# Register default download hooks
registry.register_download_hook('synthdog_ja', _synthdog_ja_download_hook)
registry.register_download_hook('doclaynet', _doclaynet_download_hook)
registry.register_download_hook('jsquad', _jsquad_download_hook)


# Utility functions
def register_dataset(name: str, path: str):
    """Convenience function to register a dataset."""
    registry.register_path(name, path)


def get_dataset_path(name: str) -> Path:
    """Convenience function to get dataset path."""
    return registry.get_path(name)


def list_available_datasets():
    """List all available datasets."""
    datasets = registry.list_datasets()
    
    print("Available Datasets:")
    print("-" * 80)
    
    for name, info in datasets.items():
        print(f"\n{name}:")
        print(f"  Registered: {info['registered']}")
        if info['path']:
            print(f"  Path: {info['path']}")
        print(f"  Download available: {info['has_download_hook']}")
        
        if info['metadata']:
            print(f"  Description: {info['metadata'].get('description', 'N/A')}")
            print(f"  License: {info['metadata'].get('license', 'N/A')}")
            print(f"  Size: {info['metadata'].get('size', 'N/A')}")


if __name__ == "__main__":
    # Show available datasets when run directly
    list_available_datasets()