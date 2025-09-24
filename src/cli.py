#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import logging
from typing import List, Optional
import json
import time

from src.pipeline import DocumentProcessor
from src.logging_filters import install_bullseye_log_filter, sanitize_bullseye_loggers
from src.exporters import HTMLExporter, MarkdownExporter, JSONLinesExporter, PDFSearchableExporter


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    try:
        Path('logs').mkdir(exist_ok=True)
        logger = logging.getLogger()
        # Prevent duplicate file handlers
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('logs/cli.log'):
                # still install filters on existing handlers
                install_bullseye_log_filter()
                sanitize_bullseye_loggers()
                return
        fh = logging.FileHandler('logs/cli.log', encoding='utf-8')
        fh.setLevel(level)
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        # Install rename filter for both console and file
        install_bullseye_log_filter()
        sanitize_bullseye_loggers()
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='DocJA - Japanese Document AI CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic OCR
  docja input.pdf -o output/
  
  # With specific format
  docja input.pdf -f md -o results/
  
  # Enable all features
  docja input.pdf --layout --table --reading-order --llm -o results/
  
  # Process directory
  docja input_dir/ -f json -o output_dir/ --recursive
        '''
    )
    
    # Input/Output
    parser.add_argument('input', type=str, help='Input file or directory')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output directory (default: results)')
    parser.add_argument('-f', '--format', type=str, default='json',
                       choices=['md', 'html', 'json', 'csv', 'pdf'],
                       help='Output format (default: json)')
    
    # Features
    parser.add_argument('--layout', action='store_true', help='Enable layout detection')
    parser.add_argument('--table', action='store_true', help='Enable table extraction')
    parser.add_argument('--reading-order', action='store_true', help='Enable reading order estimation')
    parser.add_argument('--llm', action='store_true', help='Enable LLM analysis')
    parser.add_argument('--figure', action='store_true', help='Extract figures')
    parser.add_argument('--figure_letter', action='store_true', help='Extract text inside figures')
    
    # Model selection
    parser.add_argument('--det-model', type=str, default='dbnet',
                       choices=['dbnet', 'yolo'], help='Text detection model')
    parser.add_argument('--rec-model', type=str, default='abinet',
                       choices=['abinet', 'satrn', 'none'], help='Text recognition model (use "none" to disable)')
    # Providers
    parser.add_argument('--rec-provider', type=str, default=os.getenv('DOCJA_REC_PROVIDER', 'internal'),
                       choices=['internal', 'bullseye'], help='Recognition provider: internal or bullseye')
    parser.add_argument('--rec-model-id', type=str, default=os.getenv('DOCJA_REC_MODEL_ID', ''),
                       help='HF model id when --rec-provider=bullseye')
    parser.add_argument('--det-provider', type=str, default=os.getenv('DOCJA_DET_PROVIDER', 'internal'),
                       choices=['internal', 'bullseye'], help='Text detection provider')
    parser.add_argument('--det-model-id', type=str, default=os.getenv('DOCJA_DET_MODEL_ID', ''),
                       help='HF model id when --det-provider=bullseye')
    parser.add_argument('--layout-model', type=str, default='yolo',
                       choices=['yolo', 'detr'], help='Layout detection model')
    parser.add_argument('--layout-provider', type=str, default=os.getenv('DOCJA_LAYOUT_PROVIDER', 'internal'),
                       choices=['internal', 'bullseye'], help='Layout detection provider')
    parser.add_argument('--layout-model-id', type=str, default=os.getenv('DOCJA_LAYOUT_MODEL_ID', ''),
                       help='HF model id when --layout-provider=bullseye')
    # Table provider (optional evaluation path)
    parser.add_argument('--table-provider', type=str, default=os.getenv('DOCJA_TABLE_PROVIDER', 'internal'),
                       choices=['internal', 'bullseye'], help='Table structure provider')
    parser.add_argument('--table-model-id', type=str, default=os.getenv('DOCJA_TABLE_MODEL_ID', ''),
                       help='HF/local model id hint when --table-provider=bullseye (not strictly required)')

    # LLM routing / provider
    parser.add_argument('--llm-provider', type=str, default=os.getenv('DOCJA_LLM_PROVIDER', ''),
                       choices=['', 'gptoss', 'gemma3', 'ollama', 'openai-compat'], help='LLM provider router (env override)')
    parser.add_argument('--llm-endpoint', type=str, default=os.getenv('DOCJA_LLM_ENDPOINT', ''), help='OpenAI-compatible endpoint (Gemma/OpenAI)')
    parser.add_argument('--llm-model', type=str, default=os.getenv('DOCJA_LLM_MODEL', ''), help='Model name for LLM provider')
    parser.add_argument('--llm-use-image', action='store_true', help='Attach first page image to LLM (Gemma 3)')
    
    # Processing options
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--max-pages', type=int, help='Maximum pages to process')
    parser.add_argument('--lang', type=str, default='ja', help='Primary language (default: ja)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--pdf-direct-text', action='store_true',
                        help='For PDF input, extract embedded text directly (no OCR). Useful for demos/baselines.')
    
    # Other options
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--weights', type=str, default='weights', help='Weights directory')
    parser.add_argument('--lite', action='store_true', help='Use lite models for faster processing')
    parser.add_argument('--vis', action='store_true', help='Save visualization overlays to output dir')
    # LLM options
    parser.add_argument('--llm-task', type=str, choices=['summary', 'qa', 'json'], default='summary',
                        help='LLM task type when --llm is enabled')
    parser.add_argument('--question', type=str, default=None, help='Question for --llm-task qa')
    parser.add_argument('--schema', type=str, default=None,
                        help='JSON schema string or file path for --llm-task json')
    
    # Table OCR options
    parser.add_argument('--cell-ocr-thresh', type=float, default=None,
                        help='Cell OCR confidence threshold (blank if below when --table is enabled)')
    parser.add_argument('--cell-blank-low-confidence', action='store_true',
                        help='Blank cell text when confidence is below threshold (overrides default)')
    parser.add_argument('--cell-keep-low-confidence', action='store_true',
                        help='Keep low-confidence cell text (overrides default)')
    
    return parser.parse_args()


def get_input_files(input_path: str, recursive: bool = False) -> List[Path]:
    input_path = Path(input_path)
    
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        patterns = ['*.pdf', '*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
        files = []
        for pattern in patterns:
            if recursive:
                files.extend(input_path.rglob(pattern))
            else:
                files.extend(input_path.glob(pattern))
        return sorted(files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def get_exporter(format: str):
    exporters = {
        'md': MarkdownExporter(),
        'html': HTMLExporter(),
        'json': JSONLinesExporter(),
        'csv': JSONLinesExporter(),  # CSV is handled by JSONLines with different extension
        'pdf': PDFSearchableExporter()
    }
    return exporters.get(format)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Get input files
    try:
        input_files = get_input_files(args.input, args.recursive)
        if not input_files:
            logger.error("No input files found")
            return 1
        logger.info(f"Found {len(input_files)} files to process")
    except Exception as e:
        logger.error(f"Error finding input files: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply LLM router env overrides (optional)
    if args.llm_provider:
        os.environ['DOCJA_LLM_PROVIDER'] = args.llm_provider
    if args.llm_endpoint:
        os.environ['DOCJA_LLM_ENDPOINT'] = args.llm_endpoint
    if args.llm_model:
        os.environ['DOCJA_LLM_MODEL'] = args.llm_model
    if args.llm_use_image:
        os.environ['DOCJA_LLM_USE_IMAGE'] = '1'

    # Initialize processor
    try:
        # Normalize provider aliases
        def _norm_provider(name: str) -> str:
            return str(name).lower()
        # When any normalized provider is bullseye, align metadata label alias
        if any(_norm_provider(p) == 'bullseye' for p in [args.rec_provider, args.det_provider, args.layout_provider, args.table_provider]):
            os.environ['DOCJA_PROVIDER_ALIAS_LABEL'] = 'bullseye'

        processor = DocumentProcessor(
            det_model=args.det_model,
            rec_model=args.rec_model,
            rec_provider=_norm_provider(args.rec_provider),
            rec_hf_model_id=(args.rec_model_id or None),
            det_provider=_norm_provider(args.det_provider),
            det_hf_model_id=(args.det_model_id or None),
            layout_model=args.layout_model if args.layout else None,
            layout_provider=_norm_provider(args.layout_provider),
            layout_hf_model_id=(args.layout_model_id or None),
            enable_table=args.table,
            table_provider=_norm_provider(args.table_provider),
            table_hf_model_id=(args.table_model_id or None),
            enable_reading_order=args.reading_order,
            enable_llm=args.llm,
            device=args.device,
            weights_dir=args.weights,
            lite_mode=args.lite,
            table_ocr_conf_threshold=args.cell_ocr_thresh,
            table_blank_low_confidence=(True if args.cell_blank_low_confidence else (False if args.cell_keep_low_confidence else None))
        )
        logger.info("Document processor initialized")
    except Exception as e:
        logger.error(f"Error initializing processor: {e}")
        return 1
    
    # Get exporter
    exporter = get_exporter(args.format)
    if not exporter:
        logger.error(f"Unknown format: {args.format}")
        return 1
    
    # Process files
    total_files = len(input_files)
    success_count = 0
    
    for i, input_file in enumerate(input_files, 1):
        logger.info(f"Processing [{i}/{total_files}]: {input_file}")
        
        try:
            # Process document
            start_time = time.time()
            vis_dir = None
            if args.vis:
                vis_dir = str(output_dir / "vis")
            # Parse schema if provided
            schema_obj = None
            if args.schema:
                sp = Path(args.schema)
                try:
                    if sp.exists():
                        schema_obj = json.loads(sp.read_text(encoding='utf-8'))
                    else:
                        schema_obj = json.loads(args.schema)
                except Exception:
                    schema_obj = None
            result = processor.process(
                str(input_file),
                max_pages=args.max_pages,
                extract_figures=args.figure,
                vis_save_dir=vis_dir,
                llm_task=(args.llm_task if args.llm else None),
                llm_question=args.question,
                llm_schema=schema_obj,
                extract_figure_text=args.figure_letter,
                pdf_direct_text=args.pdf_direct_text
            )
            process_time = time.time() - start_time
            
            # Export result
            output_name = input_file.stem
            if args.format == 'csv':
                output_file = output_dir / f"{output_name}.csv"
            else:
                output_file = output_dir / f"{output_name}.{args.format}"
            
            exporter.export(result, str(output_file))

            # For non-JSON formats, also emit a sidecar metadata JSON for jq-friendly access
            try:
                if args.format not in ('json', 'csv'):
                    meta = {
                        'filename': result.filename,
                        'metadata': result.metadata
                    }
                    meta_path = output_file.with_suffix(output_file.suffix + '.meta.json')
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception as _e:
                logger.warning(f"Failed to write sidecar metadata JSON: {_e}")
            
            # Log statistics
            total_blocks = sum(len(page.text_blocks) for page in result.pages)
            logger.info(
                f"Completed: {len(result.pages)} pages, "
                f"{total_blocks} text blocks, "
                f"Time: {process_time:.2f}s"
            )
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Summary
    logger.info(f"\nProcessing complete: {success_count}/{total_files} files succeeded")
    
    return 0 if success_count == total_files else 1


if __name__ == '__main__':
    sys.exit(main())
