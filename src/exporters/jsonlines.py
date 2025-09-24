import json
from typing import List, Dict, Any
from pathlib import Path
import csv


class JSONLinesExporter:
    """Export document processing results to JSON Lines or CSV format."""
    
    def __init__(self, 
                 include_bbox: bool = True,
                 include_confidence: bool = True,
                 flatten_structure: bool = False):
        self.include_bbox = include_bbox
        self.include_confidence = include_confidence
        self.flatten_structure = flatten_structure
    
    def export(self, result: Any, output_path: str):
        """Export document result to JSON Lines or CSV file."""
        output_path = Path(output_path)
        
        if output_path.suffix.lower() == '.csv':
            self._export_csv(result, output_path)
        else:
            self._export_jsonl(result, output_path)
    
    def _export_jsonl(self, result: Any, output_path: Path):
        """Export to JSON Lines format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Document-level metadata
            doc_meta = {
                'type': 'document',
                'filename': result.filename,
                'num_pages': len(result.pages),
                'metadata': result.metadata
            }
            f.write(json.dumps(doc_meta, ensure_ascii=False) + '\n')
            
            # Process each page
            for page in result.pages:
                # Page-level metadata
                page_data = {
                    'type': 'page',
                    'page_num': page.page_num,
                    'width': page.width,
                    'height': page.height
                }
                
                if not self.flatten_structure:
                    # Include full structure
                    page_data['text_blocks'] = [
                        self._serialize_text_block(block) 
                        for block in page.text_blocks
                    ]
                    page_data['tables'] = [
                        self._serialize_table(table)
                        for table in page.tables
                    ]
                    page_data['reading_order'] = page.reading_order
                    # Flow/Gantt structures when available
                    try:
                        page_data['graphs'] = getattr(page, 'graphs', [])
                        page_data['charts'] = getattr(page, 'charts', [])
                    except Exception:
                        pass
                    f.write(json.dumps(page_data, ensure_ascii=False) + '\n')
                else:
                    # Flatten structure - one line per element
                    f.write(json.dumps(page_data, ensure_ascii=False) + '\n')
                    
                    # Text blocks
                    for idx, block in enumerate(page.text_blocks):
                        block_data = {
                            'type': 'text_block',
                            'page_num': page.page_num,
                            'block_idx': idx,
                            **self._serialize_text_block(block)
                        }
                        f.write(json.dumps(block_data, ensure_ascii=False) + '\n')
                    
                    # Tables
                    for idx, table in enumerate(page.tables):
                        table_data = {
                            'type': 'table',
                            'page_num': page.page_num,
                            'table_idx': idx,
                            **self._serialize_table(table)
                        }
                        f.write(json.dumps(table_data, ensure_ascii=False) + '\n')
    
    def _export_csv(self, result: Any, output_path: Path):
        """Export to CSV format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        
        # Process each page
        for page in result.pages:
            # Text blocks
            for idx, block in enumerate(page.text_blocks):
                row = {
                    'filename': result.filename,
                    'page_num': page.page_num,
                    'element_type': 'text',
                    'element_idx': idx,
                    'block_type': block.block_type,
                    'text': block.text.replace('\n', ' '),
                }
                
                if self.include_bbox:
                    row.update({
                        'x1': block.bbox[0],
                        'y1': block.bbox[1],
                        'x2': block.bbox[2],
                        'y2': block.bbox[3]
                    })
                
                if self.include_confidence:
                    row['confidence'] = block.confidence
                
                rows.append(row)
            
            # Tables
            for idx, table in enumerate(page.tables):
                # Table summary row
                row = {
                    'filename': result.filename,
                    'page_num': page.page_num,
                    'element_type': 'table',
                    'element_idx': idx,
                    'block_type': 'table',
                    'text': f"Table with {len(table.cells)} cells",
                }
                
                if self.include_bbox:
                    row.update({
                        'x1': table.bbox[0],
                        'y1': table.bbox[1],
                        'x2': table.bbox[2],
                        'y2': table.bbox[3]
                    })
                
                rows.append(row)
                
                # Individual cells
                for cell in table.cells:
                    cell_row = {
                        'filename': result.filename,
                        'page_num': page.page_num,
                        'element_type': 'table_cell',
                        'element_idx': f"{idx}_{cell['row']}_{cell['col']}",
                        'block_type': 'table_header' if cell.get('is_header') else 'table_cell',
                        'text': cell['text'],
                    }
                    
                    if self.include_bbox and 'bbox' in cell:
                        cell_row.update({
                            'x1': cell['bbox'][0],
                            'y1': cell['bbox'][1],
                            'x2': cell['bbox'][2],
                            'y2': cell['bbox'][3]
                        })
                    
                    rows.append(cell_row)
        
        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    
    def _serialize_text_block(self, block: Any) -> Dict[str, Any]:
        """Serialize a text block to dictionary."""
        data = {
            'text': block.text,
            'block_type': block.block_type
        }
        
        if self.include_bbox:
            data['bbox'] = block.bbox
        
        if self.include_confidence:
            data['confidence'] = block.confidence
        
        if block.metadata:
            data['metadata'] = block.metadata
        
        return data
    
    def _serialize_table(self, table: Any) -> Dict[str, Any]:
        """Serialize a table to dictionary."""
        data = {
            'cells': table.cells,
            'num_cells': len(table.cells)
        }
        
        if self.include_bbox:
            data['bbox'] = table.bbox
        
        # Include structure info
        if table.cells:
            max_row = max(cell['row'] for cell in table.cells)
            max_col = max(cell['col'] for cell in table.cells)
            data['dimensions'] = {
                'rows': max_row + 1,
                'cols': max_col + 1
            }
        
        return data
