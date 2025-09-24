from typing import List, Dict, Any
from pathlib import Path
import html as html_lib
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import json


class HTMLExporter:
    """Export document processing results to HTML format."""
    
    def __init__(self, include_css: bool = True, include_images: bool = True):
        self.include_css = include_css
        self.include_images = include_images
    
    def export(self, result: Any, output_path: str):
        """Export document result to HTML file."""
        html_content = self._generate_html(result)
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html(self, result: Any) -> str:
        """Generate HTML content from document result."""
        parts = []
        
        # HTML header
        parts.append('<!DOCTYPE html>')
        parts.append('<html lang="ja">')
        parts.append('<head>')
        parts.append('<meta charset="UTF-8">')
        parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        parts.append(f'<title>{Path(result.filename).stem}</title>')
        
        # CSS
        if self.include_css:
            parts.append('<style>')
            parts.append(self._get_css())
            parts.append('</style>')
        
        parts.append('</head>')
        parts.append('<body>')
        
        # Document info
        parts.append('<div class="document-info">')
        parts.append(f'<h1>{Path(result.filename).name}</h1>')
        parts.append(f'<p>Pages: {len(result.pages)}</p>')
        parts.append('</div>')

        # LLM results (if any)
        try:
            llm = (result.metadata or {}).get('llm')
            if llm:
                parts.append('<section class="llm-results">')
                parts.append('<h2>LLM Results</h2>')
                # Summary
                if isinstance(llm.get('summary'), str) and llm.get('summary'):
                    parts.append('<div class="llm-summary">')
                    parts.append('<h3>Summary</h3>')
                    parts.append(f'<p>{html_lib.escape(llm["summary"])}</p>')
                    parts.append('</div>')
                # QA
                qa = llm.get('qa') if isinstance(llm.get('qa'), dict) else None
                if qa:
                    q = html_lib.escape(str(qa.get('question', '')))
                    a = html_lib.escape('' if qa.get('answer') is None else str(qa.get('answer')))
                    parts.append('<div class="llm-qa">')
                    parts.append('<h3>Q&A</h3>')
                    parts.append(f'<p><strong>Q:</strong> {q}</p>')
                    parts.append(f'<p><strong>A:</strong> {a}</p>')
                    parts.append('</div>')
                # JSON extraction
                ext = llm.get('extraction') if isinstance(llm.get('extraction'), dict) else None
                if ext:
                    js = ext.get('result')
                    try:
                        js_pretty = json.dumps(json.loads(js) if isinstance(js, str) else js, ensure_ascii=False, indent=2)
                    except Exception:
                        js_pretty = str(js)
                    parts.append('<div class="llm-json">')
                    parts.append('<h3>JSON Extraction</h3>')
                    parts.append('<pre><code>')
                    parts.append(html_lib.escape(js_pretty))
                    parts.append('</code></pre>')
                    parts.append('</div>')
                parts.append('</section>')
        except Exception:
            pass
        
        # Process each page
        for page in result.pages:
            parts.append(f'<div class="page" id="page-{page.page_num}">')
            parts.append(f'<h2>Page {page.page_num}</h2>')
            
            # Add text blocks in reading order
            if page.reading_order:
                ordered_blocks = [page.text_blocks[i] for i in page.reading_order]
            else:
                ordered_blocks = page.text_blocks
            
            for block in ordered_blocks:
                parts.append(self._render_text_block(block))
            
            # Add tables
            for table in page.tables:
                parts.append(self._render_table(table))
            
            # Add figures if any
            if self.include_images and page.figures:
                parts.append('<div class="figures">')
                for fig in page.figures:
                    parts.append(self._render_figure(fig))
                parts.append('</div>')
            
            parts.append('</div>')  # Close page div
        
        parts.append('</body>')
        parts.append('</html>')
        
        return '\n'.join(parts)
    
    def _get_css(self) -> str:
        """Get CSS styles."""
        return '''
        body {
            font-family: "Noto Sans JP", sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .document-info {
            background-color: #fff;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .page {
            background-color: #fff;
            padding: 40px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .llm-results {
            background-color: #fff;
            padding: 20px;
            margin-bottom: 30px;
            border-left: 4px solid #4a90e2;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        }
        .llm-results h3 { margin-top: 10px; }
        .llm-results pre { background: #f0f0f0; padding: 10px; overflow-x: auto; }
        
        .text-block {
            margin-bottom: 15px;
        }
        
        .text-block.title {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 20px;
        }
        
        .text-block.section_heading {
            font-size: 1.3em;
            font-weight: bold;
            margin-top: 15px;
        }
        
        .text-block.caption {
            font-style: italic;
            color: #666;
            font-size: 0.9em;
        }
        
        .text-block.footnote {
            font-size: 0.8em;
            color: #666;
            border-left: 2px solid #ddd;
            padding-left: 10px;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        
        table th, table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        table th {
            background-color: #f0f0f0;
            font-weight: bold;
        }
        
        .figure {
            margin: 20px 0;
            text-align: center;
        }
        
        .figure img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }
        
        .confidence-low {
            color: #999;
        }

        .metadata {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        
        .low-confidence {
            color: #b00;
            border-bottom: 1px dashed #b00;
        }
        '''
    
    def _render_text_block(self, block: Any) -> str:
        """Render a text block as HTML."""
        # Escape HTML
        text = html_lib.escape(block.text)
        
        # Apply styling based on block type
        css_class = f"text-block {block.block_type}"
        
        # Add confidence indicator
        if block.confidence < 0.8:
            css_class += " confidence-low"
        
        # Create HTML
        html = f'<div class="{css_class}">'
        
        # Add text with proper formatting
        if block.block_type in ['title', 'section_heading']:
            level = 'h3' if block.block_type == 'title' else 'h4'
            html += f'<{level}>{text}</{level}>'
        else:
            # Preserve line breaks
            text = text.replace('\n', '<br>')
            html += f'<p>{text}</p>'
        
        # Add metadata if requested
        if hasattr(block, 'metadata') and block.metadata:
            html += f'<div class="metadata">Confidence: {block.confidence:.2f}</div>'
        
        html += '</div>'
        
        return html
    
    def _render_table(self, table: Any) -> str:
        """Render a table as HTML."""
        # Use pre-generated HTML if available
        if table.html:
            return f'<div class="table-container">{table.html}</div>'
        
        # Otherwise generate from cells
        html = '<table>'
        
        # Group cells by row
        cells_by_row = {}
        for cell in table.cells:
            row = cell['row']
            if row not in cells_by_row:
                cells_by_row[row] = []
            cells_by_row[row].append(cell)
        
        # Render rows
        for row_idx in sorted(cells_by_row.keys()):
            html += '<tr>'
            for cell in sorted(cells_by_row[row_idx], key=lambda c: c['col']):
                tag = 'th' if cell.get('is_header') else 'td'
                text = html_lib.escape(cell.get('text', ''))
                if cell.get('low_confidence'):
                    text = f'<span class="low-confidence" title="low confidence">{text}</span>'
                html += f'<{tag}>{text}</{tag}>'
            html += '</tr>'
        
        html += '</table>'
        
        return html
    
    def _render_figure(self, figure: Dict[str, Any]) -> str:
        """Render a figure as HTML."""
        html = '<div class="figure">'
        
        # Convert image to base64 if needed
        if 'image' in figure and isinstance(figure['image'], np.ndarray):
            img_base64 = self._numpy_to_base64(figure['image'])
            html += f'<img src="data:image/png;base64,{img_base64}" alt="Figure">'
        elif 'path' in figure:
            html += f'<img src="{figure["path"]}" alt="Figure">'
        
        # Add caption if available
        if 'caption' in figure:
            caption = html_lib.escape(figure['caption'])
            html += f'<p class="caption">{caption}</p>'
        
        html += '</div>'
        
        return html
    
    def _numpy_to_base64(self, img_array: np.ndarray) -> str:
        """Convert numpy array to base64 string."""
        # Convert to PIL Image
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        img = Image.fromarray(img_array)
        
        # Save to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
