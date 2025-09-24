from typing import List, Dict, Any
from pathlib import Path
import re


class MarkdownExporter:
    """Export document processing results to Markdown format."""
    
    def __init__(self, 
                 include_metadata: bool = False,
                 include_confidence: bool = False):
        self.include_metadata = include_metadata
        self.include_confidence = include_confidence
    
    def export(self, result: Any, output_path: str):
        """Export document result to Markdown file."""
        md_content = self._generate_markdown(result)
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _generate_markdown(self, result: Any) -> str:
        """Generate Markdown content from document result."""
        lines = []
        
        # Document title
        lines.append(f"# {Path(result.filename).name}")
        lines.append("")

        # LLM results (if any)
        try:
            llm = (result.metadata or {}).get('llm')
            if llm:
                lines.append("## LLM Results")
                # Summary
                if isinstance(llm.get('summary'), str) and llm.get('summary'):
                    lines.append("### Summary")
                    lines.append(str(llm['summary']))
                    lines.append("")
                # QA
                qa = llm.get('qa') if isinstance(llm.get('qa'), dict) else None
                if qa:
                    q = str(qa.get('question', ''))
                    a = '' if qa.get('answer') is None else str(qa.get('answer'))
                    lines.append("### Q&A")
                    lines.append(f"- Q: {q}")
                    lines.append(f"- A: {a}")
                    lines.append("")
                # JSON extraction
                ext = llm.get('extraction') if isinstance(llm.get('extraction'), dict) else None
                if ext:
                    lines.append("### JSON Extraction")
                    js = ext.get('result')
                    try:
                        import json as _json
                        js_pretty = _json.dumps(_json.loads(js) if isinstance(js, str) else js, ensure_ascii=False, indent=2)
                    except Exception:
                        js_pretty = str(js)
                    lines.append("```")
                    lines.append(js_pretty)
                    lines.append("```")
                    lines.append("")
        except Exception:
            pass
        
        # Metadata
        if self.include_metadata:
            lines.append("## Document Information")
            lines.append(f"- **File**: {result.filename}")
            lines.append(f"- **Pages**: {len(result.pages)}")
            if result.metadata:
                for key, value in result.metadata.items():
                    lines.append(f"- **{key}**: {value}")
            lines.append("")
        
        # Process each page
        for page in result.pages:
            # Page header
            if len(result.pages) > 1:
                lines.append(f"## Page {page.page_num}")
                lines.append("")
            
            # Get blocks in reading order
            if page.reading_order:
                ordered_blocks = [page.text_blocks[i] for i in page.reading_order]
            else:
                ordered_blocks = page.text_blocks
            
            # Current section tracking for hierarchy
            current_section = None
            
            # Render text blocks
            for block in ordered_blocks:
                md_text = self._render_text_block(block, current_section)
                if md_text:
                    lines.append(md_text)
                    lines.append("")
                
                # Update current section
                if block.block_type in ['title', 'section_heading']:
                    current_section = block.text
            
            # Render tables
            for table in page.tables:
                md_table = self._render_table(table)
                if md_table:
                    lines.append(md_table)
                    lines.append("")
            
            # Add page separator
            if page.page_num < len(result.pages):
                lines.append("---")
                lines.append("")
        
        return '\n'.join(lines)
    
    def _render_text_block(self, block: Any, current_section: str = None) -> str:
        """Render a text block as Markdown."""
        text = block.text.strip()
        if not text:
            return ""
        
        # Add confidence indicator if requested
        confidence_suffix = ""
        if self.include_confidence and block.confidence < 0.8:
            confidence_suffix = f" <!-- confidence: {block.confidence:.2f} -->"
        
        # Format based on block type
        if block.block_type == 'title':
            # Main title (H2 since H1 is document name)
            return f"## {text}{confidence_suffix}"
        
        elif block.block_type == 'section_heading':
            # Section heading (H3)
            return f"### {text}{confidence_suffix}"
        
        elif block.block_type == 'list':
            # Format as list items
            lines = text.split('\n')
            formatted_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    # Detect if numbered or bullet list
                    if re.match(r'^\d+[\.\)]\s*', line):
                        # Already numbered
                        formatted_lines.append(line)
                    elif re.match(r'^[•・◦▪▸]\s*', line):
                        # Convert bullet to markdown
                        line = re.sub(r'^[•・◦▪▸]\s*', '- ', line)
                        formatted_lines.append(line)
                    else:
                        # Add bullet
                        formatted_lines.append(f"- {line}")
            return '\n'.join(formatted_lines) + confidence_suffix
        
        elif block.block_type == 'caption':
            # Italic text for captions
            return f"*{text}*{confidence_suffix}"
        
        elif block.block_type == 'footnote':
            # Footnote formatting
            return f"> {text}{confidence_suffix}"
        
        elif block.block_type == 'equation':
            # Math block
            if '\n' in text:
                return f"$$\n{text}\n$${confidence_suffix}"
            else:
                return f"${text}${confidence_suffix}"
        
        elif block.block_type == 'reference':
            # Reference formatting
            return f"[{text}]{confidence_suffix}"
        
        else:
            # Regular paragraph
            # Preserve line breaks within paragraph
            text = text.replace('\n', '  \n')
            return f"{text}{confidence_suffix}"
    
    def _render_table(self, table: Any) -> str:
        """Render a table as Markdown."""
        # Use pre-generated markdown if available
        if table.markdown:
            return table.markdown
        
        # Otherwise generate from cells
        if not table.cells:
            return ""
        
        # Group cells by row
        cells_by_row = {}
        max_col = 0
        for cell in table.cells:
            row = cell['row']
            col = cell['col']
            if row not in cells_by_row:
                cells_by_row[row] = {}
            cells_by_row[row][col] = cell
            max_col = max(max_col, col)
        
        # Build markdown table
        lines = []
        
        for row_idx in sorted(cells_by_row.keys()):
            row_cells = cells_by_row[row_idx]
            
            # Build row
            row_parts = []
            for col_idx in range(max_col + 1):
                if col_idx in row_cells:
                    cell = row_cells[col_idx]
                    text = cell.get('text', '')
                    if cell.get('low_confidence'):
                        text = f"{text} (low)"
                    # Escape pipe characters
                    text = text.replace('|', '\\|')
                    row_parts.append(text)
                else:
                    row_parts.append('')
            
            lines.append('| ' + ' | '.join(row_parts) + ' |')
            
            # Add separator after first row
            if row_idx == 0:
                separator_parts = ['---' for _ in range(max_col + 1)]
                lines.append('|' + '|'.join(separator_parts) + '|')
        
        return '\n'.join(lines)
    
    def _escape_markdown(self, text: str) -> str:
        """Escape special Markdown characters."""
        # Escape characters that have special meaning in Markdown
        special_chars = ['\\', '*', '_', '[', ']', '(', ')', '#', '+', '-', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
