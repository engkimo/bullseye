"""
Harmony format utilities for LLM training and inference.
Handles formatting of prompts and responses according to reasoning levels.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re


class ReasoningLevel(Enum):
    """Reasoning levels in Harmony format."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HarmonyFormatter:
    """Format prompts and responses in Harmony format."""
    
    def __init__(self):
        self.doc_delimiter = "[文書]"
        self.question_delimiter = "[質問]"
        self.context_delimiter = "[文脈]"
        self.instruction_delimiter = "[指示]"
    
    def format_prompt(self, 
                     document: str,
                     question: str,
                     reasoning_level: ReasoningLevel = ReasoningLevel.LOW,
                     additional_context: Optional[str] = None) -> str:
        """Format input prompt in Harmony style."""
        
        parts = []
        
        # Add context if provided
        if additional_context:
            parts.append(f"{self.context_delimiter} {additional_context}")
        
        # Add document
        parts.append(f"{self.doc_delimiter} {document}")
        
        # Add question with reasoning hint
        if reasoning_level == ReasoningLevel.HIGH:
            parts.append(f"{self.question_delimiter} {question} (詳細な分析と説明を含めて回答してください)")
        elif reasoning_level == ReasoningLevel.MEDIUM:
            parts.append(f"{self.question_delimiter} {question} (理由も含めて回答してください)")
        else:
            parts.append(f"{self.question_delimiter} {question}")
        
        return " ".join(parts)
    
    def format_response(self,
                       answer: str,
                       reasoning_level: ReasoningLevel = ReasoningLevel.LOW,
                       confidence: Optional[float] = None) -> str:
        """Format response according to reasoning level."""
        
        if reasoning_level == ReasoningLevel.LOW:
            # Direct answer only
            return answer.strip()
        
        elif reasoning_level == ReasoningLevel.MEDIUM:
            # Answer with brief reasoning
            if "なぜなら" not in answer and "理由は" not in answer:
                # Add reasoning prefix if not present
                return f"{answer}。"
            return answer
        
        else:  # HIGH
            # Detailed analysis
            if confidence is not None:
                return f"{answer}\n(確信度: {confidence:.2f})"
            return answer
    
    def parse_prompt(self, prompt: str) -> Dict[str, str]:
        """Parse Harmony format prompt into components."""
        
        components = {
            'document': '',
            'question': '',
            'context': '',
            'reasoning_level': ReasoningLevel.LOW
        }
        
        # Extract document
        doc_match = re.search(f"{re.escape(self.doc_delimiter)}\\s*(.+?)\\s*{re.escape(self.question_delimiter)}", prompt, re.DOTALL)
        if doc_match:
            components['document'] = doc_match.group(1).strip()
        
        # Extract question
        q_match = re.search(f"{re.escape(self.question_delimiter)}\\s*(.+?)$", prompt, re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
            components['question'] = question
            
            # Detect reasoning level from question
            if "詳細な分析" in question or "説明を含めて" in question:
                components['reasoning_level'] = ReasoningLevel.HIGH
            elif "理由も含めて" in question or "なぜ" in question:
                components['reasoning_level'] = ReasoningLevel.MEDIUM
        
        # Extract context if present
        ctx_match = re.search(f"{re.escape(self.context_delimiter)}\\s*(.+?)\\s*{re.escape(self.doc_delimiter)}", prompt, re.DOTALL)
        if ctx_match:
            components['context'] = ctx_match.group(1).strip()
        
        return components
    
    def format_json_extraction_prompt(self,
                                    document: str,
                                    schema: Dict[str, Any],
                                    examples: Optional[List[Dict]] = None) -> str:
        """Format prompt for JSON extraction task."""
        
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        prompt_parts = [f"{self.doc_delimiter} {document}"]
        
        if examples:
            examples_str = "\n".join([
                f"例{i+1}: {json.dumps(ex, ensure_ascii=False)}"
                for i, ex in enumerate(examples[:3])
            ])
            prompt_parts.append(f"{self.context_delimiter} 以下の形式で抽出してください:\n{examples_str}")
        
        prompt_parts.append(f"{self.question_delimiter} 次のスキーマに従ってJSONで情報を抽出してください:\n{schema_str}")
        
        return " ".join(prompt_parts)
    
    def format_qa_prompt(self,
                        document: str,
                        question: str,
                        options: Optional[List[str]] = None) -> str:
        """Format QA prompt with optional multiple choice."""
        
        if options:
            # Multiple choice format
            options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
            question_with_options = f"{question}\n選択肢:\n{options_str}"
            return self.format_prompt(document, question_with_options, ReasoningLevel.LOW)
        else:
            # Open-ended question
            return self.format_prompt(document, question, ReasoningLevel.LOW)
    
    def format_summary_prompt(self,
                            document: str,
                            max_length: Optional[int] = None,
                            focus_points: Optional[List[str]] = None) -> str:
        """Format summarization prompt."""
        
        question_parts = ["この文書を要約してください"]
        
        if max_length:
            question_parts.append(f"（{max_length}文字以内）")
        
        if focus_points:
            points_str = "、".join(focus_points)
            question_parts.append(f"。特に{points_str}に注目してください")
        
        question = "".join(question_parts)
        
        return self.format_prompt(document, question, ReasoningLevel.MEDIUM)
    
    def validate_json_response(self, response: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Validate JSON response against schema."""
        
        # Try to parse JSON
        try:
            # Handle responses that might have extra text
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response)
        except json.JSONDecodeError as e:
            return False, None, f"JSON parse error: {str(e)}"
        
        # Basic schema validation
        errors = []
        for key, expected_type in schema.items():
            if key not in parsed:
                errors.append(f"Missing required field: {key}")
            elif expected_type == "string" and not isinstance(parsed[key], str):
                errors.append(f"Field {key} should be string")
            elif expected_type == "number" and not isinstance(parsed[key], (int, float)):
                errors.append(f"Field {key} should be number")
            elif expected_type == "date" and not self._is_date_format(parsed[key]):
                errors.append(f"Field {key} should be date format (YYYY-MM-DD)")
        
        if errors:
            return False, parsed, "; ".join(errors)
        
        return True, parsed, None
    
    def _is_date_format(self, value: Any) -> bool:
        """Check if value is in date format."""
        if not isinstance(value, str):
            return False
        
        # Simple date format check (YYYY-MM-DD)
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        return bool(date_pattern.match(value))
    
    def inject_na_response(self, prompt: str, na_probability: float = 0.25) -> Tuple[str, bool]:
        """Inject N/A cases for training robustness."""
        
        import random
        
        if random.random() < na_probability:
            # Modify prompt to make answer unavailable
            components = self.parse_prompt(prompt)
            
            # Add contradictory or missing information
            modifications = [
                "（該当する情報は文書に含まれていません）",
                "（この部分は判読不能です）",
                "（データが欠損しています）"
            ]
            
            modified_doc = components['document'] + " " + random.choice(modifications)
            
            return self.format_prompt(
                modified_doc,
                components['question'],
                components['reasoning_level']
            ), True
        
        return prompt, False