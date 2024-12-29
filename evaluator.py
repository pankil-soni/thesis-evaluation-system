from dotenv import load_dotenv
import spacy
import fitz
import re
import os
import language_tool_python
from typing import Dict, Tuple
from openai import OpenAI
import string
from collections import defaultdict
import numpy as np

load_dotenv()

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')
language_tool = language_tool_python.LanguageTool("en-US")


class BaseEvaluator:
    """Base class with common functionality"""

    def __init__(self, pdf_path, use_llm: bool = True, base_instance=None):

        self.use_llm = use_llm
        self._init_section_patterns()

        if base_instance:
            self.nlp = base_instance.nlp
            self.language_tool = base_instance.language_tool
            self.full_text = base_instance.full_text
            self.sections = base_instance.sections
            self.pdf_path = base_instance.pdf_path
        else:
            self.nlp = nlp
            self.language_tool = language_tool
            self.pdf_path = pdf_path
            print("Extracting text...")
            if self._extract_text():
                print("Text extracted successfully ✅")
            else:
                raise Exception("Error extracting text ❌")

            self.sections = self._extract_sections()

        if use_llm:
            self.open_ai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )

    def _init_section_patterns(self):
        # Enhanced section patterns with more variations and numbered sections
        self.section_patterns = {
            "title": r"^(?:\d{1,2}\s+)?(?:title\s*page|title)$",
            "abstract": r"^(?:\d{1,2}\s+)?(?:abstract|summary|synopsis|overview|précis|executive\s+summary)$",
            "dedication": r"^(?:\d{1,2}\s+)?(?:dedication|dedications|acknowledgement|acknowledgements|acknowledgments|preface)$",
            "toc": r"^(?:\d{1,2}\s+)?(?:table\s+of\s+contents|contents|toc|outline)$",
            "list_of_figures": r"^(?:\d{1,2}\s+)?(?:list\s+of\s+figures|figures|figure\s+index|list\s+of\s+illustrations)$",
            "list_of_tables": r"^(?:\d{1,2}\s+)?(?:list\s+of\s+tables|tables|table\s+index|list\s+of\s+exhibits)$",
            "list_of_abbreviations": r"^(?:\d{1,2}\s+)?(?:list\s+of\s+abbreviations|abbreviations|acronyms|glossary|nomenclature|list\s+of\s+symbols)$",
            "introduction": r"^(?:\d{1,2}\s+)?(?:introduction|preliminaries|overview|general\s+introduction)$",
            "background": r"^(?:\d{1,2}\s+)?(?:background|preliminaries|prerequisites|background\s+and\s+related\s+work|theoretical\s+background|research\s+context)$",
            "literature_review": r"^(?:\d{1,2}\s+)?(?:literature\s*review|related\s*(?:work|research)|previous\s*work|state\s+of\s+the\s+art|literature\s+survey)$",
            "methodology": r"^(?:\d{1,2}\s+)?(?:method(?:ology)?s?|approach|proposed\s*(?:method|approach|framework|solution)|implementation|experimental\s*setup|research\s*design|system\s*design)$",
            "analysis_design": r"(?i)^(?:(?:CHAPTER|Chapter)?\s*\d{1,2}\.?\s*)?(?:ANALYSIS,?\s*DESIGN,?\s*(?:AND\s+)?EXPERIMENTS?|DESIGN\s+AND\s+IMPLEMENTATION|SYSTEM\s+DESIGN\s+AND\s+DEVELOPMENT|DEVELOPMENT\s+AND\s+IMPLEMENTATION|IMPLEMENTATION|SYSTEM\s+IMPLEMENTATION|EXPERIMENTAL\s+SETUP|SYSTEM\s+DESIGN\s+AND\s+DEVELOPMENT)\s*$",
            "results": r"(?i)^(?:(?:CHAPTER|Chapter)?\s*\d{1,2}\.?\s*)?(?:RESULTS(?:\s+AND\s+DISCUSSIONS?)?|FINDINGS|EXPERIMENTAL\s+RESULTS|SIMULATION\s+AND\s+RESULTS|PERFORMANCE\s+EVALUATION|OUTCOMES|DATA\s+ANALYSIS)\s*$",
            "discussion": r"(?i)^(?:(?:CHAPTER|Chapter)?\s*\d{1,2}\.?\s*)?(?:DISCUSSIONS?|ANALYSIS|INTERPRETATION|IMPLICATIONS|DISCUSSION\s+OF\s+RESULTS|CRITICAL\s+ANALYSIS)\s*$",
            "conclusion": r"(?i)^(?:(?:CHAPTER|Chapter)?\s*\d{1,2}\.?\s*)?(?:CONCLUSIONS?\s*((?:AND\s+RECOMMENDATIONS?)|(?:&\s+RECOMMENDATIONS?))?|SUMMARY(?:\s*AND\s*(?:FUTURE\s*WORK|RECOMMENDATIONS?))?|FUTURE\s*(?:WORK|DIRECTIONS)|CONCLUDING\s*REMARKS|RECOMMENDATIONS?|FINAL\s+REMARKS)\s*$",
            "references": r"^(?:\d{1,2}\s+)?(?:references|bibliography|works\s*cited|cited\s*works|sources|literature\s+cited)$",
            "appendix": r"^(?:\d{1,2}\s+)?(?:appendix|appendices|supplementary\s*material|supplemental|annexure|annex)$",
        }

        # Additional numbered section patterns
        self.numbered_section_pattern = re.compile(
            r"^(?:(?:CHAPTER|Chapter|SECTION|Section)\s+)?\d{1,2}(?:\.\d{1,2})*\s+([A-Z][A-Za-z\s\-&,]+)$"
        )

        # Enhanced section number prefixes
        self.section_number_prefixes = [
            r"^\d{1,2}\s+",  # "1 Introduction"
            r"^\d{1,2}\.\d{1,2}\s+",  # "1.1 Background"
            r"^\d{1,2}\.\d{1,2}\.\d{1,2}\s+",  # "1.1.1 Detailed Section"
            r"^(?:CHAPTER|Chapter)\s+\d{1,2}(?:\.)?\s+",  # "CHAPTER 1." or "Chapter 1"
            r"^(?:SECTION|Section)\s+\d{1,2}(?:\.\d{1,2})?\s+",  # "SECTION 1.1"
            r"^[IVX]+\.\s+",  # "I. Introduction" (Roman numerals)
            r"^[A-Z]\.\s+",  # "A. Section" (Letter sections)
        ]

        # Enhanced ignore patterns
        self.ignore_patterns = [
            r"^\d+$",  # Just numbers
            r"^page\s+\d+$",  # Page numbers
            r"^[Pp]\.\s*\d+$",  # P. 1 or p. 1
            r"^figure\s+\d+(?:\.\d+)?[.:]\s*$",  # Figure captions with subsections
            r"^table\s+\d+(?:\.\d+)?[.:]\s*$",  # Table captions with subsections
            r"^\s*$",  # Empty lines
            r"^[A-Za-z]\.$",  # Single letter with period
            r"^\[\d+\]$",  # Reference numbers
            r"^[\u2022\u2023\u2043\u2219]\s",  # Bullet points
            r"^\d+\)\s",  # Numbered lists
            r"^NOTE:",  # Notes
            r"^Source:",  # Source citations
        ]

        # Compile all patterns
        self.compiled_section_patterns = {
            section: re.compile(pattern, re.IGNORECASE)
            for section, pattern in self.section_patterns.items()
        }
        self.compiled_ignore_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.ignore_patterns
        ]
        self.section_number_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.section_number_prefixes
        ]

    def _extract_text(self) -> bool:
        try:
            pdf_content = ""
            with fitz.open(self.pdf_path) as doc:
                for page in doc:
                    blocks = page.get_text("dict")["blocks"]
                    page_width = page.rect.width
                    page_height = page.rect.height

                    for block in blocks:
                        if not block.get("lines"):
                            continue

                        # Skip potential page numbers based on several criteria
                        is_page_number = False

                        # Check if block contains only digits
                        text = "".join(
                            span["text"]
                            for line in block["lines"]
                            for span in line["spans"]
                        ).strip()
                        if text.isdigit():
                            # Get block position
                            bbox = block["bbox"]
                            x0, y0, x1, y1 = bbox

                            # Check if block is in typical page number locations
                            # (bottom center, top center, or corners)
                            is_bottom = y0 > page_height * 0.85
                            is_top = y1 < page_height * 0.15
                            is_center = x0 > page_width * 0.4 and x1 < page_width * 0.6
                            is_corner = (
                                x0 < page_width * 0.15 or x0 > page_width * 0.85
                            ) and (is_top or is_bottom)

                            # Check if block is small (typical for page numbers)
                            is_small = (y1 - y0) < page_height * 0.05

                            is_page_number = (
                                is_small
                                and (is_center or is_corner)
                                and (is_top or is_bottom)
                            )

                        if is_page_number:
                            continue

                        # Extract regular text
                        text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text += span["text"] + " "
                        pdf_content += text + "\n"

            self.full_text = pdf_content
            return True
        except Exception as e:
            print(f"Error extracting text: {e}")
            return False

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and special characters."""
        # Remove multiple spaces and newlines
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep basic punctuation
        text = "".join(ch for ch in text if ch in string.printable)
        # Remove citation numbers and footnote marks
        text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
        text = re.sub(r"\s*\*+\s*$", "", text)
        return text.strip()

    def _should_ignore(self, text: str) -> bool:
        """Check if text should be ignored based on patterns."""
        text = text.strip()
        if not text:
            return True
        return any(pattern.match(text) for pattern in self.compiled_ignore_patterns)

    def _extract_section_info(self, text: str) -> Tuple[str, str]:
        """Extract section title and number from text."""
        text = text.strip()

        # Try to match numbered section patterns
        for pattern in self.section_number_patterns:
            match = pattern.match(text)
            if match:
                section_number = match.group(0)
                clean_title = text[len(section_number) :].strip()
                return section_number, clean_title

        return "", text

    def _get_font_stats(self, doc) -> Dict:
        """Analyze font statistics across the document."""
        font_sizes = []
        font_weights = defaultdict(list)

        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("lines"):
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
                            if "font" in span:
                                font_weights[span["font"]].append(span["size"])

        return {
            "mean_size": np.mean(font_sizes) if font_sizes else 12,
            "std_size": np.std(font_sizes) if font_sizes else 1,
            "median_size": np.median(font_sizes) if font_sizes else 12,
            "font_weights": {
                font: np.mean(sizes) for font, sizes in font_weights.items()
            },
        }

    def _is_likely_section_header(
        self, text: str, font_size: float, font_stats: Dict, prev_text: str = None
    ) -> Tuple[bool, str]:
        """Enhanced section header detection with better number handling."""
        text = text.strip()
        if self._should_ignore(text):
            return False, None

        # Extract section number and clean title
        section_number, clean_title = self._extract_section_info(text)

        # Check numbered section pattern
        numbered_match = self.numbered_section_pattern.match(text)
        if numbered_match:
            return True, "numbered_section"

        # Check if clean title matches any section pattern
        for section, pattern in self.compiled_section_patterns.items():
            if pattern.match(clean_title):
                return True, section

        # Enhanced header detection heuristics
        conditions = [
            len(text.split()) <= 12,  # Not too long
            font_size
            > font_stats["mean_size"]
            + 0.3 * font_stats["std_size"],  # Larger than average
            bool(section_number)
            or text[0].isupper(),  # Either numbered or starts with capital
            not text.lower().startswith(("fig.", "figure", "table", "eq.", "equation")),
            not any(text.lower().endswith(end) for end in [".jpg", ".png", ".pdf"]),
            len(text) > 3,  # Avoid very short headers
        ]

        # Context-aware checks
        if prev_text:
            conditions.append(
                abs(len(text) - len(prev_text))
                > len(text) * 0.5  # Different length from previous
            )

        return all(conditions), None

    def _classify_text_block(
        self, text: str, font_size: float, stats: Dict, prev_block: Dict = None
    ) -> Dict:
        """Classify a block of text with enhanced context awareness."""
        is_header, section_type = self._is_likely_section_header(
            text, font_size, stats, prev_block["text"] if prev_block else None
        )

        return {
            "text": self._clean_text(text),
            "is_header": is_header,
            "section_type": section_type,
            "font_size": font_size,
        }

    def _merge_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Merge related sections and handle special cases."""
        merged = defaultdict(str)

        for section, content in sections.items():
            # Handle background sections
            if section == "numbered_section":
                first_line = content.split("\n")[0].lower()
                if "background" in first_line:
                    merged["background"] += content
                else:
                    # Try to match with known sections
                    matched = False
                    for known_section in self.section_patterns.keys():
                        if known_section in first_line:
                            merged[known_section] += content
                            matched = True
                            break
                    if not matched:
                        merged["other"] += content
            else:
                merged[section] += content

        return dict(merged)

    def _extract_sections(self) -> Dict[str, str]:
        """
        Enhanced section extraction with improved accuracy and organization.
        """
        doc = fitz.open(self.pdf_path)
        font_stats = self._get_font_stats(doc)

        # Initialize section tracking
        sections = defaultdict(str)
        current_section = None
        prev_block = None

        for page in doc:
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                if not block.get("lines"):
                    continue

                # Extract text and font properties
                text = ""
                max_font_size = 0

                for line in block["lines"]:
                    for span in line["spans"]:
                        text += span["text"] + " "
                        max_font_size = max(max_font_size, span["size"])

                text = self._clean_text(text)
                if not text or self._should_ignore(text):
                    continue

                # Classify the block
                block_info = self._classify_text_block(
                    text, max_font_size, font_stats, prev_block
                )

                if block_info["is_header"]:
                    if block_info["section_type"] == "numbered_section":
                        sections[current_section] += f"\n{text}\n"
                    else:
                        current_section = block_info["section_type"] or current_section
                        sections[current_section] += f"\n{text}\n"
                elif current_section:
                    # Add content to current section
                    sections[current_section] += f"{text}\n"

                prev_block = block_info

        # Clean up and merge sections
        cleaned_sections = {}
        for section, content in sections.items():
            if section and section != "None":
                # Remove excessive whitespace and normalize newlines
                cleaned_content = re.sub(r"\n{3,}", "\n\n", content.strip())
                cleaned_sections[section] = cleaned_content

        doc.close()
        return self._merge_sections(cleaned_sections)

    def _get_llm_scores(self, prompt: str) -> Dict[str, float]:
        try:
            response = self.open_ai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic evaluator.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()
            return text
        except Exception as e:
            print(f"Error getting LLM scores: {e}")
            return ""
