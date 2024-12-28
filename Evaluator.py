from dotenv import load_dotenv
import spacy
import pdfplumber
import re
import os
import language_tool_python
from typing import Dict, List, Tuple
from openai import OpenAI

load_dotenv()


class BaseEvaluator:
    """Base class with common functionality"""

    def __init__(self, pdf_path, use_llm: bool = True, base_instance=None):

        self.use_llm = use_llm

        if base_instance:
            self.nlp = base_instance.nlp
            self.language_tool = base_instance.language_tool
            self.full_text = base_instance.full_text
            self.pdf_path = base_instance.pdf_path
        else:
            self.nlp = spacy.load("en_core_web_sm")
            self.language_tool = language_tool_python.LanguageTool("en-US")
            self.pdf_path = pdf_path
            print("Extracting text...")
            if self._extract_text():
                print("Text extracted successfully ✅")
            else:
                raise Exception("Error extracting text ❌")

        if use_llm:
            self.open_ai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )

    def _extract_text(self) -> bool:
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.full_text = "\n".join(page.extract_text() for page in pdf.pages)
            return True
        except Exception as e:
            print(f"Error extracting text: {e}")
            return False

    def _clean_section_text(self, text: str) -> str:
        """Clean up the extracted section text"""
        # Remove subsection numbers and their titles
        text = re.sub(r"\d+\.\d+\s+.*?\n", "", text)

        # Remove page numbers
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)

        # Remove any headers that might have been caught
        text = re.sub(r"(?i)^(chapter|section)\s+\d+.*?\n", "", text)
        return text.strip()

    def _extract_section(
        self, full_text: str, start_patterns: List[str], end_patterns: List[str]
    ) -> str:
        """Generic section extraction"""
        try:
            section_text = ""
            in_section = False
            page_texts = full_text.split("\f")

            for page_text in page_texts:
                lines = page_text.split("\n")

                for line in lines:
                    line = line.strip()

                    if re.match(r"^\d+$", line):
                        continue

                    if not in_section:
                        for pattern in start_patterns:
                            if re.match(pattern, line):
                                in_section = True
                                break
                        continue

                    if in_section:
                        for pattern in end_patterns:
                            if re.match(pattern, line):
                                in_section = False
                                break

                        if not in_section:
                            break

                        if line and not re.match(r"^\d+\.\d+\s+", line):
                            section_text += line + "\n"

            return self._clean_section_text(section_text)

        except Exception as e:
            print(f"Error extracting implementation section: {e}")
            return ""

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
