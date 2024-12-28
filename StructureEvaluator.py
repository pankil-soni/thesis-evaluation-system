from nltk.sem.util import evaluate
from Evaluator import BaseEvaluator
import pypdf
import numpy as np
import re
from textblob import TextBlob


class StructureAndGrammarEvaluator(BaseEvaluator):
    """Evaluates thesis structure and grammar"""

    def __init__(self, pdf_path, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)

    def check_structure(self):
        """Check for presence of essential sections"""
        essential_sections = [
            "abstract",
            "introduction",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references",
        ]

        found_sections = []
        score = 0
        text_lower = self.full_text.lower()

        for section in essential_sections:
            if section in text_lower:
                found_sections.append(section)
                score += 1

        return score / len(essential_sections), found_sections

    def _check_formatting(self):
        """Check formatting consistency"""
        formatting_score = 0

        # Check page numbers
        with pypdf.PdfReader(self.pdf_path) as pdf:
            num_pages = len(pdf.pages)
            if num_pages > 1:  # Basic check for multiple pages
                formatting_score += 0.2

        # Check for consistent line spacing
        lines = self.full_text.split("\n")
        line_lengths = [len(line) for line in lines if line.strip()]
        avg_length = np.mean(line_lengths)
        std_length = np.std(line_lengths)
        if std_length / avg_length < 0.5:  # Check for consistency
            formatting_score += 0.2

        # Check for proper headers (looking for capitalization patterns)
        header_pattern = re.compile(r"^[A-Z][^.!?]*$", re.MULTILINE)
        potential_headers = header_pattern.findall(self.full_text)
        if len(potential_headers) >= 3:  # At least some headers found
            formatting_score += 0.2

        return formatting_score

    def _check_grammar_spelling(self):
        """Check grammar and spelling"""
        # Use LanguageTool for grammar checking
        matches = self.language_tool.check(
            self.full_text[:10000]
        )  # Check first 10000 chars for performance
        errors_per_word = len(matches) / len(self.full_text.split())

        # Use TextBlob for spelling
        blob = TextBlob(self.full_text[:10000])
        words = blob.words
        misspelled = len([word for word in words if not word.spellcheck()[0][1] == 1])
        spelling_error_rate = misspelled / len(words)

        # Calculate combined score
        grammar_spelling_score = 1.0 - (errors_per_word + spelling_error_rate) / 2
        return max(0, min(grammar_spelling_score, 1))  # Normalize between 0 and 1

    def _check_writing_style(self):
        """Analyze writing style"""
        doc = self.nlp(self.full_text[:10000])  # Analyze first 10000 chars

        # Check sentence variety
        sentence_lengths = [len(sent) for sent in doc.sents]
        length_variety = np.std(sentence_lengths) / np.mean(sentence_lengths)

        # Check vocabulary richness
        words = [token.text.lower() for token in doc if token.is_alpha]
        unique_words = len(set(words)) / len(words)

        # Calculate style score
        style_score = length_variety * 0.5 + unique_words * 0.5
        return min(style_score, 1.0)  # Normalize to 1

    def evaluate(self):
        """Perform complete evaluation and return final score"""
        if not self._extract_text():
            return 0

        # Get individual scores
        structure_score, found_sections = self.check_structure()
        self.sections = found_sections
        formatting_score = self._check_formatting()
        grammar_spelling_score = self._check_grammar_spelling()
        style_score = self._check_writing_style()

        # Calculate weighted final score (out of 5)
        weights = {
            "structure": 0.3,
            "formatting": 0.2,
            "grammar_spelling": 0.3,
            "style": 0.2,
        }

        final_score = 5 * (
            structure_score * weights["structure"]
            + formatting_score * weights["formatting"]
            + grammar_spelling_score * weights["grammar_spelling"]
            + style_score * weights["style"]
        )

        # Determine grade based on score
        if final_score >= 4.5:
            grade = "Distinction (5)"
        elif final_score >= 4.0:
            grade = "Distinction (4)"
        elif final_score >= 3.0:
            grade = "Merit (3)"
        elif final_score >= 2.0:
            grade = "Pass (2)"
        elif final_score >= 1.0:
            grade = "Fail (1)"
        else:
            grade = "Fail (0)"

        return {
            "final_score": round(final_score, 2),
            "grade": grade,
            "details": {
                "structure": {
                    "score": round(structure_score * 5, 2),
                    "found_sections": found_sections,
                },
                "formatting": round(formatting_score * 5, 2),
                "grammar_spelling": round(grammar_spelling_score * 5, 2),
                "style": round(style_score * 5, 2),
            },
        }


def evaluate_structure(pdf_path, use_llm: bool = True):
    """Function to evaluate a thesis PDF"""
    evaluator = StructureAndGrammarEvaluator(pdf_path, use_llm)
    results = evaluator.evaluate()
    # Print detailed results
    print("\n=== Thesis Evaluation Results ===")
    print(f"Final Score: {results['final_score']}/5.0")
    print(f"Grade: {results['grade']}")
    print("\nDetailed Scores:")
    print(f"Structure: {results['details']['structure']['score']}/5.0")
    print(
        "Found Sections:", ", ".join(results["details"]["structure"]["found_sections"])
    )
    print(f"Formatting: {results['details']['formatting']}/5.0")
    print(f"Grammar & Spelling: {results['details']['grammar_spelling']}/5.0")
    print(f"Writing Style: {results['details']['style']}/5.0")

    return results
