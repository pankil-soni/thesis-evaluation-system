from evaluator import BaseEvaluator
import pypdf
import re
from spellchecker import SpellChecker


class StructureAndGrammarEvaluator(BaseEvaluator):
    def __init__(self, pdf_path, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)
        # Define these once during initialization
        self.essential_sections = [
            "abstract",
            "introduction",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references",
        ]
        # Common academic/technical words to exclude from spell check
        self.technical_words = set(
            [
                "methodology",
                "analysis",
                "data",
                "research",
                "hypothesis",
                "theoretical",
                "empirical",
            ]
        )

    def _structure_check(self):
        """structure checking using regex"""
        text_lower = self.full_text.lower()
        found_sections = []
        section_positions = []

        for section in self.essential_sections:
            matches = list(re.finditer(rf"\b{section}\b", text_lower))
            if matches:
                found_sections.append(section)
                section_positions.append(matches[0].start())

        # Check if sections are in logical order
        order_score = 1.0 if sorted(section_positions) == section_positions else 0.7
        return (
            len(found_sections) / len(self.essential_sections)
        ) * order_score, found_sections

    def _efficient_formatting_check(self):
        """Simplified formatting check focusing on key indicators"""
        score = 0

        # check if greater than 1000 words
        if len(self.full_text) >= 1000:
            score += 0.25

        # Sample-based line spacing check
        lines = self.full_text.split("\n")[:100]  # Check first 100 lines only
        if lines:
            non_empty_lines = [len(line) for line in lines if line.strip()]
            if non_empty_lines:
                avg_length = sum(non_empty_lines) / len(non_empty_lines)
                if 40 <= avg_length <= 100:  # Reasonable line length
                    score += 0.25

        # header check using simplified pattern
        caps_lines = len(
            re.findall(
                r"^[A-Z][^a-z\n]{2,}[A-Za-z ]*$", self.full_text[:5000], re.MULTILINE
            )
        )
        if caps_lines >= 3:
            score += 0.25

        return score

    def _grammar_spell_check(self):
        """Efficient grammar and spelling check"""
        # Take a small sample for analysis
        sample_text = self.full_text[:10000]
        words = re.findall(r"\b\w+\b", sample_text.lower())

        if not words:
            return 0

        # Basic grammar checks (than LanguageTool)
        grammar_errors = 0

        # Check for basic patterns
        grammar_patterns = {
            r"\b(a)\s+[aeiou]": 1,  # Articles
            r"\b(is|are|am)\s+\w+ed\b": 1,  # Verb agreement
            r"\b(their|there|they\'re)\b": 0.5,  # Common confusions
            r"\b(its|it\'s)\b": 0.5,
            r"\b(have|has|had)\s+been\b": 0.5,
            r"\b(\w+ed)\s+(?:\1|a|an)\b": 0.5,
            r"\b(\w+ing)\s+(?:\1|a|an)\b": 0.5,
        }

        for pattern, weight in grammar_patterns.items():
            grammar_errors += len(re.findall(pattern, sample_text.lower())) * weight

        # spell check on unique words
        unique_words = set(words) - self.technical_words
        spell = SpellChecker()
        misspelled = spell.unknown(unique_words)
        spelling_error_rate = len(misspelled) / len(words)

        # Combined score
        error_score = (grammar_errors / len(words) + spelling_error_rate) / 2
        return max(0, 1 - error_score)

    def _style_check(self):
        """Simplified style analysis"""
        sample_text = self.full_text[:3000]
        sentences = re.split(r"[.!?]+", sample_text)

        if not sentences:
            return 0

        # sentence variety check
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0

        avg_length = sum(lengths) / len(lengths)
        length_variety = sum(1 for l in lengths if abs(l - avg_length) < 10) / len(
            lengths
        )

        # Basic vocabulary check
        words = re.findall(r"\b\w+\b", sample_text.lower())
        unique_ratio = len(set(words)) / len(words) if words else 0

        return length_variety * 0.6 + unique_ratio * 0.4

    def evaluate(self):
        """Optimized evaluation process"""
        if not self._extract_text():
            return 0

        # Get scores using methods
        structure_score, found_sections = self._structure_check()
        formatting_score = self._efficient_formatting_check()
        grammar_spelling_score = self._grammar_spell_check()
        style_score = self._style_check()

        # Calculate weighted final score (out of 5)
        weights = {
            "structure": 0.35,  # Increased weight for structure
            "formatting": 0.15,
            "grammar_spelling": 0.3,
            "style": 0.2,
        }

        final_score = 5 * (
            structure_score * weights["structure"]
            + formatting_score * weights["formatting"]
            + grammar_spelling_score * weights["grammar_spelling"]
            + style_score * weights["style"]
        )

        # Grade mapping
        grade_mapping = [
            (4.5, "Distinction (5)"),
            (4.0, "Distinction (4)"),
            (3.0, "Merit (3)"),
            (2.0, "Pass (2)"),
            (1.0, "Fail (1)"),
            (0, "Fail (0)"),
        ]

        grade = next(
            (grade for threshold, grade in grade_mapping if final_score >= threshold),
            "Fail (0)",
        )

        return {
            "score": float(round(final_score, 2)),
            "grade": grade,
            "details": {
                "structure": {
                    "score": float(round(structure_score * 5, 2)),
                    "found_sections": found_sections,
                },
                "formatting": float(round(formatting_score * 5, 2)),
                "grammar_spelling": float(round(grammar_spelling_score * 5, 2)),
                "style": float(round(style_score * 5, 2)),
            },
        }
