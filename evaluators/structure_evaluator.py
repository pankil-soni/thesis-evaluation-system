from evaluator import BaseEvaluator
import re
from spellchecker import SpellChecker

class StructureAndGrammarEvaluator(BaseEvaluator):
    def __init__(self, pdf_path, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)
        self.spell = SpellChecker()
        self.essential_sections = [
            "abstract",
            "introduction",
            "methodology",
            "results",
            "discussion",
            "conclusion",
            "references",
        ]
        # Expanded technical words list
        self.technical_words = set(
            [
                "methodology",
                "analysis",
                "data",
                "research",
                "hypothesis",
                "theoretical",
                "empirical",
                "quantitative",
                "qualitative",
                "algorithm",
                "implementation",
                "framework",
                "paradigm",
                "correlation",
                "coefficient",
                "visualization",
                "parameter",
                "optimization",
                "regression",
                "validation",
                "metrics",
                "dissertation",
                "thesis",
                "academia",
                "scholarly",
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

    def _enhanced_grammar_check(self, text):
        """More comprehensive grammar checking using spaCy and patterns"""
        doc = self.nlp(text)
        errors = 0

        # Common grammar patterns to check
        grammar_patterns = {
            r"\b(a)\s+[aeiou]": 2,  # Article errors
            r"\b(is|are|am)\s+\w+ed\b": 1,  # BE verb agreement
            r"\b(their|there|they're)\b": 1,  # Common confusions
            r"\b(its|it's)\b": 1,
            r"\b(have|has|had)\s+been\b": 0.5,
            r"\b(\w+ed)\s+(?:\1)\b": 1,  # Repeated words
            r"\b(in|on|at)\s+(?:\1)\b": 1,  # Repeated prepositions
            r"\b(this|that|these|those)\s+(?:is|are)\b": 0.5,  # Demonstrative agreement
        }

        # Check patterns
        for pattern, weight in grammar_patterns.items():
            errors += len(re.findall(pattern, text.lower())) * weight

        # Check for sentence boundary detection issues
        for sent in doc.sents:
            if len(sent.text.split()) < 3:  # Very short sentences
                errors += 0.5
            if len(sent.text.split()) > 40:  # Very long sentences
                errors += 1

        # Check for subject-verb agreement using spaCy
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                # Simple number agreement check
                if token.tag_.startswith("NN") and token.head.tag_.startswith("VB"):
                    if (token.tag_ == "NNS" and token.head.tag_ == "VBZ") or (
                        token.tag_ == "NN" and token.head.tag_ == "VBP"
                    ):
                        errors += 1

        return errors

    def _advanced_style_check(self, text):
        """Enhanced style analysis using spaCy"""
        doc = self.nlp(text)

        # Analyze sentence variety
        sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
        if not sentence_lengths:
            return 0

        # Calculate sentence length variance (good writing has varied sentence lengths)
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(
            sentence_lengths
        )
        length_score = min(1.0, length_variance / 100)  # Normalize variance score

        # Vocabulary richness
        words = [
            token.text.lower() for token in doc if token.is_alpha and not token.is_stop
        ]
        unique_words = set(words)
        vocabulary_score = len(unique_words) / (
            len(words) + 1
        )  # Add 1 to avoid division by zero

        # Check for passive voice (too many passives is typically not good)
        passives = sum(1 for token in doc if token.dep_ == "nsubjpass")
        passive_ratio = passives / len(doc)
        passive_score = 1 - min(1.0, passive_ratio * 5)  # Penalize high passive usage

        return length_score * 0.4 + vocabulary_score * 0.4 + passive_score * 0.2

    def _grammar_spell_check(self):
        """Enhanced grammar and spelling check"""
        sample_text = self.full_text[
            :10000
        ]  # Analyze first 10000 chars for performance

        # Spell check
        words = [word.lower() for word in re.findall(r"\b\w+\b", sample_text)]
        unique_words = set(words) - self.technical_words
        misspelled = self.spell.unknown(unique_words)
        spelling_error_rate = len(misspelled) / (len(words) + 1)

        # Grammar check
        grammar_errors = self._enhanced_grammar_check(sample_text)
        grammar_error_rate = grammar_errors / (len(words) + 1)

        # Combined score with weights
        spelling_weight = 0.4
        grammar_weight = 0.6

        final_score = max(
            0,
            1
            - (
                spelling_error_rate * spelling_weight
                + grammar_error_rate * grammar_weight
            ),
        )

        return final_score

    def _style_check(self):
        """Enhanced style check implementation"""
        sample_text = self.full_text[:5000]
        return self._advanced_style_check(sample_text)

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
