from Evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple
from datetime import datetime


class LiteratureReviewEvaluator(BaseEvaluator):
    """Evaluates thesis literature review"""

    def __init__(self, pdf_path, use_llm: bool = True, base_instance=None):
        """
        Initialize the LiteratureReviewEvaluator with option to use LLM
        """
        super().__init__(pdf_path, use_llm, base_instance)

    def extract_literature_review(self) -> str:
        """
        Extract the introduction section accurately using multiple extraction methods
        """

        # Patterns for identifying literature review section
        lit_review_patterns = [
            r"(?i)^(?:CHAPTER\s+2\.?\s*)?LITERATURE\s+REVIEW\s*$",
            r"(?i)^(?:2\.?\s+)?LITERATURE\s+REVIEW\s*$",
            r"(?i)^(?:BACKGROUND|RELATED\s+WORK)\s*$",
            r"(?i)^(?:STATE\s+OF\s+THE\s+ART)\s*$",
            r"(?i)^(?:THEORETICAL\s+FRAMEWORK)\s*$",
        ]

        # Patterns for next section
        next_section_patterns = [
            r"(?i)^(?:CHAPTER\s+3)",
            r"(?i)^(?:3\.?\s+)",
            r"(?i)^(?:METHODOLOGY)",
            r"(?i)^(?:RESEARCH\s+DESIGN)",
            r"(?i)^(?:EXPERIMENTAL\s+SETUP)",
            r"(?i)^(?:RESEARCH\s+METHODOLOGY)",
        ]

        try:
            lit_review_text = self._extract_section(
                self.full_text, lit_review_patterns, next_section_patterns
            )
            return lit_review_text
        except Exception as e:
            print(f"Error in extract_literature_review: {e}")
            return ""

    def _check_research_components(self, intro_text: str) -> Dict:
        """
        Check for presence and quality of key research components
        """
        components = {
            "problem_statement": False,
            "research_questions": False,
            "objectives": False,
            "justification": False,
        }

        # Keywords and patterns for each component
        patterns = {
            "problem_statement": [
                r"problem statement",
                r"(?i)problem statement",
                r"(?i)research problem",
                r"(?i)this study addresses",
                r"(?i)the problem is",
                r"(?i)the main issue",
            ],
            "research_questions": [
                r"(?i)research question",
                r"(?i)\?",
                r"(?i)this study seeks to",
                r"(?i)we investigate",
                r"(?i)aims to answer",
            ],
            "objectives": [
                r"(?i)objective",
                r"(?i)aim of this",
                r"(?i)purpose of this",
                r"(?i)goal of this",
                r"(?i)this study aims",
            ],
            "justification": [
                r"(?i)significance",
                r"(?i)importance",
                r"(?i)justification",
                r"(?i)rationale",
                r"(?i)this is important because",
            ],
        }

        # Check for each component
        for component, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, intro_text):
                    components[component] = True
                    break

        return components

    def analyze_citations(self, text: str) -> Dict:
        """Analyze citations and references in the literature review"""
        # Extract citations using regex patterns
        citation_patterns = [
            r"\(\w+\s*(?:et al\.?)?,\s*\d{4}\)",  # (Author et al., 2020)
            r"\[[\d,\s-]+\]",  # [1] or [1,2] or [1-3]
            r"\(\d{4}\)",  # (2020)
        ]

        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, text))

        # Analyze citation years
        years = []
        for citation in citations:
            year_match = re.search(r"\d{4}", citation)
            if year_match:
                years.append(int(year_match.group()))

        if not years:
            return {"count": 0, "recency": 0, "distribution": 0}

        current_year = datetime.now().year
        avg_year = np.mean(years) if years else 0
        recency_score = min(1.0, max(0, (avg_year - (current_year - 10)) / 10))

        return {
            "count": len(citations),
            "recency": recency_score,
            "distribution": np.std(years) if len(years) > 1 else 0,
        }

    def analyze_coverage(self, text: str) -> float:
        """Analyze the breadth and depth of literature coverage"""
        doc = self.nlp(text)

        # Identify key research concepts and methodologies
        research_terms = set()
        methodology_terms = set()

        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3:
                research_terms.add(token.text.lower())
            if token.dep_ == "compound" and token.head.pos_ == "NOUN":
                research_terms.add((token.text + " " + token.head.text).lower())

        # Calculate coverage score based on unique concepts
        coverage_score = min(1.0, len(research_terms) / 100)  # Normalize to max of 1.0
        return coverage_score

    def _evaluate_with_llm(self, text: str) -> Dict:
        """Use LLM to evaluate the literature review quality"""

        prompt = f"""
        Evaluate this literature review section based on the following criteria:
        1. Comprehensiveness (0-15):
           - Breadth and depth of covered literature
           - Systematic approach to review
           - Critical analysis of sources
        2. Citation quality (0-15):
           - Use of recent and relevant sources
           - Proper integration of citations
        3. Synthesis (0-15):
           - Connection between different works
           - Identification of research gaps
           - Critical evaluation

        output format:

        {{
          comprehensiveness: number,
          citation_quality: number,
          synthesis: number,
          justification: string
        }}

        Literature review text:
        {text[:4000]}...

        Provide numerical scores and brief justification in JSON format.
        """

        try:
            response = self._get_llm_scores(prompt)
            return eval(response)
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                "comprehensiveness": 0,
                "citation_quality": 0,
                "synthesis": 0,
                "justification": "Error in LLM evaluation",
            }

    def calculate_final_score(
        self, citation_analysis: Dict, coverage_score: float, llm_scores: Dict = None
    ) -> Tuple[float, str]:
        """Calculate final score and determine grade"""
        # Base scores (without LLM)
        citation_score = (
            min(1.0, citation_analysis["count"] / 30) * 5
        )  # Normalize to 5 points
        recency_score = citation_analysis["recency"] * 5
        coverage_score = coverage_score * 5

        if self.use_llm and llm_scores:
            # Combine with LLM scores (normalized to 15 points)
            llm_avg = (
                llm_scores["comprehensiveness"]
                + llm_scores["citation_quality"]
                + llm_scores["synthesis"]
            ) / 3
            final_score = (
                citation_score * 0.2
                + recency_score * 0.2
                + coverage_score * 0.2
                + llm_avg * 0.4
            )
        else:
            final_score = (citation_score + recency_score + coverage_score) / 3

        # Determine grade
        if final_score >= 14:
            grade = "Distinction (14-15)"
        elif final_score >= 11:
            grade = "Distinction (11-13)"
        elif final_score >= 9:
            grade = "Merit (9-10)"
        elif final_score >= 8:
            grade = "Pass (8)"
        elif final_score >= 4:
            grade = "Fail (4-7)"
        else:
            grade = "Fail (0-3)"

        return final_score, grade

    def evaluate(self) -> Dict:
        """Perform complete evaluation of the literature review"""
        # Extract literature review
        lit_review_text = self.extract_literature_review()
        if not lit_review_text:
            return {
                "score": 0,
                "grade": "Fail (0-3)",
                "feedback": "Literature review section not found",
            }

        # Analyze citations
        citation_analysis = self.analyze_citations(lit_review_text)

        # Analyze coverage
        coverage_score = self.analyze_coverage(lit_review_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_llm(lit_review_text)

        # Calculate final score
        score, grade = self.calculate_final_score(
            citation_analysis, coverage_score, llm_scores
        )

        # Generate feedback
        feedback = []
        if citation_analysis["count"] < 20:
            feedback.append("Insufficient number of citations")
        if citation_analysis["recency"] < 0.5:
            feedback.append("Citations are not recent enough")
        if coverage_score < 0.6:
            feedback.append("Limited coverage of the research area")

        if llm_scores:
            feedback.append(f"LLM Analysis: {llm_scores.get('justification', '')}")

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "citation_analysis": citation_analysis,
            "coverage_score": float(round(coverage_score * 5, 2)),
            "llm_scores": llm_scores,
            "feedback": ". ".join(feedback),
        }
