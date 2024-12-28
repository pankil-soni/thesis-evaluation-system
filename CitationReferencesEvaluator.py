from evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple
from datetime import datetime


class CitationReferencesEvaluator(BaseEvaluator):
    """Evaluates thesis citations and references"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        """Initialize the CitationReferencesEvaluator with option to use LLM"""
        super().__init__(pdf_path, use_llm, base_instance)

        # Initialize citation and reference patterns
        self.citation_patterns = {
            "in_text_citations": [
                r"\(\w+\s*(?:et al\.?)?,\s*\d{4}\)",  # (Author et al., 2020)
                r"\[\d+\]",  # [1]
                r"\[\d+(?:,\s*\d+)*\]",  # [1,2,3]
                r"\[\d+(?:\s*-\s*\d+)\]",  # [1-3]
                r"\w+\s+(?:et al\.?)?\s*\(\d{4}\)",  # Author et al. (2020)
                r"\w+\s+and\s+\w+\s*\(\d{4}\)",  # Author and Author (2020)
            ],
            "reference_formats": [
                r"^[A-Z][^.]*\.\s*\(\d{4}\)",  # Author, A. (2020)
                r"^\[\d+\]\s*[A-Z]",  # [1] Author
                r"^\d+\.\s*[A-Z]",  # 1. Author
            ],
        }

        self.reference_components = {
            "author": r"(?:[A-Z][a-z]+(?:[,.]|\s+and\s+|,\s+et\s+al\.?))",
            "year": r"\(\d{4}\)",
            "title": r'["\'].*?["\']|\S+[.]',
            "journal": r"[A-Z][A-Za-z\s&]+[.,]",
            "volume": r"(?:Vol\.|Volume)\s*\d+",
            "pages": r"pp?\.\s*\d+(?:-\d+)?",
        }

    def extract_references_section(self) -> str:
        """Extract the references section"""
        reference_patterns = [
            r"(?i)^REFERENCES?\s*$",
            r"(?i)^BIBLIOGRAPHY\s*$",
            r"(?i)^WORKS?\s+CITED\s*$",
            r"(?i)^LIST\s+OF\s+REFERENCES?\s*$",
        ]

        next_section_patterns = [
            r"(?i)^APPENDIX|APPENDICES\s*$",
            r"(?i)^LIST\s+OF\s+PUBLICATIONS?\s*$",
            r"(?i)^PUBLICATIONS?\s*$",
            r"(?i)^VITA\s*$",
        ]

        try:
            references_text = self._extract_section(
                self.full_text, reference_patterns, next_section_patterns
            )
            return references_text
        except Exception as e:
            print(f"Error extracting references section: {e}")
            return ""

    def analyze_citations(self, text: str) -> Dict:
        """Analyze in-text citations throughout the document"""
        citations = []
        for pattern_list in self.citation_patterns["in_text_citations"]:
            citations.extend(re.findall(pattern_list, text))

        # Extract years from citations
        years = []
        for citation in citations:
            year_match = re.search(r"\d{4}", citation)
            if year_match:
                years.append(int(year_match.group()))

        # Calculate recency score (based on last 10 years)
        current_year = datetime.now().year
        if years:
            avg_year = np.mean(years)
            recency_score = min(1.0, max(0, (avg_year - (current_year - 10)) / 10))
            consistency_score = 1.0 - (np.std(years) / 10 if len(years) > 1 else 0)
        else:
            recency_score = 0
            consistency_score = 0

        return {
            "count": len(citations),
            "unique_count": len(set(citations)),
            "years": years,
            "recency_score": recency_score,
            "consistency_score": consistency_score,
        }

    def analyze_references(self, text: str) -> Dict:
        """Analyze the reference list format and components"""
        # Split into individual references
        references = [
            ref.strip()
            for ref in text.split("\n")
            if any(
                re.match(pattern, ref.strip())
                for pattern in self.citation_patterns["reference_formats"]
            )
        ]

        # Analyze each reference for components
        complete_refs = 0
        components_found = {comp: 0 for comp in self.reference_components}

        for ref in references:
            components_in_ref = 0
            for component, pattern in self.reference_components.items():
                if re.search(pattern, ref):
                    components_found[component] += 1
                    components_in_ref += 1

            if (
                components_in_ref >= 4
            ):  # Consider reference complete if it has at least 4 components
                complete_refs += 1

        # Calculate format consistency
        format_patterns = {
            pattern: 0 for pattern in self.citation_patterns["reference_formats"]
        }
        for ref in references:
            for pattern in format_patterns:
                if re.match(pattern, ref):
                    format_patterns[pattern] += 1
                    break

        dominant_format = max(format_patterns.values()) if format_patterns else 0
        format_consistency = dominant_format / len(references) if references else 0

        return {
            "count": len(references),
            "complete_refs": complete_refs,
            "component_counts": components_found,
            "format_consistency": format_consistency,
        }

    def _evaluate_with_llm(self, references_text: str) -> Dict:
        """Use LLM to evaluate citation and reference quality"""
        prompt = f"""
        Evaluate these citations and references based on:
        1. Citation Quality (citation_score) (0-5):
           - Appropriate use of citations
           - Citation frequency
           - Citation relevance
        2. Reference Quality (reference_score) (0-5):
           - Format consistency
           - Reference completeness
           - Source reliability
        3. Overall Format (format_score) (0-5):
           - Adherence to academic standards
           - Consistency
           - Professional presentation
        
        Output Format:
        {{
          citation_score: float,
          reference_score: float,
          format_score: float,
          justification: str
        }}

        References section:
        {references_text[:2000]}...

        Provide scores and brief justification in JSON format.
        """

        try:
            response = self._get_llm_scores(prompt)
            return eval(response)
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                "citation_score": 0,
                "reference_score": 0,
                "format_score": 0,
                "justification": "Error in LLM evaluation",
            }

    def calculate_final_score(
        self, citation_analysis: Dict, reference_analysis: Dict, llm_scores: Dict = None
    ) -> Tuple[float, str]:
        """Calculate final score and determine grade"""
        # Calculate citation score
        citation_score = (
            min(1.0, citation_analysis["count"] / 50) * 0.4  # Citation count
            + citation_analysis["recency_score"] * 0.3  # Recency
            + citation_analysis["consistency_score"] * 0.3  # Consistency
        ) * 5  # Scale to 5 points

        # Calculate reference score
        reference_score = (
            min(1.0, reference_analysis["count"] / 30) * 0.3  # Reference count
            + (
                reference_analysis["complete_refs"]
                / max(1, reference_analysis["count"])
            )
            * 0.4  # Completeness
            + reference_analysis["format_consistency"] * 0.3  # Format consistency
        ) * 5  # Scale to 5 points

        if self.use_llm and llm_scores:
            # Combine with LLM scores
            llm_avg = (
                llm_scores["citation_score"]
                + llm_scores["reference_score"]
                + llm_scores["format_score"]
            ) / 3
            final_score = citation_score * 0.3 + reference_score * 0.3 + llm_avg * 0.4
        else:
            final_score = (citation_score + reference_score) / 2

        # Determine grade based on rubric
        if final_score >= 5:
            grade = "Distinction (5)"
        elif final_score >= 4:
            grade = "Distinction (4)"
        elif final_score >= 3:
            grade = "Merit (3)"
        elif final_score >= 2:
            grade = "Pass (2)"
        elif final_score >= 1:
            grade = "Fail (1)"
        else:
            grade = "Fail (0)"

        return final_score, grade

    def evaluate(self) -> Dict:
        """Perform complete evaluation of citations and references"""
        # Extract references section
        references_text = self.extract_references_section()
        if not references_text:
            return {
                "score": 0,
                "grade": "Fail (0)",
                "feedback": "References section not found",
            }

        # Analyze citations throughout the document
        citation_analysis = self.analyze_citations(self.full_text)

        # Analyze references section
        reference_analysis = self.analyze_references(references_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_llm(references_text)

        # Calculate final score
        score, grade = self.calculate_final_score(
            citation_analysis, reference_analysis, llm_scores
        )

        # Generate feedback
        feedback = []
        if citation_analysis["count"] < 30:
            feedback.append("Insufficient number of citations")
        if citation_analysis["recency_score"] < 0.5:
            feedback.append("Citations are not recent enough")
        if reference_analysis["format_consistency"] < 0.8:
            feedback.append("Inconsistent reference formatting")
        if (
            reference_analysis["complete_refs"] / max(1, reference_analysis["count"])
            < 0.8
        ):
            feedback.append("Many incomplete references")

        if llm_scores:
            feedback.append(f"LLM Analysis: {llm_scores.get('justification', '')}")

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "citation_analysis": {
                "count": citation_analysis["count"],
                "recency_score": float(round(citation_analysis["recency_score"], 2)),
                "consistency_score": float(
                    round(citation_analysis["consistency_score"], 2)
                ),
            },
            "reference_analysis": {
                "count": reference_analysis["count"],
                "complete_refs": reference_analysis["complete_refs"],
                "format_consistency": float(
                    round(reference_analysis["format_consistency"], 2)
                ),
            },
            "llm_scores": llm_scores,
            "feedback": ". ".join(feedback),
        }
