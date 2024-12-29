from evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict
from datetime import datetime


class CitationReferencesEvaluator(BaseEvaluator):
    """Enhanced evaluator for thesis citations and references with improved assessment criteria"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)
        self.quality_indicators = self._initialize_quality_indicators()

    def extract_references_section(self) -> str:
        """Extract the references section"""
        return self.sections.get("references", "")

    def _initialize_quality_indicators(self) -> Dict:
        """Initialize comprehensive quality indicators for citation assessment"""
        return {
            "citation_style": {
                "author_year": [
                    r"\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}\)",  # (Smith et al., 2020)
                    r"\([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,\s*\d{4}\)",  # (Smith and Jones, 2020)
                    r"[A-Z][a-z]+(?:\s+et\s+al\.?)?\s+\(\d{4}\)",  # Smith et al. (2020)
                    r"[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s+\(\d{4}\)",  # Smith and Jones (2020)
                ],
                "numeric": [
                    r"\[\d+\]",  # [1]
                    r"\[\d+(?:,\s*\d+)*\]",  # [1,2,3]
                    r"\[\d+(?:\s*-\s*\d+)\]",  # [1-3]
                    r"\(\d+\)",  # (1)
                    r"(?<!\d)\d+(?:,\s*\d+)*(?!\d)",  # 1,2,3 (standalone)
                ],
            },
            "reference_format": {
                "journal_article": [
                    r"^[A-Z][^.]*\.\s*\(\d{4}\)\.\s*[^.]+\.\s*[A-Z][A-Za-z\s&]+\s*,\s*\d+",
                    r"^[A-Z][^.]*\.\s*\(\d{4}\)\.\s*[^.]+\.\s*[A-Z][A-Za-z\s&]+\s*\d+\s*\(\d+\)",
                ],
                "book": [
                    r"^[A-Z][^.]*\.\s*\(\d{4}\)\.\s*[^.]+\.\s*[A-Z][A-Za-z\s&:]+\s*Press",
                    r"^[A-Z][^.]*\.\s*\(\d{4}\)\.\s*[^.]+\.\s*[A-Z][A-Za-z\s&:]+\s*Publisher",
                ],
                "conference": [
                    r"^[A-Z][^.]*\.\s*\(\d{4}\)\.\s*[^.]+\.\s*In\s*[^.]+Conference",
                    r"^[A-Z][^.]*\.\s*\(\d{4}\)\.\s*[^.]+\.\s*Proceedings\s+of",
                ],
            },
            "source_quality": {
                "high_impact": [
                    r"Nature",
                    r"Science",
                    r"IEEE\s+Transactions",
                    r"ACM\s+Transactions",
                    r"Journal\s+of\s+(?:the\s+)?[A-Z]",
                ],
                "peer_reviewed": [
                    r"Journal\s+of",
                    r"International\s+Journal",
                    r"Transactions\s+on",
                    r"Review[s]?\s+of",
                ],
                "conferences": [
                    r"Proceedings\s+of\s+the",
                    r"International\s+Conference",
                    r"Symposium\s+on",
                    r"Workshop\s+on",
                ],
            },
        }

    def _analyze_citation_style_consistency(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze the consistency of citation style usage
        """
        style_counts = defaultdict(int)
        style_examples = defaultdict(list)
        total_citations = 0

        for style, patterns in self.quality_indicators["citation_style"].items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    citation = match.group()
                    style_counts[style] += 1
                    if (
                        len(style_examples[style]) < 3
                    ):  # Keep up to 3 examples per style
                        style_examples[style].append(citation)
                    total_citations += 1

        # Calculate consistency score
        dominant_style = max(style_counts.values()) if style_counts else 0
        consistency_score = dominant_style / max(1, total_citations)

        return consistency_score, {
            "counts": dict(style_counts),
            "examples": dict(style_examples),
            "total": total_citations,
        }

    def _analyze_reference_quality(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze the quality and completeness of references
        """
        format_scores = defaultdict(int)
        quality_scores = defaultdict(int)
        details = defaultdict(list)

        # Split into individual references
        references = [ref.strip() for ref in text.split("\n") if ref.strip()]
        total_refs = len(references)

        for ref in references:
            # Check format quality
            for format_type, patterns in self.quality_indicators[
                "reference_format"
            ].items():
                for pattern in patterns:
                    if re.search(pattern, ref):
                        format_scores[format_type] += 1
                        details[f"{format_type}_examples"].append(
                            ref[:100]
                        )  # First 100 chars
                        break

            # Check source quality
            for quality_type, patterns in self.quality_indicators[
                "source_quality"
            ].items():
                for pattern in patterns:
                    if re.search(pattern, ref):
                        quality_scores[quality_type] += 1
                        details[f"{quality_type}_examples"].append(ref[:100])
                        break

        # Calculate overall quality score
        format_score = sum(format_scores.values()) / max(1, total_refs)
        quality_score = sum(quality_scores.values()) / max(1, total_refs)

        return (format_score + quality_score) / 2, {
            "format_scores": dict(format_scores),
            "quality_scores": dict(quality_scores),
            "details": dict(details),
            "total_refs": total_refs,
        }

    def _analyze_temporal_distribution(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze the temporal distribution of citations
        """
        years = []
        year_counts = defaultdict(int)
        current_year = datetime.now().year

        # Extract years from text
        year_pattern = r"\b(19|20)\d{2}\b"
        matches = re.finditer(year_pattern, text)

        for match in matches:
            year = int(match.group())
            if 1900 <= year <= current_year:  # Validate year
                years.append(year)
                year_counts[year] += 1

        if not years:
            return 0.0, {"years": {}, "metrics": {}}

        # Calculate metrics
        avg_year = np.mean(years)
        recency_score = min(1.0, max(0, (avg_year - (current_year - 10)) / 10))

        # Calculate distribution metrics
        year_range = max(years) - min(years)
        distribution_score = min(1.0, year_range / 20)  # Normalize to max 20 years span

        recent_count = sum(1 for y in years if y >= current_year - 5)
        recent_ratio = recent_count / len(years)

        return (recency_score + distribution_score + recent_ratio) / 3, {
            "years": dict(year_counts),
            "metrics": {
                "average_year": float(round(avg_year, 2)),
                "recency_score": float(round(recency_score, 2)),
                "distribution_score": float(round(distribution_score, 2)),
                "recent_ratio": float(round(recent_ratio, 2)),
            },
        }

    def _evaluate_with_enhanced_llm(self, text: str) -> Dict:
        """Enhanced LLM evaluation with specific citation criteria"""
        prompt = f"""
        Evaluate these citations and references based on these detailed criteria:
        
        1. Citation Accuracy (0-5):
           - Correct citation format
           - Appropriate citation placement
           - Citation-reference matching
           - Style consistency
        
        2. Source Quality (0-5):
           - Source reliability
           - Academic credibility
           - Peer-review status
           - Publisher reputation
        
        3. Temporal Distribution (0-5):
           - Recency of sources
           - Balance of historical and current
           - Coverage of development
           - Currency of research
        
        4. Format Adherence (0-5):
           - Style guide compliance
           - Format consistency
           - Reference completeness
           - Professional presentation

        References section:
        {text[:4000]}...

        Output Format:

        {{
            "citation_accuracy": float,
            "source_quality": float,
            "temporal_distribution": float,
            "format_adherence": float,
            "justification": str,
            "strengths": [str],
            "improvements": [str]
        }}
        
        Return in JSON format.
        """

        try:
            response = self._get_llm_scores(prompt)
            return eval(response)
        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return {
                "citation_accuracy": 0,
                "source_quality": 0,
                "temporal_distribution": 0,
                "format_adherence": 0,
                "justification": "Error in LLM evaluation",
                "strengths": [],
                "improvements": [],
            }

    def _calculate_enhanced_final_score(
        self,
        style_consistency: float,
        reference_quality: float,
        temporal_score: float,
        llm_scores: Dict = None,
    ) -> Tuple[float, str, List[str]]:
        """Calculate final score with detailed feedback based on rubric criteria"""
        if self.use_llm and llm_scores:
            # Combine scores with weights
            final_score = (
                style_consistency * 0.25
                + reference_quality * 0.25
                + temporal_score * 0.2
                + np.mean(
                    [
                        llm_scores["citation_accuracy"],
                        llm_scores["source_quality"],
                        llm_scores["temporal_distribution"],
                        llm_scores["format_adherence"],
                    ]
                )
                / 5
                * 0.3  # Scale LLM scores to 0-1
            ) * 5  # Scale to 0-5
        else:
            final_score = (
                style_consistency * 0.35
                + reference_quality * 0.35
                + temporal_score * 0.3
            ) * 5  # Scale to 0-5

        # Generate detailed feedback
        feedback = []

        # Determine grade and feedback based on rubric
        if final_score >= 5:
            grade = "Distinction (5)"
            feedback.append(
                "Accurate, professional citations from recent and reliable sources"
            )
        elif final_score >= 4:
            grade = "Distinction (4)"
            feedback.append(
                "Generally correct citations with mostly recent and reliable references"
            )
        elif final_score >= 3:
            grade = "Merit (3)"
            feedback.append("Mostly correct citations with some outdated sources")
        elif final_score >= 2:
            grade = "Pass (2)"
            feedback.append("Acceptable citations but lacking high-quality sources")
        elif final_score >= 1:
            grade = "Fail (1)"
            feedback.append("Inadequate citations with outdated references")
        else:
            grade = "Fail (0)"
            feedback.append("Very poor citations and references")

        # Add specific feedback based on scores
        if style_consistency < 0.7:
            feedback.append("Improve citation style consistency")
        if reference_quality < 0.7:
            feedback.append("Enhance reference quality and completeness")
        if temporal_score < 0.6:
            feedback.append("Include more recent sources")

        return final_score, grade, feedback

    def evaluate(self) -> Dict:
        """Perform enhanced evaluation of citations and references"""
        # Extract references section
        references_text = self.extract_references_section()
        if not references_text:
            return {
                "score": 0,
                "grade": "Fail (0)",
                "feedback": ["References section not found"],
                "details": {},
            }

        # Analyze citation style consistency
        style_consistency, style_details = self._analyze_citation_style_consistency(
            references_text
        )

        # Analyze reference quality
        reference_quality, quality_details = self._analyze_reference_quality(
            references_text
        )

        # Analyze temporal distribution
        temporal_score, temporal_details = self._analyze_temporal_distribution(
            references_text
        )

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_enhanced_llm(references_text)

        # Calculate final score and generate feedback
        score, grade, feedback = self._calculate_enhanced_final_score(
            style_consistency, reference_quality, temporal_score, llm_scores
        )

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "feedback": feedback,
            "style_consistency": float(round(style_consistency * 5, 2)),
            "reference_quality": float(round(reference_quality * 5, 2)),
            "temporal_score": float(round(temporal_score * 5, 2)),
            "llm_scores": llm_scores,
            "details": {
                "style": style_details,
                "quality": quality_details,
                "temporal": temporal_details,
            },
        }
