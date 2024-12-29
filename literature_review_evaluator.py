from evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
from collections import defaultdict


class LiteratureReviewEvaluator(BaseEvaluator):
    """Enhanced evaluator for thesis literature review with improved systematic review assessment"""

    def __init__(self, pdf_path, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)
        self.quality_indicators = self._initialize_quality_indicators()

    def extract_literature_review(self) -> str:
        """
        Extract the introduction section accurately using multiple extraction methods
        """
        return self.sections.get("literature_review", "")

    def _initialize_quality_indicators(self) -> Dict:
        """Initialize comprehensive quality indicators for systematic review assessment"""
        return {
            "systematic_approach": {
                "methodology": [
                    r"(?i)systematic(\s+literature)?\s+review",
                    r"(?i)search\s+strategy",
                    r"(?i)inclusion\s+criteria",
                    r"(?i)exclusion\s+criteria",
                    r"(?i)database[s]?\s+search",
                    r"(?i)search\s+string[s]?",
                ],
                "structure": [
                    r"(?i)chronological(\s+review)?",
                    r"(?i)thematic(\s+analysis)?",
                    r"(?i)systematic(\s+organization)?",
                    r"(?i)methodological(\s+review)?",
                ],
                "synthesis": [
                    r"(?i)synthesis\s+of",
                    r"(?i)integrate[d]?\s+findings",
                    r"(?i)critical\s+analysis",
                    r"(?i)comparative\s+analysis",
                ],
            },
            "evidence_quality": {
                "source_types": [
                    r"(?i)journal\s+article[s]?",
                    r"(?i)conference\s+paper[s]?",
                    r"(?i)book\s+chapter[s]?",
                    r"(?i)technical\s+report[s]?",
                    r"(?i)thesis|dissertation[s]?",
                ],
                "authority": [
                    r"(?i)peer[\s-]reviewed",
                    r"(?i)impact\s+factor",
                    r"(?i)cited\s+by",
                    r"(?i)seminal\s+work",
                ],
                "relevance": [
                    r"(?i)relevant\s+to",
                    r"(?i)addresses\s+the\s+research",
                    r"(?i)pertinent\s+to",
                    r"(?i)applicable\s+to",
                ],
            },
            "critical_analysis": {
                "comparison": [
                    r"(?i)compare[d]?\s+to",
                    r"(?i)in\s+contrast\s+to",
                    r"(?i)similar\s+to",
                    r"(?i)differs?\s+from",
                ],
                "evaluation": [
                    r"(?i)strength[s]?",
                    r"(?i)limitation[s]?",
                    r"(?i)weakness[es]?",
                    r"(?i)advantage[s]?",
                    r"(?i)disadvantage[s]?",
                ],
                "gaps": [
                    r"(?i)gap[s]?\s+in(\s+the)?\s+literature",
                    r"(?i)future\s+research",
                    r"(?i)unexplored",
                    r"(?i)needs?\s+further",
                ],
            },
            "depth_indicators": {
                "theoretical": [
                    r"(?i)theoretical\s+framework",
                    r"(?i)conceptual\s+framework",
                    r"(?i)underlying\s+theory",
                    r"(?i)theoretical\s+foundation",
                ],
                "methodology": [
                    r"(?i)methodological\s+approach",
                    r"(?i)research\s+design",
                    r"(?i)experimental\s+setup",
                    r"(?i)study\s+design",
                ],
                "findings": [
                    r"(?i)key\s+finding[s]?",
                    r"(?i)result[s]?\s+show",
                    r"(?i)concluded\s+that",
                    r"(?i)demonstrate[d]?\s+that",
                ],
            },
        }

    def _analyze_citation_network(self, text: str) -> Dict:
        """Analyze the citation network and relationships"""
        citation_patterns = {
            "author_year": r"\((?:\w+(?:\s+and\s+\w+)?(?:\s+et\s+al\.?)?)?,\s*\d{4}\)",
            "numeric": r"\[[\d,\s-]+\]",
            "integrated": r"(?:(?:\w+(?:\s+and\s+\w+)?(?:\s+et\s+al\.?)?)\s+\(\d{4}\))",
        }

        citations = defaultdict(list)
        for style, pattern in citation_patterns.items():
            citations[style].extend(re.findall(pattern, text))

        # Extract years and analyze temporal distribution
        years = []
        for style_citations in citations.values():
            for citation in style_citations:
                year_match = re.search(r"\d{4}", citation)
                if year_match:
                    years.append(int(year_match.group()))

        current_year = datetime.now().year

        # Calculate citation metrics
        metrics = {
            "total_citations": sum(len(cites) for cites in citations.values()),
            "unique_citations": len(
                set().union(*[set(cites) for cites in citations.values()])
            ),
            "citation_styles": sum(1 for style, cites in citations.items() if cites),
            "years_range": max(years) - min(years) if years else 0,
            "recency_score": self._calculate_recency_score(years, current_year),
            "temporal_distribution": self._calculate_temporal_distribution(
                years, current_year
            ),
        }

        return metrics

    def _calculate_recency_score(self, years: List[int], current_year: int) -> float:
        """Calculate the recency score of citations"""
        if not years:
            return 0.0

        weights = [
            (
                1.0
                if year >= current_year - 5
                else (
                    0.8
                    if year >= current_year - 7
                    else 0.6 if year >= current_year - 10 else 0.4
                )
            )
            for year in years
        ]

        return sum(weights) / len(weights)

    def _calculate_temporal_distribution(
        self, years: List[int], current_year: int
    ) -> float:
        """Calculate how well citations are distributed across time periods"""
        if not years:
            return 0.0

        periods = defaultdict(int)
        for year in years:
            period = (current_year - year) // 3  # 3-year periods
            periods[period] += 1

        # Calculate distribution evenness
        total_periods = len(periods)
        if total_periods <= 1:
            return 0.5

        expected_per_period = len(years) / total_periods
        variance = sum((count - expected_per_period) ** 2 for count in periods.values())

        # Normalize the distribution score
        return 1.0 / (1.0 + variance / len(years))

    def _evaluate_systematic_approach(self, text: str) -> Tuple[float, Dict]:
        """Evaluate the systematic approach of the literature review"""
        scores = {}
        evidence = defaultdict(list)

        # Evaluate each quality indicator category
        for category, indicators in self.quality_indicators.items():
            category_scores = {}
            for aspect, patterns in indicators.items():
                matches = 0
                for pattern in patterns:
                    found_matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in found_matches:
                        context = text[
                            max(0, match.start() - 50) : min(
                                len(text), match.end() + 50
                            )
                        ]
                        matches += 1
                        evidence[f"{category}_{aspect}"].append(context.strip())

                # Normalize score for this aspect
                category_scores[aspect] = min(1.0, matches / len(patterns))

            scores[category] = np.mean(list(category_scores.values()))

        # Calculate weighted final score
        weights = {
            "systematic_approach": 0.3,
            "evidence_quality": 0.25,
            "critical_analysis": 0.25,
            "depth_indicators": 0.2,
        }

        final_score = sum(
            score * weights[category] for category, score in scores.items()
        )
        return final_score * 15, dict(evidence)  # Scale to 15 points

    def _evaluate_with_enhanced_llm(self, text: str) -> Dict:
        """Enhanced LLM evaluation with specific focus on systematic review criteria"""
        prompt = f"""
        Evaluate this literature review based on these detailed criteria:
        
        1. Systematic Approach (0-15):
           - Clear methodology for literature selection
           - Structured organization of review
           - Comprehensive coverage of the field
           - Evidence of systematic search strategy
        
        2. Evidence Quality (0-15):
           - Use of high-quality sources
           - Range of source types
           - Authority of sources
           - Currency and relevance
        
        3. Critical Analysis (0-15):
           - Depth of analysis
           - Comparison between sources
           - Identification of gaps
           - Synthesis of findings
        
        4. Research Integration (0-15):
           - Connection to research objectives
           - Theoretical framework development
           - Research gap identification
           - Future research directions

        Literature review text:
        {text[:4000]}...

        Output Format:
        {{
            "systematic_approach": float,
            "evidence_quality": float,
            "critical_analysis": float,
            "research_integration": float,
            "justification": str,
            "strengths": [str],
            "improvements": [str],
        }}
        
        Return in JSON format.
        """

        try:
            response = self._get_llm_scores(prompt)
            return eval(response)
        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return {
                "systematic_approach": 0,
                "evidence_quality": 0,
                "critical_analysis": 0,
                "research_integration": 0,
                "justification": "Error in LLM evaluation",
                "strengths": [],
                "improvements": [],
            }

    def _calculate_enhanced_final_score(
        self, systematic_score: float, citation_metrics: Dict, llm_scores: Dict = None
    ) -> Tuple[float, str, List[str]]:
        """Calculate final score with detailed feedback based on rubric criteria"""
        # Calculate citation quality score (0-15)
        citation_score = (
            min(1.0, citation_metrics["total_citations"] / 50) * 5
            + citation_metrics["recency_score"] * 5
            + citation_metrics["temporal_distribution"] * 5
        )

        if self.use_llm and llm_scores:
            # Combine scores with weights
            final_score = (
                systematic_score * 0.4
                + citation_score * 0.3
                + np.mean(
                    [
                        llm_scores["systematic_approach"],
                        llm_scores["evidence_quality"],
                        llm_scores["critical_analysis"],
                        llm_scores["research_integration"],
                    ]
                )
                * 0.3
            )
        else:
            final_score = systematic_score * 0.6 + citation_score * 0.4

        # Generate detailed feedback
        feedback = []

        # Determine grade and feedback based on rubric
        if final_score >= 14:
            grade = "Distinction (14-15)"
            feedback.append("Outstanding systematic review with comprehensive evidence")
        elif final_score >= 11:
            grade = "Distinction (11-13)"
            feedback.append("Well-executed review with minor gaps in coverage")
        elif final_score >= 9:
            grade = "Merit (9-10)"
            feedback.append("Good review but lacks depth in critical analysis")
        elif final_score >= 8:
            grade = "Pass (8)"
            feedback.append("Basic review with incomplete systematic approach")
        elif final_score >= 4:
            grade = "Fail (4-7)"
            feedback.append("Insufficient literature coverage and analysis")
        else:
            grade = "Fail (0-3)"
            feedback.append("Inadequate systematic approach and evidence")

        # Add specific feedback based on metrics
        if citation_metrics["total_citations"] < 30:
            feedback.append(
                f"Limited number of citations ({citation_metrics['total_citations']})"
            )
        if citation_metrics["recency_score"] < 0.6:
            feedback.append("Need more recent sources")
        if citation_metrics["temporal_distribution"] < 0.5:
            feedback.append("Improve temporal distribution of sources")

        return final_score, grade, feedback

    def evaluate(self) -> Dict:
        """Perform enhanced evaluation of the literature review"""
        # Extract literature review
        lit_review_text = self.extract_literature_review()
        if not lit_review_text:
            return {
                "score": 0,
                "grade": "Fail (0-3)",
                "feedback": ["Literature review section not found"],
                "details": {},
            }

        # Analyze systematic approach and evidence
        systematic_score, evidence_details = self._evaluate_systematic_approach(
            lit_review_text
        )

        # Analyze citation network
        citation_metrics = self._analyze_citation_network(lit_review_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_enhanced_llm(lit_review_text)

        # Calculate final score and generate feedback
        score, grade, feedback = self._calculate_enhanced_final_score(
            systematic_score, citation_metrics, llm_scores
        )

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "feedback": feedback,
            "systematic_score": float(round(systematic_score, 2)),
            "citation_metrics": {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in citation_metrics.items()
            },
            "llm_scores": llm_scores,
            "evidence_details": evidence_details,
        }
