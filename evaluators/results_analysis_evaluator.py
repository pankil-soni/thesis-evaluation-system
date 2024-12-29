from evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


class ResultsAnalysisEvaluator(BaseEvaluator):
    """Enhanced evaluator for thesis results and analysis with improved assessment criteria"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)
        self.quality_indicators = self._initialize_quality_indicators()

    def extract_results_section(self) -> str:
        """Extract the results and discussion section"""
        return self.sections.get("results","")

    def _initialize_quality_indicators(self) -> Dict:
        """Initialize comprehensive quality indicators for results analysis assessment"""
        return {
            "results_evidence": {
                "quantitative": [
                    r"(?i)statistical(\s+analysis)?",
                    r"(?i)numerical\s+results?",
                    r"(?i)data\s+analysis",
                    r"(?i)(mean|average|median|mode)",
                    r"(?i)standard\s+deviation",
                    r"(?i)variance",
                    r"(?i)significance\s+level",
                    r"(?i)p[\s-]value",
                ],
                "visualization": [
                    r"(?i)figure\s+\d+",
                    r"(?i)table\s+\d+",
                    r"(?i)graph\s+shows",
                    r"(?i)plot\s+illustrates",
                    r"(?i)chart\s+depicts",
                    r"(?i)visualization\s+demonstrates",
                    r"(?i)as\s+shown\s+in\s+(figure|table)",
                ],
                "qualitative": [
                    r"(?i)qualitative\s+analysis",
                    r"(?i)thematic\s+analysis",
                    r"(?i)content\s+analysis",
                    r"(?i)participant\s+responses?",
                    r"(?i)interview\s+results?",
                    r"(?i)observation\s+findings?",
                ],
            },
            "critical_analysis": {
                "comparison": [
                    r"(?i)compared?\s+to",
                    r"(?i)in\s+contrast\s+to",
                    r"(?i)higher/lower\s+than",
                    r"(?i)better/worse\s+than",
                    r"(?i)differs?\s+from",
                    r"(?i)similar\s+to",
                    r"(?i)unlike",
                ],
                "interpretation": [
                    r"(?i)this\s+(suggests|indicates|implies|shows)",
                    r"(?i)these\s+results?\s+(suggest|indicate|imply|show)",
                    r"(?i)interpretation\s+of",
                    r"(?i)meaning\s+of",
                    r"(?i)significance\s+of",
                    r"(?i)importance\s+of",
                ],
                "causation": [
                    r"(?i)because(\s+of)?",
                    r"(?i)due\s+to",
                    r"(?i)as\s+a\s+result\s+of",
                    r"(?i)consequently",
                    r"(?i)therefore",
                    r"(?i)thus",
                    r"(?i)hence",
                ],
            },
            "discussion_depth": {
                "literature_integration": [
                    r"(?i)previous\s+studies?",
                    r"(?i)existing\s+literature",
                    r"(?i)prior\s+research",
                    r"(?i)consistent\s+with",
                    r"(?i)aligns?\s+with",
                    r"(?i)supports?\s+findings?\s+of",
                    r"(?i)contradicts?\s+findings?\s+of",
                ],
                "implications": [
                    r"(?i)implications?\s+for",
                    r"(?i)impact\s+on",
                    r"(?i)contribution\s+to",
                    r"(?i)practical\s+applications?",
                    r"(?i)theoretical\s+implications?",
                    r"(?i)potential\s+benefits?",
                ],
                "limitations": [
                    r"(?i)limitation(s)?",
                    r"(?i)constraint(s)?",
                    r"(?i)drawback(s)?",
                    r"(?i)shortcoming(s)?",
                    r"(?i)weakness(es)?",
                    r"(?i)future\s+research",
                ],
            },
        }

    def _evaluate_analysis_quality(self, text: str) -> Tuple[float, Dict]:
        """
        Evaluate the quality of results analysis using weighted criteria
        """
        scores = {}
        details = defaultdict(lambda: defaultdict(list))

        for category, subcategories in self.quality_indicators.items():
            category_scores = {}

            for subcategory, patterns in subcategories.items():
                matches = 0
                total_patterns = len(patterns)

                for pattern in patterns:
                    found_matches = re.finditer(pattern, text)
                    for match in found_matches:
                        context = text[
                            max(0, match.start() - 50) : min(
                                len(text), match.end() + 50
                            )
                        ]
                        matches += 1
                        details[category][subcategory].append(context.strip())

                # Calculate normalized score for this subcategory
                category_scores[subcategory] = min(1.0, matches / total_patterns)

            # Calculate weighted average for category
            subcategory_weights = {
                "quantitative": 0.4,
                "visualization": 0.3,
                "qualitative": 0.3,
                "comparison": 0.35,
                "interpretation": 0.35,
                "causation": 0.3,
                "literature_integration": 0.35,
                "implications": 0.35,
                "limitations": 0.3,
            }

            category_weight = sum(
                subcategory_weights.get(subcat, 0.33) for subcat in subcategories.keys()
            )
            scores[category] = (
                sum(
                    score * subcategory_weights.get(subcat, 0.33)
                    for subcat, score in category_scores.items()
                )
                / category_weight
            )

        # Calculate overall weighted score
        category_weights = {
            "results_evidence": 0.35,
            "critical_analysis": 0.35,
            "discussion_depth": 0.3,
        }

        total_weight = sum(category_weights.values())
        weighted_score = (
            sum(
                score * category_weights[category] for category, score in scores.items()
            )
            / total_weight
        )

        return weighted_score * 15, dict(details)  # Scale to 0-15

    def _analyze_argumentation_quality(self, text: str) -> float:
        """
        Analyze the quality of argumentation in the discussion
        """
        argumentation_patterns = {
            "logical_flow": [
                r"(?i)first(ly)?",
                r"(?i)second(ly)?",
                r"(?i)third(ly)?",
                r"(?i)finally",
                r"(?i)moreover",
                r"(?i)furthermore",
                r"(?i)in\s+addition",
            ],
            "critical_thinking": [
                r"(?i)however",
                r"(?i)nevertheless",
                r"(?i)although",
                r"(?i)despite",
                r"(?i)while",
                r"(?i)whereas",
                r"(?i)on\s+the\s+other\s+hand",
            ],
            "evidence_based": [
                r"(?i)evidence\s+suggests",
                r"(?i)data\s+shows",
                r"(?i)results\s+indicate",
                r"(?i)analysis\s+reveals",
                r"(?i)findings\s+demonstrate",
            ],
        }

        scores = []
        for category, patterns in argumentation_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, text))
            scores.append(min(1.0, matches / len(patterns)))

        return np.mean(scores)

    def _evaluate_with_enhanced_llm(self, text: str) -> Dict:
        """Enhanced LLM evaluation with specific results analysis criteria"""
        prompt = f"""
        Evaluate this results and analysis section based on these detailed criteria:
        
        1. Results Quality (0-15):
           - Comprehensive presentation of findings
           - Effective use of evidence
           - Clear data presentation
           - Appropriate use of visualizations
        
        2. Analysis Depth (0-15):
           - Critical interpretation
           - Comparative analysis
           - Causal relationships
           - Statistical significance
        
        3. Discussion Quality (0-15):
           - Integration with literature
           - Theoretical implications
           - Practical implications
           - Limitations addressed
        
        4. Argumentation (0-15):
           - Logical flow
           - Evidence-based reasoning
           - Critical thinking
           - Balanced perspective

        Results text:
        {text[:4000]}...

        Output Format:

        {{
            "results_quality": float,
            "analysis_depth": float,
            "discussion_quality": float,
            "argumentation": float,
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
                "results_quality": 0,
                "analysis_depth": 0,
                "discussion_quality": 0,
                "argumentation": 0,
                "justification": "Error in LLM evaluation",
                "strengths": [],
                "improvements": [],
            }

    def _calculate_enhanced_final_score(
        self, analysis_score: float, argumentation_score: float, llm_scores: Dict = None
    ) -> Tuple[float, str, List[str]]:
        """Calculate final score with detailed feedback based on rubric criteria"""
        if self.use_llm and llm_scores:
            # Combine scores with weights
            final_score = (
                analysis_score * 0.4
                + argumentation_score * 0.2
                + np.mean(
                    [
                        llm_scores["results_quality"],
                        llm_scores["analysis_depth"],
                        llm_scores["discussion_quality"],
                        llm_scores["argumentation"],
                    ]
                )
                * 0.4
            )
        else:
            final_score = analysis_score * 0.7 + argumentation_score * 15 * 0.3

        # Generate detailed feedback
        feedback = []

        # Determine grade and feedback based on rubric
        if final_score >= 14:
            grade = "Distinction (14-15)"
            feedback.append(
                "Outstanding evidence and critical analysis with comprehensive discussion"
            )
        elif final_score >= 11:
            grade = "Distinction (11-13)"
            feedback.append("Very good results with strong critical analysis")
        elif final_score >= 9:
            grade = "Merit (9-10)"
            feedback.append(
                "Appropriate analysis but needs improvement in critical depth"
            )
        elif final_score >= 8:
            grade = "Pass (8)"
            feedback.append("Satisfactory results but analysis is limited")
        elif final_score >= 4:
            grade = "Fail (4-7)"
            feedback.append("Discussion is unsatisfactory and lacks critical depth")
        else:
            grade = "Fail (0-3)"
            feedback.append("Insufficient evidence of analysis and discussion")

        # Add argumentation-specific feedback
        if argumentation_score < 0.6:
            feedback.append(
                "Improve critical thinking and evidence-based argumentation"
            )

        return final_score, grade, feedback

    def evaluate(self) -> Dict:
        """Perform enhanced evaluation of the results and analysis section"""
        # Extract results section
        results_text = self.extract_results_section()
        if not results_text:
            return {
                "score": 0,
                "grade": "Fail (0-3)",
                "feedback": ["Results and analysis section not found"],
                "details": {},
            }

        # Evaluate analysis quality
        analysis_score, quality_details = self._evaluate_analysis_quality(results_text)

        # Analyze argumentation quality
        argumentation_score = self._analyze_argumentation_quality(results_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_enhanced_llm(results_text)

        # Calculate final score and generate feedback
        score, grade, feedback = self._calculate_enhanced_final_score(
            analysis_score, argumentation_score, llm_scores
        )

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "feedback": feedback,
            "analysis_score": float(round(analysis_score, 2)),
            "argumentation_score": float(round(argumentation_score * 15, 2)),
            "llm_scores": llm_scores,
            "quality_details": quality_details,
        }
