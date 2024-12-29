from evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


class ResearchMethodsEvaluator(BaseEvaluator):
    """Enhanced evaluator for thesis research methodology with improved assessment criteria"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)
        self.quality_indicators = self._initialize_quality_indicators()

    def extract_methodology(self) -> str:
        """Extract the methodology section using patterns"""
        return self.sections.get("methodology", "")

    def _initialize_quality_indicators(self) -> Dict:
        """Initialize comprehensive quality indicators for methodology assessment"""
        return {
            "research_design": {
                "approach": [
                    r"(?i)research\s+design",
                    r"(?i)research\s+approach",
                    r"(?i)research\s+strategy",
                    r"(?i)research\s+framework",
                    r"(?i)research\s+methodology",
                ],
                "method_type": [
                    r"(?i)quantitative(\s+research)?",
                    r"(?i)qualitative(\s+research)?",
                    r"(?i)mixed[\s-]method(s)?",
                    r"(?i)experimental(\s+design)?",
                    r"(?i)quasi[\s-]experimental",
                ],
                "justification": [
                    r"(?i)chosen\s+because",
                    r"(?i)selected\s+(due|because|for)",
                    r"(?i)justification\s+for",
                    r"(?i)rationale\s+for",
                    r"(?i)this\s+approach\s+is\s+appropriate",
                ],
            },
            "data_collection": {
                "methods": [
                    r"(?i)data\s+collection(\s+method(s)?)?",
                    r"(?i)survey(\s+design)?",
                    r"(?i)interview(\s+protocol)?",
                    r"(?i)observation(\s+technique)?",
                    r"(?i)experiment(al)?\s+setup",
                ],
                "sampling": [
                    r"(?i)sampling(\s+strategy)?",
                    r"(?i)sample\s+size",
                    r"(?i)participant(s)?(\s+selection)?",
                    r"(?i)population(\s+selection)?",
                    r"(?i)inclusion(\s+criteria)?",
                ],
                "instruments": [
                    r"(?i)research\s+instrument(s)?",
                    r"(?i)measurement\s+tool(s)?",
                    r"(?i)questionnaire(\s+design)?",
                    r"(?i)survey\s+instrument",
                    r"(?i)data\s+collection\s+tool(s)?",
                ],
            },
            "data_analysis": {
                "techniques": [
                    r"(?i)data\s+analysis(\s+technique(s)?)?",
                    r"(?i)statistical(\s+analysis)?",
                    r"(?i)thematic(\s+analysis)?",
                    r"(?i)content(\s+analysis)?",
                    r"(?i)analysis\s+method(s)?",
                ],
                "procedures": [
                    r"(?i)analysis\s+procedure(s)?",
                    r"(?i)coding(\s+process)?",
                    r"(?i)data\s+processing",
                    r"(?i)analytical\s+framework",
                    r"(?i)analysis\s+approach",
                ],
                "tools": [
                    r"(?i)software(\s+tool(s)?)?",
                    r"(?i)statistical\s+package",
                    r"(?i)analysis\s+tool(s)?",
                    r"(?i)data\s+analysis\s+software",
                    r"(?i)analytical\s+tool(s)?",
                ],
            },
            "validity_reliability": {
                "validity": [
                    r"(?i)validity(\s+measure(s)?)?",
                    r"(?i)internal\s+validity",
                    r"(?i)external\s+validity",
                    r"(?i)construct\s+validity",
                    r"(?i)face\s+validity",
                ],
                "reliability": [
                    r"(?i)reliability(\s+measure(s)?)?",
                    r"(?i)test[\s-]retest",
                    r"(?i)inter[\s-]rater",
                    r"(?i)cronbach'?s\s+alpha",
                    r"(?i)consistency(\s+measure(s)?)?",
                ],
                "limitations": [
                    r"(?i)limitation(s)?(\s+of)?(\s+the)?(\s+study)?",
                    r"(?i)constraint(s)?",
                    r"(?i)potential\s+bias(es)?",
                    r"(?i)methodological\s+limitation(s)?",
                    r"(?i)scope(\s+and\s+limitations)?",
                ],
            },
        }

    def _evaluate_methodology_quality(self, text: str) -> Tuple[float, Dict]:
        """
        Evaluate the quality of methodology using weighted criteria
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
                "approach": 0.4,
                "method_type": 0.3,
                "justification": 0.3,
                "methods": 0.35,
                "sampling": 0.35,
                "instruments": 0.3,
                "techniques": 0.4,
                "procedures": 0.3,
                "tools": 0.3,
                "validity": 0.35,
                "reliability": 0.35,
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
            "research_design": 0.3,
            "data_collection": 0.25,
            "data_analysis": 0.25,
            "validity_reliability": 0.2,
        }

        total_weight = sum(category_weights.values())
        weighted_score = (
            sum(
                score * category_weights[category] for category, score in scores.items()
            )
            / total_weight
        )

        return weighted_score * 20, dict(details)  # Scale to 0-20

    def _analyze_methodology_structure(self, text: str) -> float:
        """
        Analyze the structural coherence and organization of the methodology
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return 0.0

        # Expected methodology flow patterns
        flow_patterns = [
            (r"(?i)research\s+design|approach|strategy", 0),
            (r"(?i)data\s+collection|sampling|participants", 1),
            (r"(?i)data\s+analysis|analytical|processing", 2),
            (r"(?i)validity|reliability|limitations", 3),
        ]

        # Score based on proper ordering
        flow_score = 0
        total_patterns = len(flow_patterns)

        for i, paragraph in enumerate(paragraphs):
            normalized_position = i / len(paragraphs)

            for pattern, expected_position in flow_patterns:
                if re.search(pattern, paragraph):
                    expected_normalized = expected_position / total_patterns
                    position_score = 1 - min(
                        1, abs(normalized_position - expected_normalized)
                    )
                    flow_score += position_score

        return min(1.0, flow_score / total_patterns)

    def _evaluate_with_enhanced_llm(self, text: str) -> Dict:
        """Enhanced LLM evaluation with specific methodology criteria"""
        prompt = f"""
        Evaluate this research methodology based on these detailed criteria:
        
        1. Research Design (0-20):
           - Clear articulation of research approach
           - Justification of chosen methods
           - Alignment with research objectives
           - Appropriateness of design
        
        2. Data Collection (0-20):
           - Clarity of data collection methods
           - Appropriateness of sampling strategy
           - Description of instruments/tools
           - Consideration of data quality
        
        3. Data Analysis (0-20):
           - Clear analytical framework
           - Appropriate analysis methods
           - Tool/software justification
           - Processing procedures
        
        4. Validity & Reliability (0-20):
           - Validity measures
           - Reliability considerations
           - Limitation awareness
           - Bias mitigation strategies

        Methodology text:
        {text[:4000]}...

        Output Format:
        {{
          "research_design": float,
          "data_collection": float,
          "data_analysis": float,
          "validity_reliability": float,
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
                "research_design": 0,
                "data_collection": 0,
                "data_analysis": 0,
                "validity_reliability": 0,
                "justification": "Error in LLM evaluation",
                "strengths": [],
                "improvements": [],
            }

    def _calculate_enhanced_final_score(
        self, methodology_score: float, structure_score: float, llm_scores: Dict = None
    ) -> Tuple[float, str, List[str]]:
        """Calculate final score with detailed feedback based on rubric criteria"""
        if self.use_llm and llm_scores:
            # Combine scores with weights
            final_score = (
                methodology_score * 0.4
                + structure_score * 0.2
                + np.mean(
                    [
                        llm_scores["research_design"],
                        llm_scores["data_collection"],
                        llm_scores["data_analysis"],
                        llm_scores["validity_reliability"],
                    ]
                )
                * 0.4
            )
        else:
            final_score = methodology_score * 0.7 + structure_score * 20 * 0.3

        # Generate detailed feedback
        feedback = []

        # Determine grade and feedback based on rubric
        if final_score >= 17:
            grade = "Distinction (17-20)"
            feedback.append("Clearly articulated and well-justified methodology")
        elif final_score >= 14:
            grade = "Distinction (14-16)"
            feedback.append("Well-argued methodology with minor improvements possible")
        elif final_score >= 12:
            grade = "Merit (12-13)"
            feedback.append("Appropriate methodology but could be explained better")
        elif final_score >= 10:
            grade = "Pass (10-11)"
            feedback.append("Methodology outlined but lacks clarity in key areas")
        elif final_score >= 5:
            grade = "Fail (5-9)"
            feedback.append("Methodology is poorly articulated or inappropriate")
        else:
            grade = "Fail (0-4)"
            feedback.append("Missing or irrelevant methodology information")

        # Add structure-specific feedback
        if structure_score < 0.6:
            feedback.append("Improve logical flow and organization of methodology")

        return final_score, grade, feedback

    def evaluate(self) -> Dict:
        """Perform enhanced evaluation of the methodology section"""
        # Extract methodology
        methodology_text = self.extract_methodology()
        if not methodology_text:
            return {
                "score": 0,
                "grade": "Fail (0-4)",
                "feedback": ["Methodology section not found"],
                "details": {},
            }

        # Evaluate methodology quality
        methodology_score, quality_details = self._evaluate_methodology_quality(
            methodology_text
        )

        # Analyze structural coherence
        structure_score = self._analyze_methodology_structure(methodology_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_enhanced_llm(methodology_text)

        # Calculate final score and generate feedback
        score, grade, feedback = self._calculate_enhanced_final_score(
            methodology_score, structure_score, llm_scores
        )

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "feedback": feedback,
            "methodology_score": float(round(methodology_score, 2)),
            "structure_score": float(round(structure_score * 20, 2)),
            "llm_scores": llm_scores,
            "quality_details": quality_details,
        }
