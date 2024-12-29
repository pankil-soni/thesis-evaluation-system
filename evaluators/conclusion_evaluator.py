from evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict

class ConclusionEvaluator(BaseEvaluator):
    """Enhanced evaluator for thesis conclusion with improved assessment criteria"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)
        self.quality_indicators = self._initialize_quality_indicators()

    def extract_conclusion_section(self) -> str:
        """Extract the conclusion section"""
        return self.sections.get("conclusion", "")

    def _initialize_quality_indicators(self) -> Dict:
        """Initialize comprehensive quality indicators for conclusion assessment"""
        return {
            "objectives_achievement": {
                "explicit_reference": [
                    r"(?i)objective(s)?\s+(?:was|were)\s+(?:achieved|met|fulfilled)",
                    r"(?i)aim(s)?\s+(?:was|were)\s+(?:achieved|met|fulfilled)",
                    r"(?i)goal(s)?\s+(?:was|were)\s+(?:achieved|met|fulfilled)",
                    r"(?i)successfully\s+(?:achieved|demonstrated|showed)",
                    r"(?i)research\s+question(s)?\s+(?:was|were)\s+answered",
                ],
                "findings_linkage": [
                    r"(?i)findings?\s+show(s|ed)?",
                    r"(?i)results?\s+demonstrate(s|d)?",
                    r"(?i)study\s+(?:showed|proved|confirmed)",
                    r"(?i)research\s+(?:showed|proved|confirmed)",
                    r"(?i)evidence\s+supports?",
                ],
                "accomplishment_proof": [
                    r"(?i)demonstrated\s+through",
                    r"(?i)proved\s+by",
                    r"(?i)validated\s+(?:through|by)",
                    r"(?i)confirmed\s+(?:through|by)",
                    r"(?i)supported\s+by\s+(?:results|findings|evidence)",
                ],
            },
            "findings_summary": {
                "key_findings": [
                    r"(?i)key\s+finding(s)?",
                    r"(?i)main\s+result(s)?",
                    r"(?i)primary\s+outcome(s)?",
                    r"(?i)significant\s+(?:finding|result|outcome)",
                    r"(?i)important\s+(?:finding|result|outcome)",
                ],
                "synthesis": [
                    r"(?i)synthesizing\s+the\s+results",
                    r"(?i)combining\s+(?:all|the)\s+findings",
                    r"(?i)overall\s+(?:findings|results)",
                    r"(?i)collectively\s+(?:show|demonstrate|indicate)",
                    r"(?i)taken\s+together",
                ],
                "significance": [
                    r"(?i)significant(?:ly)?\s+(?:shows|demonstrates|proves)",
                    r"(?i)notably",
                    r"(?i)importantly",
                    r"(?i)crucial(?:ly)?",
                    r"(?i)substantial(?:ly)?",
                ],
            },
            "implications_impact": {
                "theoretical": [
                    r"(?i)theoretical\s+implications?",
                    r"(?i)contributes?\s+to\s+(?:theory|knowledge|literature)",
                    r"(?i)advances?\s+(?:understanding|knowledge)",
                    r"(?i)theoretical\s+contribution",
                    r"(?i)conceptual\s+advancement",
                ],
                "practical": [
                    r"(?i)practical\s+implications?",
                    r"(?i)real[\s-]world\s+applications?",
                    r"(?i)industry\s+applications?",
                    r"(?i)practical\s+applications?",
                    r"(?i)business\s+implications?",
                ],
                "societal": [
                    r"(?i)societal\s+impact",
                    r"(?i)social\s+implications?",
                    r"(?i)benefit\s+to\s+society",
                    r"(?i)community\s+impact",
                    r"(?i)broader\s+impact",
                ],
            },
            "recommendations": {
                "future_research": [
                    r"(?i)future\s+research",
                    r"(?i)further\s+studies?",
                    r"(?i)future\s+studies?",
                    r"(?i)future\s+directions?",
                    r"(?i)research\s+opportunities?",
                ],
                "improvements": [
                    r"(?i)recommend(?:ed)?\s+improvements?",
                    r"(?i)suggest(?:ed)?\s+enhancements?",
                    r"(?i)potential\s+improvements?",
                    r"(?i)areas?\s+for\s+improvement",
                    r"(?i)could\s+be\s+improved",
                ],
                "implementation": [
                    r"(?i)implementation\s+recommendations?",
                    r"(?i)practical\s+recommendations?",
                    r"(?i)recommend(?:ed)?\s+actions?",
                    r"(?i)suggest(?:ed)?\s+steps?",
                    r"(?i)proposed\s+measures?",
                ],
            },
        }

    def _evaluate_conclusion_quality(self, text: str) -> Tuple[float, Dict]:
        """
        Evaluate the quality of conclusion using weighted criteria
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
                "explicit_reference": 0.4,
                "findings_linkage": 0.3,
                "accomplishment_proof": 0.3,
                "key_findings": 0.4,
                "synthesis": 0.3,
                "significance": 0.3,
                "theoretical": 0.35,
                "practical": 0.35,
                "societal": 0.3,
                "future_research": 0.35,
                "improvements": 0.35,
                "implementation": 0.3,
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
            "objectives_achievement": 0.3,
            "findings_summary": 0.25,
            "implications_impact": 0.25,
            "recommendations": 0.2,
        }

        total_weight = sum(category_weights.values())
        weighted_score = (
            sum(
                score * category_weights[category] for category, score in scores.items()
            )
            / total_weight
        )

        return weighted_score * 10, dict(details)  # Scale to 0-10

    def _analyze_coherence(self, text: str) -> float:
        """
        Analyze the coherence and flow of the conclusion
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return 0.0

        # Expected conclusion flow patterns
        flow_patterns = [
            (r"(?i)objective|aim|goal|purpose", 0),
            (r"(?i)finding|result|outcome", 1),
            (r"(?i)implication|impact|significance", 2),
            (r"(?i)recommend|future|suggest", 3),
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
        """Enhanced LLM evaluation with specific conclusion criteria"""
        prompt = f"""
        Evaluate this conclusion based on these detailed criteria:
        
        1. Objectives Achievement (0-10):
           - Clear demonstration of achieved objectives
           - Evidence-based achievement claims
           - Comprehensive coverage of all objectives
           - Link between findings and objectives
        
        2. Findings Summary (0-10):
           - Clear presentation of key findings
           - Synthesis of results
           - Significance of findings
           - Integration of evidence
        
        3. Implications & Impact (0-10):
           - Theoretical implications
           - Practical implications
           - Societal impact
           - Contribution to knowledge
        
        4. Recommendations (0-10):
           - Quality of recommendations
           - Future research directions
           - Implementation guidance
           - Practical steps

        Conclusion text:
        {text[:4000]}...

        Output Format:

        {{
            "objectives_achievement": float,
            "findings_summary": float,
            "implications_impact": float,
            "recommendations": float,
            "justification": str
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
                "objectives_achievement": 0,
                "findings_summary": 0,
                "implications_impact": 0,
                "recommendations": 0,
                "justification": "Error in LLM evaluation",
                "strengths": [],
                "improvements": [],
            }

    def _calculate_enhanced_final_score(
        self, conclusion_score: float, coherence_score: float, llm_scores: Dict = None
    ) -> Tuple[float, str, List[str]]:
        """Calculate final score with detailed feedback based on rubric criteria"""
        if self.use_llm and llm_scores:
            # Combine scores with weights
            final_score = (
                conclusion_score * 0.4
                + coherence_score * 0.2
                + np.mean(
                    [
                        llm_scores["objectives_achievement"],
                        llm_scores["findings_summary"],
                        llm_scores["implications_impact"],
                        llm_scores["recommendations"],
                    ]
                )
                * 0.4
            )
        else:
            final_score = conclusion_score * 0.7 + coherence_score * 10 * 0.3

        # Generate detailed feedback
        feedback = []

        # Determine grade and feedback based on rubric
        if final_score >= 9:
            grade = "Distinction (9-10)"
            feedback.append(
                "Outstanding conclusion with perfect objectives achievement and recommendations"
            )
        elif final_score >= 7:
            grade = "Distinction (7-8)"
            feedback.append("Very good conclusion demonstrating achieved objectives")
        elif final_score >= 6:
            grade = "Merit (6)"
            feedback.append("Good conclusion but needs improvement in recommendations")
        elif final_score >= 5:
            grade = "Pass (5)"
            feedback.append(
                "Satisfactory conclusion but only partially addresses objectives"
            )
        elif final_score >= 3:
            grade = "Fail (3-4)"
            feedback.append("Inadequate conclusions and recommendations")
        else:
            grade = "Fail (0-2)"
            feedback.append("Little or no evidence of conclusions and recommendations")

        # Add coherence-specific feedback
        if coherence_score < 0.6:
            feedback.append("Improve logical flow and organization of conclusion")

        return final_score, grade, feedback

    def evaluate(self) -> Dict:
        """Perform enhanced evaluation of the conclusion section"""
        # Extract conclusion section
        conclusion_text = self.extract_conclusion_section()
        if not conclusion_text:
            return {
                "score": 0,
                "grade": "Fail (0-2)",
                "feedback": ["Conclusion section not found"],
                "details": {},
            }

        # Evaluate conclusion quality
        conclusion_score, quality_details = self._evaluate_conclusion_quality(
            conclusion_text
        )

        # Analyze coherence
        coherence_score = self._analyze_coherence(conclusion_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_enhanced_llm(conclusion_text)

        # Calculate final score and generate feedback
        score, grade, feedback = self._calculate_enhanced_final_score(
            conclusion_score, coherence_score, llm_scores
        )

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "feedback": feedback,
            "conclusion_score": float(round(conclusion_score, 2)),
            "coherence_score": float(round(coherence_score * 10, 2)),
            "llm_scores": llm_scores,
            "quality_details": quality_details,
        }
