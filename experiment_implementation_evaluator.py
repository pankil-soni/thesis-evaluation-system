from evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


class ExperimentImplementationEvaluator(BaseEvaluator):
    """Enhanced evaluator for thesis experiment implementation with improved assessment criteria"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)
        self.quality_indicators = self._initialize_quality_indicators()

    def extract_implementation_section(self) -> str:
        """Extract the implementation/experiment section"""
        return self.sections.get("analysis_design", "")

    def _initialize_quality_indicators(self) -> Dict:
        """Initialize comprehensive quality indicators for implementation assessment"""
        return {
            "implementation_details": {
                "architecture": [
                    r"(?i)system\s+architecture",
                    r"(?i)component\s+diagram",
                    r"(?i)system\s+design",
                    r"(?i)module[s]?(\s+design)?",
                    r"(?i)class\s+diagram",
                    r"(?i)sequence\s+diagram",
                    r"(?i)flow\s+diagram",
                ],
                "algorithms": [
                    r"(?i)algorithm(\s+design)?",
                    r"(?i)pseudocode",
                    r"(?i)flowchart",
                    r"(?i)process\s+flow",
                    r"(?i)computational\s+steps?",
                    r"(?i)processing\s+logic",
                ],
                "optimization": [
                    r"(?i)performance\s+optimization",
                    r"(?i)code\s+optimization",
                    r"(?i)efficiency\s+improvement",
                    r"(?i)resource\s+usage",
                    r"(?i)complexity(\s+analysis)?",
                ],
            },
            "experimental_setup": {
                "environment": [
                    r"(?i)experimental\s+setup",
                    r"(?i)test\s+environment",
                    r"(?i)hardware\s+specification",
                    r"(?i)software\s+requirement",
                    r"(?i)system\s+configuration",
                    r"(?i)platform(\s+setup)?",
                ],
                "parameters": [
                    r"(?i)parameter[s]?(\s+setting)?",
                    r"(?i)configuration\s+setting",
                    r"(?i)hyperparameter",
                    r"(?i)experiment\s+variable",
                    r"(?i)control\s+variable",
                    r"(?i)threshold\s+value",
                ],
                "datasets": [
                    r"(?i)dataset(\s+description)?",
                    r"(?i)data\s+preparation",
                    r"(?i)data\s+preprocessing",
                    r"(?i)training\s+data",
                    r"(?i)test\s+data",
                    r"(?i)validation\s+data",
                ],
            },
            "tool_usage": {
                "development": [
                    r"(?i)programming\s+language",
                    r"(?i)development\s+tool",
                    r"(?i)IDE",
                    r"(?i)framework",
                    r"(?i)library",
                    r"(?i)API",
                    r"(?i)SDK",
                ],
                "deployment": [
                    r"(?i)deployment(\s+tool)?",
                    r"(?i)container",
                    r"(?i)virtualization",
                    r"(?i)cloud(\s+platform)?",
                    r"(?i)server(\s+setup)?",
                    r"(?i)hosting",
                ],
                "version_control": [
                    r"(?i)version\s+control",
                    r"(?i)source\s+control",
                    r"(?i)code\s+repository",
                    r"(?i)git",
                    r"(?i)branch",
                    r"(?i)commit",
                ],
            },
            "evaluation_metrics": {
                "performance": [
                    r"(?i)performance\s+metric",
                    r"(?i)benchmark",
                    r"(?i)execution\s+time",
                    r"(?i)throughput",
                    r"(?i)latency",
                    r"(?i)response\s+time",
                    r"(?i)scalability",
                ],
                "accuracy": [
                    r"(?i)accuracy(\s+metric)?",
                    r"(?i)precision",
                    r"(?i)recall",
                    r"(?i)f1[\s-]score",
                    r"(?i)error\s+rate",
                    r"(?i)mean\s+square\s+error",
                ],
                "resource_usage": [
                    r"(?i)memory\s+usage",
                    r"(?i)cpu\s+usage",
                    r"(?i)disk\s+usage",
                    r"(?i)network\s+usage",
                    r"(?i)resource\s+consumption",
                    r"(?i)utilization",
                ],
            },
        }

    def _evaluate_implementation_quality(self, text: str) -> Tuple[float, Dict]:
        """
        Evaluate the quality of implementation using weighted criteria
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
                "architecture": 0.4,
                "algorithms": 0.3,
                "optimization": 0.3,
                "environment": 0.35,
                "parameters": 0.35,
                "datasets": 0.3,
                "development": 0.4,
                "deployment": 0.3,
                "version_control": 0.3,
                "performance": 0.35,
                "accuracy": 0.35,
                "resource_usage": 0.3,
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
            "implementation_details": 0.3,
            "experimental_setup": 0.25,
            "tool_usage": 0.25,
            "evaluation_metrics": 0.2,
        }

        total_weight = sum(category_weights.values())
        weighted_score = (
            sum(
                score * category_weights[category] for category, score in scores.items()
            )
            / total_weight
        )

        return weighted_score * 20, dict(details)  # Scale to 0-20

    def _analyze_explanation_quality(self, text: str) -> float:
        """
        Analyze the quality and completeness of explanations
        """
        explanation_patterns = {
            "detail_level": [
                r"(?i)in\s+detail",
                r"(?i)specifically",
                r"(?i)step[\s-]by[\s-]step",
                r"(?i)detailed\s+explanation",
                r"(?i)comprehensive\s+description",
            ],
            "clarity": [
                r"(?i)illustrated\s+in",
                r"(?i)demonstrated\s+by",
                r"(?i)shown\s+in\s+figure",
                r"(?i)explained\s+in",
                r"(?i)described\s+in",
            ],
            "justification": [
                r"(?i)because",
                r"(?i)therefore",
                r"(?i)consequently",
                r"(?i)as\s+a\s+result",
                r"(?i)this\s+ensures",
            ],
        }

        scores = []
        for category, patterns in explanation_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, text))
            scores.append(min(1.0, matches / len(patterns)))

        return np.mean(scores)

    def _evaluate_with_enhanced_llm(self, text: str) -> Dict:
        """Enhanced LLM evaluation with specific implementation criteria"""
        prompt = f"""
        Evaluate this implementation/experiments section based on these detailed criteria:
        
        1. Implementation Quality (0-20):
           - Detailed technical explanation
           - Clear system architecture
           - Algorithm descriptions
           - Code/implementation details
        
        2. Experimental Setup (0-20):
           - Environment configuration
           - Parameter settings
           - Dataset preparation
           - Control measures
        
        3. Tool Usage (0-20):
           - Development tools
           - Framework utilization
           - Deployment setup
           - Version control
        
        4. Results and Metrics (0-20):
           - Performance metrics
           - Accuracy measures
           - Resource utilization
           - Comparative analysis

        Implementation text:
        {text[:4000]}...

        Output Format:
        {{
            "implementation_quality": float,
            "experimental_setup": float,
            "tool_usage": float,
            "results_metrics": float,
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
                "implementation_quality": 0,
                "experimental_setup": 0,
                "tool_usage": 0,
                "results_metrics": 0,
                "justification": "Error in LLM evaluation",
                "strengths": [],
                "improvements": [],
            }

    def _calculate_enhanced_final_score(
        self,
        implementation_score: float,
        explanation_score: float,
        llm_scores: Dict = None,
    ) -> Tuple[float, str, List[str]]:
        """Calculate final score with detailed feedback based on rubric criteria"""
        if self.use_llm and llm_scores:
            # Combine scores with weights
            final_score = (
                implementation_score * 0.4
                + explanation_score * 0.2
                + np.mean(
                    [
                        llm_scores["implementation_quality"],
                        llm_scores["experimental_setup"],
                        llm_scores["tool_usage"],
                        llm_scores["results_metrics"],
                    ]
                )
                * 0.4
            )
        else:
            final_score = implementation_score * 0.7 + explanation_score * 20 * 0.3

        # Generate detailed feedback
        feedback = []

        # Determine grade and feedback based on rubric
        if final_score >= 17:
            grade = "Distinction (17-20)"
            feedback.append("Outstanding implementation with comprehensive explanation")
        elif final_score >= 14:
            grade = "Distinction (14-16)"
            feedback.append("Very good implementation with strong evidence")
        elif final_score >= 12:
            grade = "Merit (12-13)"
            feedback.append("Good implementation but some areas need improvement")
        elif final_score >= 10:
            grade = "Pass (10-11)"
            feedback.append("Satisfactory implementation but lacks detail")
        elif final_score >= 5:
            grade = "Fail (5-9)"
            feedback.append("Basic implementation with insufficient detail")
        else:
            grade = "Fail (0-4)"
            feedback.append("Poor evidence of implementation and tool usage")

        # Add explanation-specific feedback
        if explanation_score < 0.6:
            feedback.append("Implementation needs more detailed explanations")

        return final_score, grade, feedback

    def evaluate(self) -> Dict:
        """Perform enhanced evaluation of the implementation section"""
        # Extract implementation section
        impl_text = self.extract_implementation_section()
        if not impl_text:
            return {
                "score": 0,
                "grade": "Fail (0-4)",
                "feedback": ["Implementation section not found"],
                "details": {},
            }

        # Evaluate implementation quality
        implementation_score, quality_details = self._evaluate_implementation_quality(
            impl_text
        )

        # Analyze explanation quality
        explanation_score = self._analyze_explanation_quality(impl_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_enhanced_llm(impl_text)

        # Calculate final score and generate feedback
        score, grade, feedback = self._calculate_enhanced_final_score(
            implementation_score, explanation_score, llm_scores
        )

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "feedback": feedback,
            "implementation_score": float(round(implementation_score, 2)),
            "explanation_score": float(round(explanation_score * 20, 2)),
            "llm_scores": llm_scores,
            "quality_details": quality_details,
        }
