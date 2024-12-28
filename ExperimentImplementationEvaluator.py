from Evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple


class ExperimentImplementationEvaluator(BaseEvaluator):
    """Evaluates thesis experiment implementation section"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        """Initialize the ExperimentImplementationEvaluator with option to use LLM"""
        super().__init__(pdf_path, use_llm, base_instance)

        # Initialize technical terms and patterns
        self.technical_components = {
            "tools_software": [
                "python",
                "java",
                "c\+\+",
                "matlab",
                "tensorflow",
                "pytorch",
                "opencv",
                "sql",
                "mongodb",
                "docker",
                "kubernetes",
                "git",
                "linux",
                "windows",
                "mac",
                "ide",
                "framework",
                "library",
            ],
            "hardware": [
                "cpu",
                "gpu",
                "ram",
                "processor",
                "server",
                "cluster",
                "machine",
                "device",
                "sensor",
                "camera",
                "memory",
                "storage",
            ],
            "experiment_design": [
                "control group",
                "treatment",
                "variable",
                "parameter",
                "configuration",
                "setup",
                "environment",
                "condition",
                "scenario",
                "test case",
                "benchmark",
            ],
            "metrics": [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "error rate",
                "performance",
                "efficiency",
                "speed",
                "throughput",
                "latency",
                "memory usage",
                "cpu usage",
            ],
            "validation": [
                "validation",
                "testing",
                "verification",
                "evaluation",
                "comparison",
                "baseline",
                "ground truth",
                "metric",
                "measurement",
                "assessment",
            ],
        }

    def extract_implementation_section(self) -> str:
        """Extract the implementation/experiment section"""
        impl_patterns = [
            r"(?i)^(?:CHAPTER\s+4\.?\s*)?(?:ANALYSIS,?\s*DESIGN,?\s*(?:AND\s+)?EXPERIMENTS?)\s*$",
            r"(?i)^(?:CHAPTER\s+4\.?\s*)?IMPLEMENTATION\s*$",
            r"(?i)^(?:4\.?\s*)?(?:ANALYSIS,?\s*DESIGN,?\s*(?:AND\s+)?EXPERIMENTS?)\s*$",
            r"(?i)^(?:EXPERIMENTAL\s+SETUP)\s*$",
            r"(?i)^(?:SYSTEM\s+IMPLEMENTATION)\s*$",
            r"(?i)^(?:CHAPTER\s+4\.?\s*)?(?:DESIGN\s+AND\s+IMPLEMENTATION)\s*$",
            r"(?i)^(?:SYSTEM\s+DESIGN\s+AND\s+DEVELOPMENT)\s*$",
            r"(?i)^(?:DEVELOPMENT\s+AND\s+IMPLEMENTATION)\s*$",
            r"(?i)^(?:SIMULATION\s+AND\s+RESULTS)\s*$",
        ]

        next_section_patterns = [
            r"(?i)^(?:CHAPTER\s+5)",
            r"(?i)^(?:5\.?\s+)",
            r"(?i)^(?:RESULTS?\s+AND\s+DISCUSSIONS?)\s*$",
            r"(?i)^(?:RESULTS?\s+AND\s+ANALYSIS)\s*$",
            r"(?i)^(?:EVALUATION)\s*$",
            r"(?i)^(?:DISCUSSION)\s*$",
            r"(?i)^(?:CONCLUSION)",
            r"(?i)^(?:FUTURE\s+WORK)",
            r"(?i)^(?:REFERENCES)",
        ]

        try:
            implementation_text = self._extract_section(
                self.full_text, impl_patterns, next_section_patterns
            )
            return implementation_text
        except Exception as e:
            print(f"Error extracting implementation section: {e}")
            return ""

    def analyze_technical_details(self, text: str) -> Dict:
        """Analyze technical components and implementation details"""
        components_found = {category: [] for category in self.technical_components}

        # Find technical terms in text
        for category, terms in self.technical_components.items():
            for term in terms:
                if re.search(rf"\b{term}\b", text.lower()):
                    components_found[category].append(term)

        # Calculate completeness scores
        scores = {}
        for category, found_terms in components_found.items():
            scores[category] = len(found_terms) / len(
                self.technical_components[category]
            )

        return {
            "components": components_found,
            "scores": scores,
            "overall_score": np.mean(list(scores.values())),
        }

    def analyze_detail_level(self, text: str) -> float:
        """Analyze the level of detail in explanations"""
        doc = self.nlp(text)

        # Look for detailed descriptions
        detail_markers = [
            "specifically",
            "in detail",
            "step by step",
            "procedure",
            "process",
            "method",
            "approach",
            "technique",
            "algorithm",
            "implementation",
            "configuration",
            "parameter",
            "setting",
        ]

        # Count detail markers
        marker_count = sum(1 for marker in detail_markers if marker in text.lower())

        # Calculate technical term density
        technical_terms = []
        for terms in self.technical_components.values():
            technical_terms.extend(terms)

        tech_density = len(
            [token for token in doc if token.text.lower() in technical_terms]
        ) / len(doc)

        return min(1.0, (marker_count / 10 + tech_density) / 2)

    def _evaluate_with_llm(self, text: str) -> Dict:
        """Use LLM to evaluate implementation quality"""
        prompt = f"""
        Evaluate this implementation/experiments section based on:
        1. Technical Depth (technical_depth) (0-20):
           - Detail of technical implementation
           - Tool and technology usage
           - Implementation complexity
        2. Experimental Design (experimental_design) (0-20):
           - Setup and methodology
           - Parameter choices
           - Control measures
        3. Validation (validation) (0-20):
           - Testing procedures
           - Result verification
           - Performance assessment
        
        Output Format:
        {{
          technical_depth: number,
          experimental_design: number,
          validation: number,
          justification: string
        }}

        Implementation text:
        {text[:4000]}...

        Provide scores and brief justification in JSON format.
        """

        try:
            response = self._get_llm_scores(prompt)
            return eval(response)
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                "technical_depth": 0,
                "experimental_design": 0,
                "validation": 0,
                "justification": "Error in LLM evaluation",
            }

    def calculate_final_score(
        self, technical_analysis: Dict, detail_score: float, llm_scores: Dict = None
    ) -> Tuple[float, str]:
        """Calculate final score and determine grade"""
        # Base scores (without LLM)
        technical_score = technical_analysis["overall_score"] * 10
        detail_score = detail_score * 10

        if self.use_llm and llm_scores:
            # Combine with LLM scores
            llm_avg = (
                llm_scores["technical_depth"]
                + llm_scores["experimental_design"]
                + llm_scores["validation"]
            ) / 3
            final_score = technical_score * 0.3 + detail_score * 0.3 + llm_avg * 0.4
        else:
            final_score = (technical_score + detail_score) / 2

        # Determine grade
        if final_score >= 17:
            grade = "Distinction (17-20)"
        elif final_score >= 14:
            grade = "Distinction (14-16)"
        elif final_score >= 12:
            grade = "Merit (12-13)"
        elif final_score >= 10:
            grade = "Pass (10-11)"
        elif final_score >= 5:
            grade = "Fail (5-9)"
        else:
            grade = "Fail (0-4)"

        return final_score, grade

    def evaluate(self) -> Dict:
        """Perform complete evaluation of the implementation section"""
        # Extract implementation section
        impl_text = self.extract_implementation_section()
        if not impl_text:
            return {
                "score": 0,
                "grade": "Fail (0-4)",
                "feedback": "Implementation section not found",
            }

        # Analyze technical details
        technical_analysis = self.analyze_technical_details(impl_text)

        # Analyze detail level
        detail_score = self.analyze_detail_level(impl_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_llm(impl_text)

        # Calculate final score
        score, grade = self.calculate_final_score(
            technical_analysis, detail_score, llm_scores
        )

        # Generate feedback
        feedback = []
        missing_components = []
        for category, found_terms in technical_analysis["components"].items():
            if not found_terms:
                missing_components.append(category.replace("_", " "))

        if missing_components:
            feedback.append(f"Missing or insufficient: {', '.join(missing_components)}")
        if detail_score < 0.6:
            feedback.append("Implementation lacks sufficient detail")

        if llm_scores:
            feedback.append(f"LLM Analysis: {llm_scores.get('justification', '')}")

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "technical_analysis": {
                "found_components": technical_analysis["components"],
                "component_scores": technical_analysis["scores"],
            },
            "detail_score": float(round(detail_score * 10, 2)),
            "llm_scores": llm_scores,
            "feedback": ". ".join(feedback),
        }
