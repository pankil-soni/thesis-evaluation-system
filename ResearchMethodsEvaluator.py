from Evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple


class ResearchMethodsEvaluator(BaseEvaluator):
    """Evaluates thesis research methodology section"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        """Initialize the ResearchMethodsEvaluator with option to use LLM"""
        super().__init__(pdf_path, use_llm, base_instance)

        # Initialize common methodology terms and patterns
        self.methodology_terms = {
            "research_design": [
                "quantitative",
                "qualitative",
                "mixed method",
                "experimental",
                "quasi-experimental",
                "descriptive",
                "exploratory",
                "correlational",
                "case study",
                "longitudinal",
                "cross-sectional",
            ],
            "data_collection": [
                "survey",
                "interview",
                "observation",
                "questionnaire",
                "sampling",
                "focus group",
                "experiment",
                "measurement",
                "instrument",
                "data collection",
                "recording",
            ],
            "data_analysis": [
                "statistical",
                "regression",
                "analysis",
                "coding",
                "thematic",
                "content analysis",
                "factor analysis",
                "correlation",
                "t-test",
                "anova",
                "grounded theory",
            ],
            "validation": [
                "validity",
                "reliability",
                "triangulation",
                "verification",
                "credibility",
                "trustworthiness",
                "reproducibility",
                "bias",
                "limitation",
            ],
        }

    def extract_methodology(self) -> str:
        """Extract the methodology section using patterns"""
        method_patterns = [
            r"(?i)^(?:CHAPTER\s+3\.?\s*)?METHODOLOGY\s*$",
            r"(?i)^(?:3\.?\s+)?RESEARCH\s+METHODS?\s*$",
            r"(?i)^(?:RESEARCH\s+METHODOLOGY)\s*$",
            r"(?i)^(?:METHODS?\s+AND\s+MATERIALS)\s*$",
            r"(?i)^(?:EXPERIMENTAL\s+DESIGN)\s*$",
        ]

        next_section_patterns = [
            r"(?i)^(?:CHAPTER\s+4)",
            r"(?i)^(?:4\.?\s+)",
            r"(?i)^(?:RESULTS)",
            r"(?i)^(?:FINDINGS)",
            r"(?i)^(?:DATA\s+ANALYSIS)",
            r"(?i)^(?:ANALYSIS,\s+DESIGN,\s+EXPERIMENTS\s+AND\s+DISCUSSION)",
            r"(?i)^(?:ANALYSIS\s+AND\s+DISCUSSION)",
        ]

        try:
            methodology_text = self._extract_section(
                self.full_text, method_patterns, next_section_patterns
            )
            return methodology_text
        except Exception as e:
            print(f"Error extracting methodology: {e}")
            return ""

    def analyze_methodology_components(self, text: str) -> Dict:
        """Analyze the presence and quality of key methodology components"""
        components_found = {category: [] for category in self.methodology_terms}

        # Find methodology terms in text
        for category, terms in self.methodology_terms.items():
            for term in terms:
                if re.search(rf"\b{term}\b", text.lower()):
                    components_found[category].append(term)

        # Calculate completeness scores
        scores = {}
        for category, found_terms in components_found.items():
            scores[category] = len(found_terms) / len(self.methodology_terms[category])

        return {
            "components": components_found,
            "scores": scores,
            "overall_score": np.mean(list(scores.values())),
        }

    def assess_clarity(self, text: str) -> float:
        """Assess the clarity and articulation of the methodology"""
        doc = self.nlp(text)

        # Analyze sentence structure
        sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
        if not sentence_lengths:
            return 0

        avg_length = np.mean(sentence_lengths)

        # Ideal sentence length is between 15-25 words
        length_score = 1.0 - min(1.0, abs(20 - avg_length) / 15)

        # Check for methodology-specific linguistic markers
        clarity_markers = [
            "therefore",
            "thus",
            "consequently",
            "specifically",
            "in order to",
            "as a result",
            "for this purpose",
        ]

        marker_count = sum(1 for marker in clarity_markers if marker in text.lower())
        marker_score = min(1.0, marker_count / 5)

        return length_score * 0.6 + marker_score * 0.4

    def _evaluate_with_llm(self, text: str) -> Dict:
        """Use LLM to evaluate the methodology quality"""
        prompt = f"""
        Evaluate this research methodology section based on the following criteria:
        1. Research Design (research_design) (0-20):
           - Clear articulation of research approach
           - Justification of chosen methods
           - Alignment with research objectives
        2. Data Collection (data_collection) (0-20):
           - Clear description of data collection methods
           - Appropriate sampling strategies
           - Consideration of limitations
        3. Data Analysis (data_analysis) (0-20):
           - Clear analytical framework
           - Appropriate analysis methods
           - Consideration of validity/reliability
        
        Output Format:
        {{
          research_design: number,
          data_collection: number,
          data_analysis: number,
          justification: string
        }}
        
        Methodology text:
        {text[:4000]}...
        
        Provide numerical scores and brief justification in JSON format.
        """

        try:
            response = self._get_llm_scores(prompt)
            return eval(response)
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                "research_design": 0,
                "data_collection": 0,
                "data_analysis": 0,
                "justification": "Error in LLM evaluation",
            }

    def calculate_final_score(
        self, component_analysis: Dict, clarity_score: float, llm_scores: Dict = None
    ) -> Tuple[float, str]:
        """Calculate final score and determine grade"""
        # Base scores (without LLM)
        component_score = component_analysis["overall_score"] * 10  # Out of 10
        clarity_score = clarity_score * 10  # Out of 10

        if self.use_llm and llm_scores:
            # Combine with LLM scores (normalized to 20 points)
            llm_avg = (
                llm_scores["research_design"]
                + llm_scores["data_collection"]
                + llm_scores["data_analysis"]
            ) / 3
            final_score = component_score * 0.3 + clarity_score * 0.3 + llm_avg * 0.4
        else:
            final_score = (component_score + clarity_score) / 2

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
        """Perform complete evaluation of the methodology section"""
        # Extract methodology
        methodology_text = self.extract_methodology()
        if not methodology_text:
            return {
                "score": 0,
                "grade": "Fail (0-4)",
                "feedback": "Methodology section not found",
            }

        # Analyze components
        component_analysis = self.analyze_methodology_components(methodology_text)

        # Assess clarity
        clarity_score = self.assess_clarity(methodology_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_llm(methodology_text)

        # Calculate final score
        score, grade = self.calculate_final_score(
            component_analysis, clarity_score, llm_scores
        )

        # Generate feedback
        feedback = []
        missing_components = []
        for category, found_terms in component_analysis["components"].items():
            if not found_terms:
                missing_components.append(category.replace("_", " "))

        if missing_components:
            feedback.append(
                f"Missing or insufficient coverage of: {', '.join(missing_components)}"
            )
        if clarity_score < 0.6:
            feedback.append("Methodology lacks clear articulation")

        if llm_scores:
            feedback.append(f"LLM Analysis: {llm_scores.get('justification', '')}")

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "component_analysis": {
                "found_components": component_analysis["components"],
                "component_scores": component_analysis["scores"],
            },
            "clarity_score": float(round(clarity_score * 10, 2)),
            "llm_scores": llm_scores,
            "feedback": ". ".join(feedback),
        }
