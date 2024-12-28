from Evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple


class ResultsAnalysisEvaluator(BaseEvaluator):
    """Evaluates thesis results, analysis, findings, and discussion sections"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        """Initialize the ResultsAnalysisEvaluator with option to use LLM"""
        super().__init__(pdf_path, use_llm, base_instance)

        # Initialize analysis components and patterns
        self.analysis_components = {
            "results_presentation": [
                "table",
                "figure",
                "graph",
                "chart",
                "plot",
                "diagram",
                "visualization",
                "data",
                "statistics",
                "values",
                "numbers",
                "measurements",
                "observations",
            ],
            "analysis_methods": [
                "analysis",
                "statistical",
                "comparison",
                "correlation",
                "regression",
                "test",
                "evaluation",
                "assessment",
                "interpretation",
                "investigation",
                "examination",
            ],
            "critical_discussion": [
                "therefore",
                "however",
                "although",
                "despite",
                "consequently",
                "furthermore",
                "moreover",
                "thus",
                "hence",
                "indicates",
                "suggests",
                "implies",
                "demonstrates",
                "shows",
            ],
            "findings_interpretation": [
                "finding",
                "result",
                "outcome",
                "conclusion",
                "implication",
                "significance",
                "importance",
                "relevance",
                "meaning",
                "interpretation",
                "explanation",
            ],
            "limitations_future": [
                "limitation",
                "constraint",
                "restriction",
                "drawback",
                "weakness",
                "future",
                "recommendation",
                "improvement",
                "enhancement",
                "extension",
                "further",
            ],
        }

    def extract_results_section(self) -> str:
        """Extract the results and discussion section"""
        results_patterns = [
            r"(?i)^(?:CHAPTER\s+[45]\.?\s*)?RESULTS?\s*$",
            r"(?i)^(?:CHAPTER\s+[45]\.?\s*)?FINDINGS?\s*$",
            r"(?i)^(?:[45]\.?\s*)?RESULTS?\s+AND\s+DISCUSSIONS?\s*$",
            r"(?i)^(?:[45]\.?\s*)?ANALYSIS\s+AND\s+RESULTS?\s*$",
            r"(?i)^(?:CHAPTER\s+[45]\.?\s*)?ANALYSIS\s+AND\s+DISCUSSIONS?\s*$",
            r"(?i)^(?:EXPERIMENTAL\s+RESULTS?\s*$)",
            r"(?i)^(?:RESULTS?\s+AND\s+ANALYSIS)\s*$",
        ]

        next_section_patterns = [
            r"(?i)^(?:CHAPTER\s+6)",
            r"(?i)^(?:6\.?\s+)",
            r"(?i)^(?:CONCLUSIONS?)\s*$",
            r"(?i)^(?:FUTURE\s+WORK)\s*$",
            r"(?i)^(?:RECOMMENDATIONS?)\s*$",
            r"(?i)^(?:REFERENCES?)\s*$",
            r"(?i)^(?:BIBLIOGRAPHY)\s*$",
        ]

        try:
            results_text = self._extract_section(
                self.full_text, results_patterns, next_section_patterns
            )
            return results_text
        except Exception as e:
            print(f"Error extracting results section: {e}")
            return ""

    def analyze_components(self, text: str) -> Dict:
        """Analyze the presence and quality of key analysis components"""
        components_found = {category: [] for category in self.analysis_components}

        # Find analysis terms in text
        for category, terms in self.analysis_components.items():
            for term in terms:
                if re.search(rf"\b{term}\b", text.lower()):
                    components_found[category].append(term)

        # Calculate completeness scores
        scores = {}
        for category, found_terms in components_found.items():
            scores[category] = len(found_terms) / len(
                self.analysis_components[category]
            )

        return {
            "components": components_found,
            "scores": scores,
            "overall_score": np.mean(list(scores.values())),
        }

    def assess_critical_analysis(self, text: str) -> float:
        """Assess the level of critical analysis and discussion"""
        doc = self.nlp(text)

        # Critical analysis indicators
        critical_markers = [
            "because",
            "since",
            "as a result",
            "due to",
            "consequently",
            "this suggests",
            "this indicates",
            "this implies",
            "this demonstrates",
            "in contrast",
            "compared to",
            "while",
            "whereas",
            "however",
            "on the other hand",
            "alternatively",
            "although",
            "nevertheless",
        ]

        # Count critical analysis markers
        marker_count = sum(1 for marker in critical_markers if marker in text.lower())

        # Calculate density of analysis terms
        analysis_terms = (
            self.analysis_components["analysis_methods"]
            + self.analysis_components["critical_discussion"]
        )

        analysis_density = len(
            [token for token in doc if token.text.lower() in analysis_terms]
        ) / len(doc)

        # Consider sentence complexity
        complex_sentences = len(
            [
                sent
                for sent in doc.sents
                if len(list(sent.noun_chunks)) > 2
                and any(token.dep_ in ["advcl", "ccomp", "xcomp"] for token in sent)
            ]
        )
        complexity_score = min(1.0, complex_sentences / 10)

        return min(1.0, (marker_count / 15 + analysis_density + complexity_score) / 3)

    def _evaluate_with_llm(self, text: str) -> Dict:
        """Use LLM to evaluate results and discussion quality"""
        prompt = f"""
        Evaluate this results and discussion section based on:
        1. Results Presentation (results_quality) (0-15):
           - Clarity and organization of results
           - Appropriate use of tables/figures
           - Completeness of data presentation
        2. Analysis Depth (analysis_depth) (0-15):
           - Depth of analysis
           - Critical thinking
           - Interpretation quality
        3. Discussion Quality (discussion_quality) (0-15):
           - Integration with literature
           - Critical evaluation
           - Implications discussion
        
        Output Format:
        {{
          results_quality: number,
          analysis_depth: number,
          discussion_quality: number,
          justification: string
        }}

        Results text:
        {text[:4000]}...

        Provide scores and brief justification in JSON format.
        """

        try:
            response = self._get_llm_scores(prompt)
            return eval(response)
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                "results_quality": 0,
                "analysis_depth": 0,
                "discussion_quality": 0,
                "justification": "Error in LLM evaluation",
            }

    def calculate_final_score(
        self, component_analysis: Dict, critical_score: float, llm_scores: Dict = None
    ) -> Tuple[float, str]:
        """Calculate final score and determine grade"""
        # Base scores (without LLM)
        component_score = component_analysis["overall_score"] * 15  # Scale to 15
        critical_score = critical_score * 15  # Scale to 15

        if self.use_llm and llm_scores:
            # Combine with LLM scores (already on 15-point scale)
            llm_avg = (
                llm_scores["results_quality"]
                + llm_scores["analysis_depth"]
                + llm_scores["discussion_quality"]
            ) / 3
            final_score = component_score * 0.3 + critical_score * 0.3 + llm_avg * 0.4
        else:
            final_score = (component_score + critical_score) / 2

        # Determine grade based on rubric
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
        """Perform complete evaluation of the results and discussion section"""
        # Extract results section
        results_text = self.extract_results_section()
        if not results_text:
            return {
                "score": 0,
                "grade": "Fail (0-3)",
                "feedback": "Results and discussion section not found",
            }

        # Analyze components
        component_analysis = self.analyze_components(results_text)

        # Assess critical analysis
        critical_score = self.assess_critical_analysis(results_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_llm(results_text)

        # Calculate final score
        score, grade = self.calculate_final_score(
            component_analysis, critical_score, llm_scores
        )

        # Generate feedback
        feedback = []
        missing_components = []
        for category, found_terms in component_analysis["components"].items():
            if not found_terms:
                missing_components.append(category.replace("_", " "))

        if missing_components:
            feedback.append(f"Missing or insufficient: {', '.join(missing_components)}")
        if critical_score < 0.6:
            feedback.append("Insufficient critical analysis and discussion")

        if llm_scores:
            feedback.append(f"LLM Analysis: {llm_scores.get('justification', '')}")

        return {
            "score": round(score, 2),
            "grade": grade,
            "component_analysis": {
                "found_components": component_analysis["components"],
                "component_scores": component_analysis["scores"],
            },
            "critical_score": round(critical_score * 15, 2),  # Scale to 15
            "llm_scores": llm_scores,
            "feedback": ". ".join(feedback),
        }
