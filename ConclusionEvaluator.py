from Evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple


class ConclusionEvaluator(BaseEvaluator):
    """Evaluates thesis conclusion, implications, and recommendations sections"""

    def __init__(self, pdf_path: str, use_llm: bool = True, base_instance=None):
        """Initialize the ConclusionEvaluator with option to use LLM"""
        super().__init__(pdf_path, use_llm, base_instance)

        # Initialize conclusion components and patterns
        self.conclusion_components = {
            "objectives_addressed": [
                "objective",
                "goal",
                "aim",
                "purpose",
                "achieved",
                "accomplished",
                "fulfilled",
                "met",
                "demonstrated",
                "shown",
                "proven",
                "validated",
            ],
            "findings_summary": [
                "finding",
                "result",
                "outcome",
                "conclusion",
                "summary",
                "discovered",
                "identified",
                "determined",
                "revealed",
                "showed",
                "confirmed",
                "established",
            ],
            "implications": [
                "implication",
                "impact",
                "significance",
                "importance",
                "consequence",
                "effect",
                "influence",
                "contribution",
                "relevance",
                "meaning",
                "value",
            ],
            "recommendations": [
                "recommend",
                "suggest",
                "propose",
                "advise",
                "future",
                "improvement",
                "enhancement",
                "development",
                "extension",
                "direction",
                "potential",
            ],
            "limitations": [
                "limitation",
                "constraint",
                "restriction",
                "drawback",
                "weakness",
                "challenge",
                "difficulty",
                "barrier",
                "obstacle",
                "shortcoming",
                "gap",
            ],
        }

    def extract_conclusion_section(self) -> str:
        """Extract the conclusion section"""
        conclusion_patterns = [
            r"(?i)^(?:CHAPTER\s+[56]\.?\s*)?CONCLUSIONS?\s*$",
            r"(?i)^(?:[56]\.?\s*)?CONCLUSIONS?\s+AND\s+RECOMMENDATIONS?\s*$",
            r"(?i)^(?:[56]\.?\s*)?CONCLUSIONS?\s+&\s+RECOMMENDATIONS?\s*$",
            r"(?i)^(?:CHAPTER\s+[56]\.?\s*)?SUMMARY\s+AND\s+CONCLUSIONS?\s*$",
            r"(?i)^(?:[56]\.?\s*)?CONCLUSIONS?\s+AND\s+FUTURE\s+WORK\s*$",
            r"(?i)^(?:CONCLUDING\s+REMARKS)\s*$",
        ]

        next_section_patterns = [
            r"(?i)^(?:REFERENCES?)\s*$",
            r"(?i)^(?:BIBLIOGRAPHY)\s*$",
            r"(?i)^(?:APPENDIX|APPENDICES)\s*$",
            r"(?i)^(?:LIST\s+OF\s+PUBLICATIONS?)\s*$",
            r"(?i)^(?:PUBLICATIONS?)\s*$",
        ]

        try:
            conclusion_text = self._extract_section(
                self.full_text, conclusion_patterns, next_section_patterns
            )
            return conclusion_text
        except Exception as e:
            print(f"Error extracting conclusion section: {e}")
            return ""

    def analyze_components(self, text: str) -> Dict:
        """Analyze the presence and quality of conclusion components"""
        components_found = {category: [] for category in self.conclusion_components}

        # Find conclusion terms in text
        for category, terms in self.conclusion_components.items():
            for term in terms:
                if re.search(rf"\b{term}\b", text.lower()):
                    components_found[category].append(term)

        # Calculate completeness scores
        scores = {}
        for category, found_terms in components_found.items():
            scores[category] = len(found_terms) / len(
                self.conclusion_components[category]
            )

        return {
            "components": components_found,
            "scores": scores,
            "overall_score": np.mean(list(scores.values())),
        }

    def assess_completeness(self, text: str) -> float:
        """Assess how well the conclusion addresses objectives and provides recommendations"""
        doc = self.nlp(text)

        # Look for objective-linking phrases
        objective_markers = [
            "objective",
            "aim",
            "goal",
            "purpose",
            "research question",
            "hypothesis",
            "proposed",
        ]

        # Look for recommendation/implication phrases
        recommendation_markers = [
            "recommend",
            "suggest",
            "propose",
            "future",
            "implication",
            "impact",
            "significance",
        ]

        # Count markers
        obj_count = sum(1 for marker in objective_markers if marker in text.lower())
        rec_count = sum(
            1 for marker in recommendation_markers if marker in text.lower()
        )

        # Calculate density of conclusion terms
        conclusion_terms = []
        for terms in self.conclusion_components.values():
            conclusion_terms.extend(terms)

        term_density = len(
            [token for token in doc if token.text.lower() in conclusion_terms]
        ) / len(doc)

        return min(1.0, (obj_count / 5 + rec_count / 5 + term_density) / 3)

    def _evaluate_with_llm(self, text: str) -> Dict:
        """Use LLM to evaluate conclusion quality"""
        prompt = f"""
        Evaluate this conclusion section based on:
        1. Objectives Achievement (objectives_score) (0-10):
           - Clear link to research objectives
           - Evidence of achievement
           - Comprehensive coverage
        2. Implications (implications_score) (0-10):
           - Clear statement of implications
           - Significance of findings
           - Impact discussion
        3. Recommendations (recommendations_score) (0-10):
           - Quality of recommendations
           - Future directions
           - Practical applications
        
        Output Format:
        {{
          objectives_score: number,
          implications_score: number,
          recommendations_score: number,
          justification: string
        }}

        Conclusion text:
        {text[:4000]}...

        Provide scores and brief justification in JSON format.
        """

        try:
            response = self._get_llm_scores(prompt)
            return eval(response)
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return {
                "objectives_score": 0,
                "implications_score": 0,
                "recommendations_score": 0,
                "justification": "Error in LLM evaluation",
            }

    def calculate_final_score(
        self,
        component_analysis: Dict,
        completeness_score: float,
        llm_scores: Dict = None,
    ) -> Tuple[float, str]:
        """Calculate final score and determine grade"""
        # Base scores (without LLM)
        component_score = component_analysis["overall_score"] * 10  # Scale to 10
        completeness_score = completeness_score * 10  # Scale to 10

        if self.use_llm and llm_scores:
            # Combine with LLM scores (scale from 10 to match rubric)
            llm_avg = (
                llm_scores["objectives_score"]
                + llm_scores["implications_score"]
                + llm_scores["recommendations_score"]
            ) / 3
            final_score = (
                component_score * 0.3 + completeness_score * 0.3 + llm_avg * 0.4
            )
        else:
            final_score = (component_score + completeness_score) / 2

        # Determine grade based on rubric
        if final_score >= 9:
            grade = "Distinction (9-10)"
        elif final_score >= 7:
            grade = "Distinction (7-8)"
        elif final_score >= 6:
            grade = "Merit (6)"
        elif final_score >= 5:
            grade = "Pass (5)"
        elif final_score >= 3:
            grade = "Fail (3-4)"
        else:
            grade = "Fail (0-2)"

        return final_score, grade

    def evaluate(self) -> Dict:
        """Perform complete evaluation of the conclusion section"""
        # Extract conclusion section
        conclusion_text = self.extract_conclusion_section()
        if not conclusion_text:
            return {
                "score": 0,
                "grade": "Fail (0-2)",
                "feedback": "Conclusion section not found",
            }

        # Analyze components
        component_analysis = self.analyze_components(conclusion_text)

        # Assess completeness
        completeness_score = self.assess_completeness(conclusion_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_llm(conclusion_text)

        # Calculate final score
        score, grade = self.calculate_final_score(
            component_analysis, completeness_score, llm_scores
        )

        # Generate feedback
        feedback = []
        missing_components = []
        for category, found_terms in component_analysis["components"].items():
            if not found_terms:
                missing_components.append(category.replace("_", " "))

        if missing_components:
            feedback.append(f"Missing or insufficient: {', '.join(missing_components)}")
        if completeness_score < 0.6:
            feedback.append(
                "Conclusion inadequately addresses objectives and recommendations"
            )

        if llm_scores:
            feedback.append(f"LLM Analysis: {llm_scores.get('justification', '')}")

        return {
            "score": round(score, 2),
            "grade": grade,
            "component_analysis": {
                "found_components": component_analysis["components"],
                "component_scores": component_analysis["scores"],
            },
            "completeness_score": round(completeness_score * 10, 2),  # Scale to 10
            "llm_scores": llm_scores,
            "feedback": ". ".join(feedback),
        }
