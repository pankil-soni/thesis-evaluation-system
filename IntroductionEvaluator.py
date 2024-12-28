from Evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple


class IntroductionEvaluator(BaseEvaluator):
    def __init__(self, pdf_path, use_llm: bool = True, base_instance=None):
        """
        Initialize the evaluator with option to use LLM
        """
        super().__init__(pdf_path, use_llm, base_instance)

    def extract_introduction(self) -> str:
        """
        Extract the introduction section accurately using multiple extraction methods
        """

        # Patterns for identifying introduction section
        intro_patterns = [
            r"(?i)^(?:CHAPTER\s+1\.?\s*)?INTRODUCTION\s*$",
            r"(?i)^(?:1\.?\s+)?INTRODUCTION\s*$",
            r"(?i)^(?:CHAPTER\s+ONE\.?\s*)?INTRODUCTION\s*$",
            r"(?i)^(?:1\.0\s+)?INTRODUCTION\s*$",
            r"(?i)^(?:CHAPTER\s+1:?\s*)?INTRODUCTION\s*$",
        ]

        # Patterns for next section (to identify end of introduction)
        next_section_patterns = [
            r"(?i)^(?:CHAPTER\s+2)",
            r"(?i)^(?:2\.?\s+)",
            r"(?i)^(?:LITERATURE\s+REVIEW)",
            r"(?i)^(?:THEORETICAL\s+FRAMEWORK)",
            r"(?i)^(?:RESEARCH\s+METHODOLOGY)",
            r"(?i)^(?:METHODOLOGY)",
            r"(?i)^(?:BACKGROUND)",
        ]

        try:
            introduction_text = self._extract_section(
                self.full_text, intro_patterns, next_section_patterns
            )

            # Validate the extracted text
            if self._validate_introduction_content(introduction_text):
                return introduction_text
            else:
                return ""

        except Exception as e:
            print(f"Error in extract_introduction: {e}")
            return ""

    def _validate_introduction_content(self, text: str) -> bool:
        """
        Validate that the extracted text is actually an introduction
        """
        # Check minimum length
        if len(text.split()) < 50:
            return False

        # Check for common introduction indicators
        intro_indicators = [
            r"(?i)research",
            r"(?i)study",
            r"(?i)problem",
            r"(?i)purpose",
            r"(?i)objective",
            r"(?i)background",
        ]

        # Count how many indicators are present
        indicator_count = sum(
            1 for pattern in intro_indicators if re.search(pattern, text)
        )

        # Text should contain at least 2 introduction indicators
        return indicator_count >= 2

    def _check_research_components(self, intro_text: str) -> Dict:
        """
        Check for presence and quality of key research components
        """
        components = {
            "problem_statement": False,
            "research_questions": False,
            "objectives": False,
            "justification": False,
        }

        # Keywords and patterns for each component
        patterns = {
            "problem_statement": [
                r"problem statement",
                r"(?i)problem statement",
                r"(?i)research problem",
                r"(?i)this study addresses",
                r"(?i)the problem is",
                r"(?i)the main issue",
            ],
            "research_questions": [
                r"(?i)research question",
                r"(?i)\?",
                r"(?i)this study seeks to",
                r"(?i)we investigate",
                r"(?i)aims to answer",
            ],
            "objectives": [
                r"(?i)objective",
                r"(?i)aim of this",
                r"(?i)purpose of this",
                r"(?i)goal of this",
                r"(?i)this study aims",
            ],
            "justification": [
                r"(?i)significance",
                r"(?i)importance",
                r"(?i)justification",
                r"(?i)rationale",
                r"(?i)this is important because",
            ],
        }

        # Check for each component
        for component, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, intro_text):
                    components[component] = True
                    break

        return components

    def _analyze_coherence(self, intro_text: str) -> float:
        """
        Analyze the coherence and flow of the introduction
        """
        doc = self.nlp(intro_text)

        # Check for transition words and connectors
        transition_words = set(
            [
                "however",
                "therefore",
                "furthermore",
                "moreover",
                "additionally",
                "consequently",
                "thus",
                "hence",
            ]
        )

        # Count transitions
        transitions = sum(1 for token in doc if token.text.lower() in transition_words)

        # Count sentences
        sentences = len(list(doc.sents))

        if sentences == 0:
            return 0

        # Calculate transition density
        transition_density = transitions / sentences

        return min(1.0, transition_density)

    def _evaluate_with_llm(self, intro_text: str) -> Dict:
        """
        Use LLM to evaluate the introduction quality
        """
        prompt = f"""
        Evaluate this thesis introduction based on the following criteria:
        1. Problem formulation clarity (problem_score) (0-10)
        2. Research questions quality (questions_score) (0-10)
        3. Objectives clarity (objectives_score) (0-10)
        4. Project justification strength (justification_score) (0-10)

        Introduction text:
        {intro_text}

        Provide scores and brief justification in JSON format.
        """

        try:
            response = self._get_llm_scores(prompt)
            return eval(response)
        except Exception as e:
            print(e)
            return {
                "problem_score": 0,
                "questions_score": 0,
                "objectives_score": 0,
                "justification_score": 0,
                "justification": "Error in LLM evaluation",
            }

    def _calculate_final_score(
        self, components: Dict, coherence: float, llm_scores: Dict = None
    ) -> Tuple[float, str]:
        """
        Calculate final score and determine grade
        """
        # Base score from components presence
        base_score = sum(components.values()) * 1.5  # Max 6 points

        # Coherence score
        coherence_score = coherence * 2  # Max 2 points

        if self.use_llm and llm_scores:
            # Combine with LLM scores
            llm_avg = (
                np.mean(
                    [
                        llm_scores["problem_score"],
                        llm_scores["questions_score"],
                        llm_scores["objectives_score"],
                        llm_scores["justification_score"],
                    ]
                )
                / 10
            )  # Normalize to 0-1

            final_score = base_score * 0.3 + coherence_score * 0.2 + llm_avg * 10 * 0.5
        else:
            final_score = base_score + coherence_score + 2  # Max 10 points

        # Determine grade
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
        """
        Perform complete evaluation of the introduction
        """
        # Extract introduction
        intro_text = self.extract_introduction()
        if not intro_text:
            return {
                "score": 0,
                "grade": "Fail (0-2)",
                "feedback": "Introduction section not found",
            }

        # Check components
        components = self._check_research_components(intro_text)

        # Analyze coherence
        coherence = self._analyze_coherence(intro_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_llm(intro_text)

        # Calculate final score
        score, grade = self._calculate_final_score(components, coherence, llm_scores)

        # Prepare feedback
        missing_components = [k for k, v in components.items() if not v]
        feedback = []

        if missing_components:
            feedback.append(f"Missing components: {', '.join(missing_components)}")
        if coherence < 0.3:
            feedback.append("Low coherence and flow in writing")

        if llm_scores:
            feedback.append(f"LLM Analysis: {llm_scores.get('justification', '')}")

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "components_present": components,
            "coherence_score": float(round(coherence, 2)),
            "llm_scores": llm_scores if llm_scores else None,
            "feedback": " ".join(feedback),
        }
