from evaluator import BaseEvaluator
import re
import numpy as np
from typing import Dict, Tuple, List
from collections import defaultdict


class IntroductionEvaluator(BaseEvaluator):
    def __init__(self, pdf_path, use_llm: bool = True, base_instance=None):
        super().__init__(pdf_path, use_llm, base_instance)
        self.quality_indicators = self._initialize_quality_indicators()

    def _initialize_quality_indicators(self) -> Dict:
        """Initialize comprehensive quality indicators for each component"""
        return {
            "problem_statement": {
                "essential": [
                    r"(?i)problem\s+statement",
                    r"(?i)research\s+problem",
                    r"(?i)gap\s+in(\s+the)?\s+literature",
                    r"(?i)current\s+challenges",
                    r"(?i)this\s+study\s+addresses",
                ],
                "context": [
                    r"(?i)previous\s+research",
                    r"(?i)existing\s+literature",
                    r"(?i)recent\s+studies",
                    r"(?i)background",
                ],
                "specificity": [
                    r"(?i)specifically",
                    r"(?i)in\s+particular",
                    r"(?i)notably",
                    r"(?i)precisely",
                ],
                "scope": [
                    r"(?i)scope\s+of(\s+the)?\s+study",
                    r"(?i)limitations",
                    r"(?i)boundaries",
                    r"(?i)constraints",
                ],
            },
            "research_questions": {
                "clarity": [
                    r"(?i)research\s+questions?",
                    r"(?i)questions?\s+to\s+be\s+addressed",
                    r"(?i)this\s+study\s+seeks\s+to",
                ],
                "specificity": [
                    r"(?i)specifically.*\?",
                    r"(?i)in\s+particular.*\?",
                    r"(?i)how\s+does.*\?",
                    r"(?i)what\s+is.*\?",
                    r"(?i)why\s+does.*\?",
                ],
                "alignment": [
                    r"(?i)align.*with.*objective",
                    r"(?i)address.*problem",
                    r"(?i)investigate.*gap",
                ],
            },
            "objectives": {
                "clarity": [
                    r"(?i)objectives?(\s+of)?(\s+the)?\s+study",
                    r"(?i)aims?(\s+of)?(\s+the)?\s+study",
                    r"(?i)purpose(\s+of)?(\s+the)?\s+study",
                ],
                "measurability": [
                    r"(?i)measure",
                    r"(?i)analyze",
                    r"(?i)evaluate",
                    r"(?i)assess",
                    r"(?i)determine",
                ],
                "achievability": [
                    r"(?i)within\s+the\s+scope",
                    r"(?i)feasible",
                    r"(?i)attainable",
                    r"(?i)realistic",
                ],
            },
            "justification": {
                "significance": [
                    r"(?i)significance(\s+of)?(\s+the)?\s+study",
                    r"(?i)importance(\s+of)?(\s+the)?\s+study",
                    r"(?i)contribution(\s+to)?(\s+the)?\s+field",
                ],
                "impact": [
                    r"(?i)impact\s+on",
                    r"(?i)benefits?\s+to",
                    r"(?i)implications?\s+for",
                    r"(?i)potential\s+applications?",
                ],
                "novelty": [
                    r"(?i)novel",
                    r"(?i)innovative",
                    r"(?i)unique",
                    r"(?i)original",
                    r"(?i)new\s+approach",
                ],
            },
        }

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

    def _evaluate_component_quality(
        self, text: str, component_indicators: Dict
    ) -> Tuple[float, Dict]:
        """
        Evaluate the quality of a specific component using weighted criteria
        """
        scores = {}
        details = defaultdict(list)

        for aspect, patterns in component_indicators.items():
            matches = 0
            total_patterns = len(patterns)

            for pattern in patterns:
                found_matches = re.finditer(pattern, text)
                for match in found_matches:
                    context = text[
                        max(0, match.start() - 50) : min(len(text), match.end() + 50)
                    ]
                    matches += 1
                    details[aspect].append(context.strip())

            # Calculate normalized score for this aspect
            scores[aspect] = min(1.0, matches / total_patterns)

        # Weight the aspects differently based on their importance
        weights = {
            "essential": 0.4,
            "clarity": 0.35,
            "specificity": 0.3,
            "context": 0.25,
            "scope": 0.2,
            "measurability": 0.3,
            "achievability": 0.25,
            "significance": 0.35,
            "impact": 0.3,
            "novelty": 0.25,
            "alignment": 0.3,
        }

        # Calculate weighted average
        total_weight = sum(weights.get(aspect, 0.25) for aspect in scores.keys())
        weighted_score = (
            sum(score * weights.get(aspect, 0.25) for aspect, score in scores.items())
            / total_weight
        )

        return weighted_score * 10, dict(details)  # Scale to 0-10

    def _analyze_structural_coherence(self, intro_text: str) -> float:
        """
        Analyze the structural coherence of the introduction
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in intro_text.split("\n\n") if p.strip()]
        if not paragraphs:
            return 0.0

        # Expected flow patterns
        flow_patterns = [
            (r"(?i)background|context|overview", 0),  # Should appear early
            (
                r"(?i)problem\s+statement|gap|challenge",
                1,
            ),  # Should appear after background
            (
                r"(?i)research\s+questions?|objectives?|aims?",
                2,
            ),  # Should appear in middle
            (r"(?i)significance|importance|justification", 3),  # Should appear later
            (r"(?i)structure|organization|outline", 4),  # Should appear at end
        ]

        # Score based on proper ordering
        flow_score = 0
        total_patterns = len(flow_patterns)

        for i, paragraph in enumerate(paragraphs):
            normalized_position = i / len(paragraphs)

            for pattern, expected_position in flow_patterns:
                if re.search(pattern, paragraph):
                    expected_normalized = expected_position / total_patterns
                    # Calculate how close the actual position is to the expected position
                    position_score = 1 - min(
                        1, abs(normalized_position - expected_normalized)
                    )
                    flow_score += position_score

        # Normalize score
        return min(1.0, flow_score / total_patterns)

    def _evaluate_with_llm(self, intro_text: str) -> Dict:
        """LLM evaluation with more specific criteria"""
        prompt = f"""
        Evaluate this thesis introduction based on these detailed criteria:
        
        1. Problem Formulation (0-10):
           - Clarity of problem statement
           - Context and background
           - Scope definition
           - Identification of research gap
        
        2. Research Questions (0-10):
           - Clarity and specificity
           - Alignment with problem
           - Feasibility
           - Logical progression
        
        3. Objectives (0-10):
           - Clear articulation
           - Measurability
           - Achievability
           - Alignment with questions
        
        4. Project Justification (0-10):
           - Scientific/practical significance
           - Innovation/contribution
           - Impact potential
           - Stakeholder relevance

        Introduction text:
        {intro_text}


        Output Format:
        {{
            "problem_score": float,
            "questions_score": float,
            "objectives_score": float,
            "justification_score": float,
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
                "problem_score": 0,
                "questions_score": 0,
                "objectives_score": 0,
                "justification_score": 0,
                "justification": "Error in LLM evaluation",
                "strengths": [],
                "improvements": [],
            }

    def _calculate_final_score(
        self,
        component_scores: Dict[str, float],
        coherence: float,
        llm_scores: Dict = None,
    ) -> Tuple[float, str, List[str]]:
        """
        Calculate final score with detailed feedback
        """
        # Base component weights
        weights = {
            "problem_statement": 0.3,
            "research_questions": 0.25,
            "objectives": 0.25,
            "justification": 0.2,
        }

        # Calculate weighted component score
        component_score = sum(
            score * weights[component] for component, score in component_scores.items()
        )

        # Coherence contribution
        coherence_score = coherence * 10  # Scale to 0-10

        if self.use_llm and llm_scores:
            llm_component_scores = np.mean(
                [
                    llm_scores["problem_score"],
                    llm_scores["questions_score"],
                    llm_scores["objectives_score"],
                    llm_scores["justification_score"],
                ]
            )

            # Weighted combination
            final_score = (
                component_score * 0.4  # Manual evaluation
                + coherence_score * 0.2  # Structural coherence
                + llm_component_scores * 0.4  # LLM evaluation
            )
        else:
            final_score = component_score * 0.7 + coherence_score * 0.3

        # Generate detailed feedback
        feedback = []
        strengths = []
        improvements = []

        # Component-specific feedback
        for component, score in component_scores.items():
            if score >= 8:
                strengths.append(f"Strong {component.replace('_', ' ')} formulation")
            elif score <= 5:
                improvements.append(
                    f"Enhance {component.replace('_', ' ')} formulation"
                )

        # Coherence feedback
        if coherence >= 0.8:
            strengths.append("Excellent structural flow and coherence")
        elif coherence <= 0.5:
            improvements.append("Improve logical flow between sections")

        # Determine grade based on rubric
        if final_score >= 9:
            grade = "Distinction (9-10)"
            feedback.append("Outstanding formulation across all components")
        elif final_score >= 7:
            grade = "Distinction (7-8)"
            feedback.append("Good formulation with minor deficiencies")
        elif final_score >= 6:
            grade = "Merit (6)"
            feedback.append("Well-written but some components need improvement")
        elif final_score >= 5:
            grade = "Pass (5)"
            feedback.append("Satisfactory but major deficiencies exist")
        elif final_score >= 3:
            grade = "Fail (3-4)"
            feedback.append("Inadequate problem identification and justification")
        else:
            grade = "Fail (0-2)"
            feedback.append("Weak formulation with major deficiencies")

        return final_score, grade, feedback

    def evaluate(self) -> Dict:
        """
        Perform evaluation of the introduction
        """
        intro_text = self.extract_introduction()
        if not intro_text:
            return {
                "score": 0,
                "grade": "Fail (0-2)",
                "feedback": ["Introduction section not found"],
                "details": {},
            }

        # Evaluate each component's quality
        component_scores = {}
        component_details = {}
        for component, indicators in self.quality_indicators.items():
            score, details = self._evaluate_component_quality(intro_text, indicators)
            component_scores[component] = score
            component_details[component] = details

        # Analyze structural coherence
        coherence = self._analyze_structural_coherence(intro_text)

        # LLM evaluation if enabled
        llm_scores = None
        if self.use_llm:
            llm_scores = self._evaluate_with_llm(intro_text)

        # Calculate final score and generate feedback
        score, grade, feedback = self._calculate_final_score(
            component_scores, coherence, llm_scores
        )

        return {
            "score": float(round(score, 2)),
            "grade": grade,
            "feedback": feedback,
            "component_scores": {k: round(v, 2) for k, v in component_scores.items()},
            "coherence_score": float(round(coherence * 10, 2)),
            "llm_scores": llm_scores if llm_scores else None,
            "details": component_details,
        }
