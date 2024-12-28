from Evaluator import BaseEvaluator
from StructureEvaluator import StructureAndGrammarEvaluator
from IntroductionEvaluator import IntroductionEvaluator
from ResearchMethodsEvaluator import ResearchMethodsEvaluator
from ExperimentImplementationEvaluator import ExperimentImplementationEvaluator
from ResultsAnalysisEvaluator import ResultsAnalysisEvaluator

base_evaluator = BaseEvaluator("AI ML Thesis Report.pdf")
structure_evaluator = StructureAndGrammarEvaluator(
    base_evaluator.pdf_path, use_llm=False, base_instance=base_evaluator
)
structure_evaluator.evaluate()
