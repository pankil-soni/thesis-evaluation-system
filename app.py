import streamlit as st
import time
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from evaluator import BaseEvaluator
from structure_evaluator import StructureAndGrammarEvaluator
from introduction_evaluator import IntroductionEvaluator
from literature_review_evaluator import LiteratureReviewEvaluator
from research_methods_evaluator import ResearchMethodsEvaluator
from experiment_implementation_evaluator import ExperimentImplementationEvaluator
from results_analysis_evaluator import ResultsAnalysisEvaluator
from conclusion_evaluator import ConclusionEvaluator
from CitationReferencesEvaluator import CitationReferencesEvaluator

# Set page configuration
st.set_page_config(
    page_title="Thesis Evaluation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
        margin: 0.5rem 0;
    }
    .grade-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
    }
    .feedback-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    use_llm = st.toggle("Use LLM for Enhanced Evaluation", value=True)
    st.markdown("---")
    st.markdown(
        """
    ### About
    This system evaluates thesis documents based on multiple criteria:
    - Structure & Grammar
    - Introduction
    - Literature Review
    - Research Methods
    - Implementation
    - Results Analysis
    - Conclusion
    - Citations & References
    """
    )

# Main content
st.title("🎓 Thesis Evaluation System")

# File upload
uploaded_file = st.file_uploader("Upload your thesis PDF", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    temp_path = Path("temp.pdf")
    temp_path.write_bytes(uploaded_file.getvalue())

    # Initialize progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Initialize base evaluator
        base_evaluator = BaseEvaluator(str(temp_path))

        # Create all evaluators
        evaluators = {
            # "Structure & Grammar": StructureAndGrammarEvaluator(
            #     base_evaluator.pdf_path, use_llm=use_llm, base_instance=base_evaluator
            # ),
            "Introduction": IntroductionEvaluator(
                base_evaluator.pdf_path, use_llm=use_llm, base_instance=base_evaluator
            ),
            "Literature Review": LiteratureReviewEvaluator(
                base_evaluator.pdf_path, use_llm=use_llm, base_instance=base_evaluator
            ),
            "Research Methods": ResearchMethodsEvaluator(
                base_evaluator.pdf_path, use_llm=use_llm, base_instance=base_evaluator
            ),
            "Implementation": ExperimentImplementationEvaluator(
                base_evaluator.pdf_path, use_llm=use_llm, base_instance=base_evaluator
            ),
            "Results Analysis": ResultsAnalysisEvaluator(
                base_evaluator.pdf_path, use_llm=use_llm, base_instance=base_evaluator
            ),
            "Conclusion": ConclusionEvaluator(
                base_evaluator.pdf_path, use_llm=use_llm, base_instance=base_evaluator
            ),
            "Citations & References": CitationReferencesEvaluator(
                base_evaluator.pdf_path, use_llm=use_llm, base_instance=base_evaluator
            ),
        }

        # Evaluate each section
        results = {}
        for i, (section, evaluator) in enumerate(evaluators.items()):
            status_text.text(f"Evaluating {section}...")
            progress_bar.progress((i + 1) / len(evaluators))
            results[section] = evaluator.evaluate()
            time.sleep(0.1)  # Add small delay for visual feedback

        status_text.text("Evaluation complete! 🎉")
        progress_bar.progress(100)

        # Display Results
        st.markdown("## 📊 Evaluation Results")

        # Create three columns for high-level metrics
        col1, col2, col3 = st.columns(3)

        # Calculate total score and average
        total_score = sum(result["score"] for result in results.values())
        avg_score = total_score / len(results)

        with col1:
            st.metric("Total Score", f"{total_score:.2f}")
        with col2:
            st.metric("Average Score", f"{avg_score:.2f}")
        with col3:
            passing_sections = sum(
                1
                for result in results.values()
                if not result["grade"].startswith("Fail")
            )
            st.metric("Passing Sections", f"{passing_sections}/{len(results)}")

        # Detailed results in tabs
        tabs = st.tabs(list(results.keys()))
        for tab, (section, result) in zip(tabs, results.items()):
            with tab:
                # Section score and grade
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"### Score: {result['score']:.2f}")
                with col2:
                    st.markdown(f"### Grade: {result['grade']}")

                # Feedback
                st.markdown("### 📝 Feedback")
                st.markdown(
                    f"""
                <div class='feedback-box'>
                    {result['feedback']}
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Component analysis if available
                if "component_analysis" in result:
                    st.markdown("### 🔍 Component Analysis")
                    component_scores = result["component_analysis"]["component_scores"]
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=list(component_scores.keys()),
                                y=list(component_scores.values()),
                                marker_color="#4CAF50",
                            )
                        ]
                    )
                    fig.update_layout(
                        title="Component Scores",
                        xaxis_title="Components",
                        yaxis_title="Score",
                        yaxis_range=[0, 1],
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Overall visualization
        st.markdown("## 📈 Overall Performance")

        # Radar chart of all scores
        categories = list(results.keys())
        scores = [result["score"] for result in results.values()]

        fig = go.Figure(
            data=go.Scatterpolar(
                r=scores, theta=categories, fill="toself", marker_color="#4CAF50"
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(scores)])),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Download detailed report
        report_data = {
            "Section": categories,
            "Score": scores,
            "Grade": [results[section]["grade"] for section in categories],
            "Feedback": [results[section]["feedback"] for section in categories],
        }
        df = pd.DataFrame(report_data)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Detailed Report",
            data=csv,
            file_name="thesis_evaluation_report.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"An error occurred during evaluation: {str(e)}")
    finally:
        # Clean up
        temp_path.unlink(missing_ok=True)
else:
    # Display welcome message and instructions
    st.markdown(
        """
    ### Welcome to the Thesis Evaluation System! 👋
    
    Upload your thesis PDF to get started. The system will evaluate your work based on multiple criteria
    and provide detailed feedback for each section.
    
    #### Features:
    - Comprehensive evaluation of 8 key areas
    - Detailed feedback and suggestions
    - Visual performance analysis
    - Downloadable detailed report
    
    Toggle the LLM option in the sidebar for enhanced evaluation using advanced language models.
    """
    )
