import streamlit as st
import pandas as pd
from pathlib import Path
import traceback
import plotly.graph_objects as go
import plotly.express as px
from evaluator import BaseEvaluator
from structure_evaluator import StructureAndGrammarEvaluator
from introduction_evaluator import IntroductionEvaluator
from literature_review_evaluator import LiteratureReviewEvaluator
from research_methods_evaluator import ResearchMethodsEvaluator
from experiment_implementation_evaluator import ExperimentImplementationEvaluator
from results_analysis_evaluator import ResultsAnalysisEvaluator
from conclusion_evaluator import ConclusionEvaluator
from citation_references_evaluator import CitationReferencesEvaluator
import numpy as np
import io


# Set page configuration
st.set_page_config(
    page_title="Thesis Evaluation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced Custom CSS
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
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 0.25rem 0.5rem rgba(0,0,0,0.05);
        margin: 0.75rem 0;
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .grade-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 1rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .feedback-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 6px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 0.25rem 0.5rem rgba(0,0,0,0.05);
    }
    .section-header {
        padding: 1rem;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .score-pill {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: 600;
        font-size: 0.875rem;
    }
    .detail-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid #f0f0f0;
        margin: 0.75rem 0;
    }
    .chart-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 0.25rem 0.5rem rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


def create_score_gauge(score, max_score, title):
    """Create a gauge chart for scores"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, max_score]},
                "bar": {"color": "#4CAF50"},
                "steps": [
                    {"range": [0, max_score / 3], "color": "#ffebee"},
                    {"range": [max_score / 3, 2 * max_score / 3], "color": "#e8f5e9"},
                    {"range": [2 * max_score / 3, max_score], "color": "#c8e6c9"},
                ],
            },
            title={"text": title},
        )
    )
    fig.update_layout(height=250)
    return fig


def create_spider_chart(categories, values, title):
    """Create a spider/radar chart"""
    fig = go.Figure(
        data=go.Scatterpolar(
            r=values, theta=categories, fill="toself", marker_color="#4CAF50"
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values)])),
        showlegend=False,
        title=title,
    )
    return fig


def display_structure_grammar_section(result):
    """Display Structure & Grammar section details"""
    st.markdown("### 📝 Structure Details")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.plotly_chart(
            create_score_gauge(
                result["details"]["structure"]["score"], 5, "Structure Score"
            ),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            create_score_gauge(result["details"]["formatting"], 5, "Formatting Score"),
            use_container_width=True,
        )
    with col3:
        st.plotly_chart(
            create_score_gauge(
                result["details"]["grammar_spelling"], 5, "Grammar Score"
            ),
            use_container_width=True,
        )
    with col4:
        st.plotly_chart(
            create_score_gauge(result["details"]["style"], 5, "Style Score"),
            use_container_width=True,
        )

    st.markdown("### 📋 Found Sections")
    st.write(result["details"]["structure"]["found_sections"])


def display_introduction_section(result):
    """Display Introduction section details"""
    st.markdown("### 🎯 Component Scores")

    # Component scores visualization
    fig = px.bar(
        x=list(result["component_scores"].keys()),
        y=list(result["component_scores"].values()),
        labels={"x": "Components", "y": "Score"},
        title="Component Analysis",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Coherence score
    st.markdown("### 🔄 Coherence Analysis")
    st.plotly_chart(
        create_score_gauge(result["coherence_score"], 10, "Coherence Score"),
        use_container_width=True,
    )

    # LLM Scores if available
    if result["llm_scores"]:
        st.markdown("### 🤖 LLM Analysis")
        st.json(result["llm_scores"])


def display_literature_review_section(result):
    """Display Literature Review section details"""
    st.markdown("### 📚 Review Analysis")

    # Systematic score gauge
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            create_score_gauge(
                result["systematic_score"], 15, "Systematic Review Score"
            ),
            use_container_width=True,
        )

    # Citation metrics
    with col2:
        st.markdown("### 📊 Citation Metrics")
        st.write(result["citation_metrics"])

    # Evidence details visualization
    if result["evidence_details"]:
        st.markdown("### 🔍 Evidence Analysis")
        fig = px.bar(
            x=list(result["evidence_details"].keys()),
            y=[len(v) for v in result["evidence_details"].values()],
            title="Evidence Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)


def display_methods_section(result):
    """Display Research Methods section details"""
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_score_gauge(result["methodology_score"], 20, "Methodology Score"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            create_score_gauge(result["structure_score"], 20, "Structure Score"),
            use_container_width=True,
        )

    # Quality details visualization
    if result["quality_details"]:
        st.markdown("### 🔍 Quality Analysis")
        for category, details in result["quality_details"].items():
            st.markdown(f"#### {category}")
            st.write(details)


def display_implementation_section(result):
    """Display Implementation section details"""
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_score_gauge(
                result["implementation_score"], 20, "Implementation Score"
            ),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            create_score_gauge(result["explanation_score"], 20, "Explanation Score"),
            use_container_width=True,
        )

    # Quality details visualization
    if result["quality_details"]:
        st.markdown("### 🛠️ Implementation Quality Analysis")
        fig = go.Figure()
        for category, scores in result["quality_details"].items():
            if isinstance(scores, dict):
                fig.add_trace(
                    go.Bar(
                        name=category, x=list(scores.keys()), y=list(scores.values())
                    )
                )
        fig.update_layout(title="Quality Metrics Distribution")
        st.plotly_chart(fig, use_container_width=True)


def display_results_section(result):
    """Display Results Analysis section details"""
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_score_gauge(result["analysis_score"], 15, "Analysis Score"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            create_score_gauge(
                result["argumentation_score"], 15, "Argumentation Score"
            ),
            use_container_width=True,
        )

    # Quality details visualization
    if result["quality_details"]:
        st.markdown("### 📊 Results Quality Analysis")
        for category, details in result["quality_details"].items():
            st.markdown(f"#### {category}")
            if isinstance(details, dict):
                # Ensure data is properly structured for plotting
                try:
                    # Convert data to format suitable for plotting
                    data_dict = {
                        "Category": list(details.keys()),
                        "Value": list(details.values()),
                    }
                    fig = px.bar(
                        data_frame=pd.DataFrame(data_dict),
                        x="Category",
                        y="Value",
                        title=f"{category} Distribution",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(
                        f"Could not create visualization for {category}: {str(e)}"
                    )
                    # Display raw data instead
                    st.write(details)
            else:
                # If not a dictionary, display raw data
                st.write(details)


def display_conclusion_section(result):
    """Display Conclusion section details"""
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_score_gauge(result["conclusion_score"], 10, "Conclusion Score"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            create_score_gauge(result["coherence_score"], 10, "Coherence Score"),
            use_container_width=True,
            key="coherence_score",
        )

    # Quality details visualization
    if result["quality_details"]:
        st.markdown("### 🎯 Conclusion Quality Analysis")
        for category, details in result["quality_details"].items():
            st.markdown(f"#### {category}")
            if isinstance(details, dict):
                fig = px.bar(
                    x=list(details.keys()),
                    y=[len(v) if isinstance(v, list) else v for v in details.values()],
                    title=f"{category} Analysis",
                )
                st.plotly_chart(fig, use_container_width=True)


def display_citations_section(result):
    """Display Citations & References section details"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(
            create_score_gauge(result["style_consistency"], 5, "Style Consistency"),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            create_score_gauge(result["reference_quality"], 5, "Reference Quality"),
            use_container_width=True,
        )

    with col3:
        st.plotly_chart(
            create_score_gauge(result["temporal_score"], 5, "Temporal Score"),
            use_container_width=True,
        )

    # Detailed analysis
    if result["details"]:
        st.markdown("### 📚 Citation Analysis")

        # Style analysis
        if "style" in result["details"]:
            fig = px.bar(
                x=list(result["details"]["style"]["counts"].keys()),
                y=list(result["details"]["style"]["counts"].values()),
                title="Citation Style Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Temporal analysis
        if "temporal" in result["details"]:
            fig = px.line(
                x=list(result["details"]["temporal"]["years"].keys()),
                y=list(result["details"]["temporal"]["years"].values()),
                title="Citation Year Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)


# Main evaluation display function
def display_evaluation_results(results):
    """Display all evaluation results with enhanced visualizations"""
    st.markdown("## 📊 Overall Evaluation Dashboard")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    total_score = sum(result["score"] for result in results.values())
    avg_score = total_score / len(results)
    passing_sections = sum(
        1 for result in results.values() if not result["grade"].startswith("Fail")
    )

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Total Score</h3>
                <h2>{total_score:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Average Score</h3>
                <h2>{avg_score:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Passing Sections</h3>
                <h2>{passing_sections}/{len(results)}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>Overall Grade</h3>
                <h2>{"Pass" if avg_score >= 5 else "Fail"}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Overall Performance Radar Chart
    st.markdown("### 📊 Overall Performance Analysis")
    categories = list(results.keys())
    scores = [result["score"] for result in results.values()]
    radar_fig = create_spider_chart(categories, scores, "Section Scores Distribution")
    st.plotly_chart(radar_fig, use_container_width=True)

    # Section-wise Analysis
    st.markdown("## 📑 Detailed Section Analysis")

    # Create tabs for each section
    tabs = st.tabs(list(results.keys()))
    for tab, (section, result) in zip(tabs, results.items()):
        with tab:
            # Section header
            st.markdown(
                f"""
                <div class="section-header">
                    <h2>{section}</h2>
                    <div class="grade-badge">{result['grade']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Score summary
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>Score</h3>
                        <h2>{result['score']:.2f}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="feedback-box">
                        <h3>Feedback</h3>
                        <p>{result.get("feedback","")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Section-specific visualizations
            if section == "Structure & Grammar":
                display_structure_grammar_section(result)
            elif section == "Introduction":
                display_introduction_section(result)
            elif section == "Literature Review":
                display_literature_review_section(result)
            elif section == "Research Methods":
                display_methods_section(result)
            elif section == "Implementation":
                display_implementation_section(result)
            elif section == "Results Analysis":
                display_results_section(result)
            elif section == "Conclusion":
                display_conclusion_section(result)
            elif section == "Citations & References":
                display_citations_section(result)

            # LLM Scores if available
            if "llm_scores" in result and result["llm_scores"]:
                st.markdown("### 🤖 LLM Analysis")
                st.markdown(
                    """
                    <div class="detail-card">
                        <h4>AI-Enhanced Evaluation</h4>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.json(result["llm_scores"])

    # Comparative Analysis
    st.markdown("## 📈 Comparative Analysis")

    # Score distribution
    score_dist_fig = px.bar(
        x=categories,
        y=scores,
        title="Score Distribution Across Sections",
        labels={"x": "Sections", "y": "Score"},
        color=scores,
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(score_dist_fig, use_container_width=True)

    # Create downloadable report
    st.markdown("## 📥 Detailed Report")

    report_data = {
        "Section": categories,
        "Score": scores,
        "Grade": [results[section]["grade"] for section in categories],
        "Feedback": [results[section].get("feedback", "") for section in categories],
    }

    for section in results.keys():
        report_data["Section"].append(section)
        report_data["Score"].append(results[section]["score"])
        report_data["Grade"].append(results[section]["grade"])
        report_data["Feedback"].append(results[section].get("feedback", ""))

    # Create additional metrics dictionary to track available metrics
    additional_metrics = {}

    # Identify common metrics across sections
    for section, result in results.items():
        for key, value in result.items():
            if key not in ["score", "grade", "feedback"] and not isinstance(
                value, (dict, list)
            ):
                if key not in additional_metrics:
                    additional_metrics[key] = {}
                additional_metrics[key][section] = value

    # Add additional metrics only if they exist for all sections
    for metric, values in additional_metrics.items():
        if len(values) == len(results):  # Only add if metric exists for all sections
            report_data[metric] = []
            for section in results.keys():
                report_data[metric].append(values.get(section, None))

    # Create DataFrame
    try:
        df = pd.DataFrame(report_data)

        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name="thesis_evaluation_report.csv",
                mime="text/csv",
            )

        with col2:
            # Create Excel report
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Evaluation Report")
            excel_data = buffer.getvalue()

            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name="thesis_evaluation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error creating report: {str(e)}")
        st.write("Raw Data:", report_data)

    # Add a more detailed report section
    st.markdown("### 📊 Detailed Metrics")
    st.dataframe(df, use_container_width=True)

    # Add metrics visualizations
    st.markdown("### 📈 Key Metrics Visualization")

    # Create metrics visualization
    try:
        metrics_fig = go.Figure()

        # Add score bar
        metrics_fig.add_trace(
            go.Bar(name="Score", x=df["Section"], y=df["Score"], marker_color="#4CAF50")
        )

        # Update layout
        metrics_fig.update_layout(
            title="Scores by Section",
            xaxis_title="Section",
            yaxis_title="Score",
            barmode="group",
            showlegend=True,
        )

        st.plotly_chart(metrics_fig, use_container_width=True)

        # Add additional visualizations for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:  # If we have more than just the score
            for col in numeric_cols:
                if col != "Score":  # Skip the main score we already plotted
                    metric_fig = go.Figure()
                    metric_fig.add_trace(
                        go.Bar(
                            x=df["Section"], y=df[col], name=col, marker_color="#2196F3"
                        )
                    )
                    metric_fig.update_layout(
                        title=f"{col} by Section",
                        xaxis_title="Section",
                        yaxis_title=col,
                    )
                    st.plotly_chart(metric_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")


# Main app
def main():
    st.title("🎓 Thesis Evaluation System")

    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        use_llm = st.toggle("Use LLM for Enhanced Evaluation", value=True)

        st.markdown("---")
        st.markdown(
            """
            ### About
            This enhanced system provides comprehensive thesis evaluation with:
            - Detailed section analysis
            - Interactive visualizations
            - AI-enhanced feedback
            - Comparative analytics
            - Downloadable reports
            """
        )

    # File upload
    uploaded_file = st.file_uploader("Upload your thesis PDF", type="pdf")

    if uploaded_file:
        # Save uploaded file temporarily
        temp_path = Path("temp.pdf")
        temp_path.write_bytes(uploaded_file.getvalue())

        try:
            with st.spinner("Evaluating thesis..."):
                # Initialize evaluators and process
                base_evaluator = BaseEvaluator(str(temp_path))
                evaluators = {
                    "Structure & Grammar": StructureAndGrammarEvaluator(
                        base_evaluator.pdf_path,
                        use_llm=use_llm,
                        base_instance=base_evaluator,
                    ),
                    "Introduction": IntroductionEvaluator(
                        base_evaluator.pdf_path,
                        use_llm=use_llm,
                        base_instance=base_evaluator,
                    ),
                    "Literature Review": LiteratureReviewEvaluator(
                        base_evaluator.pdf_path,
                        use_llm=use_llm,
                        base_instance=base_evaluator,
                    ),
                    "Research Methods": ResearchMethodsEvaluator(
                        base_evaluator.pdf_path,
                        use_llm=use_llm,
                        base_instance=base_evaluator,
                    ),
                    "Implementation": ExperimentImplementationEvaluator(
                        base_evaluator.pdf_path,
                        use_llm=use_llm,
                        base_instance=base_evaluator,
                    ),
                    "Results Analysis": ResultsAnalysisEvaluator(
                        base_evaluator.pdf_path,
                        use_llm=use_llm,
                        base_instance=base_evaluator,
                    ),
                    "Conclusion": ConclusionEvaluator(
                        base_evaluator.pdf_path,
                        use_llm=use_llm,
                        base_instance=base_evaluator,
                    ),
                    "Citations & References": CitationReferencesEvaluator(
                        base_evaluator.pdf_path,
                        use_llm=use_llm,
                        base_instance=base_evaluator,
                    ),
                }

                # Evaluate each section
                results = {
                    section: evaluator.evaluate()
                    for section, evaluator in evaluators.items()
                }

                # Display results
                display_evaluation_results(results)

        except Exception as e:
            traceback.print_exc()
            st.error(f"An error occurred during evaluation: {str(e)}")
        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)
    else:
        # Welcome message
        st.markdown(
            """
            ### Welcome to the Enhanced Thesis Evaluation System! 👋
            
            Upload your thesis PDF to get started. Our system provides:
            
            #### Features ✨
            - Comprehensive evaluation of 8 key areas
            - Interactive visualizations and analytics
            - AI-enhanced feedback
            - Detailed component analysis
            - Comparative performance metrics
            - Exportable detailed reports
            
            #### How to Use 📝
            1. Toggle LLM option in sidebar if desired
            2. Upload your thesis PDF
            3. Wait for the analysis to complete
            4. Explore interactive results
            5. Download detailed reports
            
            Get started by uploading your thesis PDF above!
            """
        )


if __name__ == "__main__":
    main()
