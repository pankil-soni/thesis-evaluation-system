# Thesis Evaluation System 📚

A comprehensive AI-powered tool designed to analyze and evaluate academic thesis documents through advanced natural language processing, machine learning, and rule-based evaluation techniques.

## 🌟 Key Features

- **Comprehensive Section Analysis**

  - Structure & Grammar evaluation
  - Introduction assessment
  - Literature Review analysis
  - Research Methods evaluation
  - Experiment Implementation review
  - Results Analysis
  - Conclusion assessment
  - Citations & References verification

- **AI-Enhanced Evaluation** powered by OpenAI's language models
- **Interactive Visualization** through Streamlit interface
- **Multilayered Scoring** with detailed quality insights

## 🛠 Technologies

- **Core**: Python
- **NLP**: spaCy, language_tool_python
- **AI**: OpenAI GPT
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Document Processing**: PyMuPDF

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/pankil-soni/thesis-evaluation-system.git
   cd thesis-evaluation-system
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download spaCy model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Configure OpenAI API:
   ```bash
   # Create .env file and add your API key
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

### Usage

Start the application:

```bash
streamlit run app.py
```

## 📊 Evaluation System

### Methodology

The evaluation process combines multiple approaches:

1. **Rule-Based Analysis**

   - Grammar and structure verification
   - Pattern matching
   - Language quality checks

2. **AI Analysis**

   - Contextual understanding via OpenAI GPT
   - Semantic section analysis
   - Qualitative feedback generation

3. **Visualization**
   - Interactive score displays
   - Section-wise performance radar charts
   - Detailed feedback presentation

### Project Structure

```
thesis-evaluation-system/
├── app.py                         # Main Streamlit application
├── evaluator.py                   # Base evaluation class
├── evaluators/
│   ├── structure_evaluator.py
│   ├── introduction_evaluator.py
│   ├── literature_review_evaluator.py
│   ├── research_methods_evaluator.py
│   ├── experiment_implementation_evaluator.py
│   ├── results_analysis_evaluator.py
│   ├── conclusion_evaluator.py
│   └── citation_references_evaluator.py
├── requirements.txt
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Project Maintainer: [Pankil Soni](mailto:pmsoni2016@gmail.com)

Project Link: [https://github.com/pankil-soni/thesis-evaluation-system](https://github.com/pankil-soni/thesis-evaluation-system)

---

**Note**: Ensure all dependencies are properly installed and API keys are configured before running the application.
