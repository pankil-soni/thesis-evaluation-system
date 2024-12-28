from dotenv import load_dotenv

load_dotenv()

class BaseEvaluator:
    """Base class with common functionality"""

    def __init__(self, pdf_path, use_llm: bool = True):
        self.nlp = spacy.load("en_core_web_sm")
        self.use_llm = use_llm
        self.pdf_path = pdf_path
        self.full_text = ""
        if use_llm:
            self.open_ai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        if self.full_text == "":
            if self._extract_text():
                print("Text extracted successfully ✅")
            else:
                raise Exception("Error extracting text ❌")  # throw exception

    def _extract_text(self) -> bool:
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.full_text = "\n".join(page.extract_text() for page in pdf.pages)
            return True
        except Exception as e:
            print(f"Error extracting text: {e}")
            return False

    def _clean_section_text(self, text: str) -> str:
        """Clean up the extracted section text"""
        text = re.sub(r"\d+\.\d+\s+.*?\n", "", text)
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_section(
        self, full_text: str, start_patterns: List[str], end_patterns: List[str]
    ) -> str:
        """Generic section extraction"""
        try:
            section_text = ""
            in_section = False
            page_texts = full_text.split("\f")

            for page_text in page_texts:
                lines = page_text.split("\n")

                for line in lines:
                    line = line.strip()

                    if re.match(r"^\d+$", line):
                        continue

                    if not in_section:
                        for pattern in start_patterns:
                            if re.match(pattern, line):
                                in_section = True
                                print(pattern, line)
                                break
                        continue

                    if in_section:
                        for pattern in end_patterns:
                            if re.match(pattern, line):
                                in_section = False
                                break

                        if not in_section:
                            break

                        if line and not re.match(r"^\d+\.\d+\s+", line):
                            section_text += line + "\n"

            return self._clean_section_text(section_text)

        except Exception as e:
            print(f"Error extracting implementation section: {e}")
            return ""
