import pdfplumber
import spacy
import language_tool_python
import pypdf
from pdf2image import convert_from_path
import cv2
import numpy as np
from collections import defaultdict
import re
from textblob import TextBlob
import nltk
from collections import Counter
from openai import OpenAI
from typing import Dict, List, Tuple
import scholarly
from datetime import datetime