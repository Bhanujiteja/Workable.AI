import unittest
from extractor  import generate_latex_questions_from_text  # Update with actual import

class TestLatexFormatting(unittest.TestCase):

    def test_latex_formatting_structure(self):
        sample_questions = [
            "What is the integral of x squared?",
            "Evaluate the definite integral from 0 to 1 of sin(x)."
        ]
        latex = generate_latex_questions_from_text(sample_questions, chapter_number="1", topic_name="Integration")
        
        # Adjust assertions to match actual output
        assert "\\begin{enumerate}" in latex
        assert "\\item" in latex
        assert "\\end{enumerate}" in latex

    def test_latex_escape_characters(self):
        special_char_question = "Find value of $f(x) = \\int_0^1 x^2 dx$."
        latex = generate_latex_questions_from_text([special_char_question], chapter_number="2", topic_name="Definite Integrals")
        assert "$" in latex
        assert "\\int" in latex


if __name__ == "__main__":
    unittest.main()
