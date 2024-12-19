import torch
from gramformer import Gramformer
import language_tool_python
from transformers import pipeline

# Initialize Gramformer
gf = Gramformer(models=1, use_gpu=torch.cuda.is_available())  # Grammar Error Correction (GEC)

# Initialize LanguageTool
tool = language_tool_python.LanguageTool('en-US')  # Rule-based grammar checker

# Initialize T5 model for contextual rephrasing
qa_pipeline = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base")

# Function to correct grammar using Gramformer
def correct_with_gramformer(input_text):
    try:
        corrections = gf.correct(input_text, max_candidates=1)
        for corrected_sentence in corrections:
            return corrected_sentence
    except Exception as e:
        return f"Gramformer Error: {str(e)}"

# Function to correct grammar using LanguageTool
def correct_with_languagetool(input_text):
    try:
        corrected_text = tool.correct(input_text)
        return corrected_text
    except Exception as e:
        return f"LanguageTool Error: {str(e)}"

# Function to enhance clarity using T5 (optional)
def rephrase_with_t5(input_text):
    try:
        prompt = f"Rephrase this sentence: {input_text}"
        rephrased = qa_pipeline(prompt, max_length=100, num_beams=4, early_stopping=True)
        return rephrased[0]['generated_text']
    except Exception as e:
        return f"T5 Rephrasing Error: {str(e)}"

# Unified grammar correction function
def unified_grammar_check(input_text):
    try:
        # Step 1: Initial correction with Gramformer
        gf_corrected = correct_with_gramformer(input_text)

        # Step 2: Rule-based corrections with LanguageTool
        lt_corrected = correct_with_languagetool(gf_corrected)

        # Step 3: Contextual rephrasing with T5
        final_output = rephrase_with_t5(lt_corrected)

        # Return all stages of correction
        return {
            "original_text": input_text,
            "gf_corrected": gf_corrected,
            "lt_corrected": lt_corrected,
            "final_output": final_output
        }
    except Exception as e:
        return {"error": str(e)}

# Main function for user interaction
if __name__ == "__main__":
    print("Grammar Correction Tool (Integrated with Gramformer, LanguageTool, and T5)")
    while True:
        user_input = input("\nEnter a sentence (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting the tool. Goodbye!")
            break

        # Get grammar corrections
        results = unified_grammar_check(user_input)

        # Display the results
        print("\nOriginal Text:", results.get("original_text", "N/A"))
        print("Gramformer Correction:", results.get("gf_corrected", "N/A"))
        print("LanguageTool Correction:", results.get("lt_corrected", "N/A"))
        print("Final Output (T5 Rephrased):", results.get("final_output", "N/A"))
