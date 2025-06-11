import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SummaryGenerator:
    """Generates summaries using LLM"""
    def __init__(self, model_name = 'facebook/bart-large-cnn'):
        print(f"Loading summarization model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    def generate_summary(self, text, max_length = 150, min_length = 50):
        print("Generating summary...")
        start_time = time.time()
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            end_time = time.time()
            print(f"Summary generation took {end_time - start_time:.2f} seconds")
            return summary
        except Exception as e:
            print(f"Error during summary generation: {str(e)}")
            return "Error generating summary. Please try again with a different text or parameters." 