from transformers import AutoTokenizer, AutoModelForMaskedLM

splade_name = "naver/splade_v2_max"
tokenizer = AutoTokenizer.from_pretrained(splade_name)
model = AutoModelForMaskedLM.from_pretrained(splade_name)

tokenizer.save_pretrained("models/splade_v2_max")
model.save_pretrained("models/splade_v2_max")