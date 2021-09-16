from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    gen = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")
    text_out = gen("I want one pizza with", do_sample=True, min_length=20)
    print("Generated: {}".format(text_out["generated_text"]))


