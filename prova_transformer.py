from transformers import pipeline
classifier = pipeline('sentiment-analysis')
print(classifier('We are very happy to show you the 🤗 Transformers library.'))

nlp = pipeline("fill-mask") #fill a sentence
print(nlp(f"HuggingFace is creating a {nlp.tokenizer.mask_token} that the community uses to solve NLP tasks."))
print(nlp(f"A large bus sitting next to {nlp.tokenizer.mask_token}"))

nlp = pipeline("text-generation") #text generation
print(nlp(f"A large bus sitting next to {nlp.tokenizer.mask_token}"))
