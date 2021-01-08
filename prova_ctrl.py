'''
from transformers import CTRLTokenizer, CTRLModel
import torch

#Questo funziona male non capisco il motivo sembra dia output a caso
tokenizer = CTRLTokenizer.from_pretrained('ctrl')
model = CTRLModel.from_pretrained('ctrl')

text = "The highest mountain is "
indexed_tokens = tokenizer.encode(text)
token_tensor = torch.tensor([indexed_tokens])

#important for reproducible results
model.eval();

outputs = model(token_tensor)
predictions = outputs[0]

#Next sub word:
predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
print(predicted_text)
'''

'''
Ok questo funziona come nei test
from transformers import CTRLTokenizer, CTRLLMHeadModel
import torch
tokenizer = CTRLTokenizer.from_pretrained('ctrl')
model = CTRLLMHeadModel.from_pretrained('ctrl')

seq = "Legal the president is"
inputs = tokenizer(seq)
inputs_ids = torch.tensor([inputs["input_ids"]],dtype=torch.long)
# encode context the generation is conditioned on
outputs = model.generate(inputs_ids)
outputs_id = outputs[0]
for i in range(len(outputs_id)):
    print(tokenizer.decode(outputs_id[i].item()))
'''
