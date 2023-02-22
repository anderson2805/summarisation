import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, BartTokenizer, BartForConditionalGeneration, BartConfig
import csv 

modelT5 = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizerT5 = T5Tokenizer.from_pretrained('t5-base')
modelBart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizerBart = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
device = torch.device('cpu')

def summarizer(text : str, model_name:str = 'BART', min_length:int = 100, max_length:int = 250):
    preprocess_text = text.strip().replace("\n"," ")
    if model_name == 'BART':
        model = modelBart
        tokenizer = tokenizerBart
        prepared_Text = preprocess_text
        tokenized_text = tokenizer.encode(prepared_Text, return_tensors="pt",truncation=True).to(device)
    elif model_name == 'T5-base':
        model = modelT5
        tokenizer = tokenizerT5
        prepared_Text = "summarize: "+preprocess_text
        tokenized_text = tokenizer.encode(prepared_Text, return_tensors="pt", truncation=True, max_length = 2500).to(device)

    
    summary_ids = model.generate(tokenized_text,
                                num_beams=4,
                                no_repeat_ngram_size=2,
                                min_length=min_length,
                                max_length=max_length ,  early_stopping=True )

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output
if __name__ == '__main__':
    doc = """The writer opined that

at present, the forest to the north of Tagore Drive and the Tagore industrial estate was being used by the sf for training
it had become even more important to wildlife in the Upper Thomson area, given that the forest patch to the south of it fringing Yio Chu Kang Road and the Teachers’ Estate — which was called the Tagore-Lentor Forest — was almost completely wiped out for a condominium development
from as early as 2001, Singapore also had the critically endangered songbird, the straw-headed bulbul, in the forest patch north of Tagore Drive
the straw-headed bulbul was listed as critically endangered in the International Union for Conservation of Nature’s (IUCN) Red List of Threatened Species
given the demise of the neighbouring, connected forest around the Yio Chu Kang and Teachers’ Estate fringe, where there were also records of this bulbul species, it was most probable that the bulbuls here would take refuge in the forest north of Tagore Drive through a narrow forest belt to the east of the Tagore industrial estate
there were also records of other nationally threatened bird species, such as the crested serpent eagle and the grey-headed fish eagle, in this patch north of Tagore Drive
the grey-headed fish eagle was also in the IUCN’s Red List as “near-threatened”
The Nature Society (Singapore) also believed that the Sunda pangolin, another critically endangered species globally, would have likewise taken refuge in this patch north of Tagore Drive
the pangolin had been recorded in the forest patch fringing the Teachers’ Estate, which was, as mentioned, already mostly cleared
with the presence of the Raffles’ banded langur, as primatologist Andie Ang noted in the TODAY Online report, the forest north of Tagore Drive was highly important for the well-being of Singapore's biodiversity and
the Nature Society (Singapore) urged the authorities to do an environmental or a biophysical impact assessment, to determine at least some ecologically significant portion of the forested area for conservation before initiating any housing plan.

(The writer is Vice-President of the Nature Society (Singapore).)"""

    print(summarizer(doc))
