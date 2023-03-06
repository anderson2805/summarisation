import torch
import json
from pydantic import BaseModel
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, T5Config,
    BartTokenizer, BartForConditionalGeneration, BartConfig
)
from fastapi import FastAPI

app = FastAPI()

models = {
    'BART': (BartForConditionalGeneration, BartTokenizer, 'facebook/bart-large-cnn', 'cpu'),
    'T5-base': (T5ForConditionalGeneration, T5Tokenizer, 't5-base', 'cpu'),
    'flan-T5-large': (T5ForConditionalGeneration, T5Tokenizer, 'google/flan-t5-large', 'cuda')
}

NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 2

class InputData(BaseModel):
    text: str
    model_name: str = 'flan-T5-large'
    min_length: int = 100
    max_length: int = 250
    role: str = 'management'


def summarizer(text: str, model_name: str = 'flan-T5-large', min_length: int = 100, max_length: int = 250, role = 'management'):
    preprocess_text = text.strip().replace("\n"," ")
    ModelClass, TokenizerClass, pretrained_model_name, device = models[model_name]
    with torch.no_grad():
        model = ModelClass.from_pretrained(pretrained_model_name).to(device)
        tokenizer = TokenizerClass.from_pretrained(pretrained_model_name)

        if model_name == 'T5-base' or model_name == 'flan-T5-large':
            prepared_text = f"summarize for {role}: {preprocess_text}"
        else:
            prepared_text = preprocess_text

        tokenized_text = tokenizer.encode(
            prepared_text, 
            return_tensors="pt", 
            truncation=True
        ).to(device)

        summary_ids = model.generate(
            tokenized_text,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            min_length=min_length,
            max_length=max_length,
            early_stopping=True
        )

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return output

@app.post('/summarize')
def summarize(input_data: InputData):
    text = input_data.text
    model_name = input_data.model_name
    min_length = input_data.min_length
    max_length = input_data.max_length
    role = input_data.role
    summary = summarizer(text, model_name, min_length, max_length, role)
    return {'summary': summary}

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
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
