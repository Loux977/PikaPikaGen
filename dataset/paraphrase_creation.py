import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------- CONFIG ---------------
INPUTH_FILE_PATH = "dataset/pokemon_descriptions.csv"
OUTPUT_FILE_PATH = "dataset/pokemon_descriptions_with_paraphrases.csv"
# --------------------------------------

# Load your CSV
df = pd.read_csv(INPUTH_FILE_PATH)

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)

def generate_paraphrases(
    text,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=3,
    repetition_penalty=7.0,
    diversity_penalty=5.0,
    no_repeat_ngram_size=2,
    max_length=128
):
    try:
        input_ids = tokenizer(
            f'paraphrase: {text}',
            return_tensors="pt", padding="longest",
            max_length=max_length,
            truncation=True
        ).input_ids.to(device)
        
        outputs = model.generate(
            input_ids,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            max_length=max_length,
            diversity_penalty=diversity_penalty
        )
        
        res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Select two distinct paraphrases different from original
        unique_paras = []
        for p in res:
            if p not in unique_paras and p.lower() != text.lower():
                unique_paras.append(p)
            if len(unique_paras) == 2:
                break

        if len(unique_paras) < 2:
            return text, text
        else:
            return unique_paras[0], unique_paras[1]
    except Exception as e:
        print(f"Error paraphrasing: {e}")
        return text, text

# Create new columns
para_1 = []
para_2 = []

for text in tqdm(df['description']):
    p1, p2 = generate_paraphrases(text)
    para_1.append(p1)
    para_2.append(p2)

# Add to DataFrame
df['train_paraphrase'] = para_1
df['test_paraphrase'] = para_2

# Create final df with only needed columns
df_final = df[['name', 'description', 'train_paraphrase', 'test_paraphrase']]

# Save
df_final.to_csv(OUTPUT_FILE_PATH, index=False)

print(" Done!")