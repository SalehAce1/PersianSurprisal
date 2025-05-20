from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForMaskedLM
import torch
# from https://github.com/HRSadeghi/GPT-Surprisal-Analysis/blob/main/utils.py
import utils
import pandas as pd
import plotly.express as px

class Surprisal:
    tokenizer = None
    model = None

    @classmethod
    def _get_sentence_prob_dist(cls, sentence: str):
        token_ids = cls.tokenizer.encode(sentence, return_tensors='pt')
        softmax = torch.nn.Softmax(dim=-1)
        cls.model.to('cpu');
        cls.model.eval()
        with torch.no_grad():
            output = cls.model(token_ids)
            return softmax(output.logits).squeeze(0)
    
    @classmethod
    def get_surprisal_at(cls, sentence: str, word_ind: int) -> float:
        return 0.0

class Hoosh(Surprisal):
    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained('HooshvareLab/gpt2-fa')
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('HooshvareLab/gpt2-fa')

    @classmethod
    def get_surprisal_at(cls, sentence: str, word_ind: int) -> float:
        words = [word for word in sentence.split(" ") if word]
        p_next = cls._get_sentence_prob_dist(sentence)
        encoded_words = [cls.tokenizer.encode(word if i == 0 else " " + word) for i, word in enumerate(words)]
        target_token = encoded_words[word_ind]
        return float(-torch.log(p_next[word_ind][target_token]).mean())

class Bolbol(Surprisal):
    tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')
    model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained('bolbolzaban/gpt2-persian')

    @classmethod
    def get_surprisal_at(cls, sentence: str, word_ind: int) -> float:
        words = [word for word in sentence.split(" ") if word]
        surprisal, _ = utils.final_eval(cls.model, cls.tokenizer, [words])
        return surprisal[0][word_ind]

# https://huggingface.co/sbunlp/fabert
class FaBert(Surprisal):
    tokenizer = AutoTokenizer.from_pretrained("sbunlp/fabert")
    model = AutoModelForMaskedLM.from_pretrained("sbunlp/fabert")

    @classmethod
    def get_surprisal_at(cls, sentence: str, word_ind: int) -> float:
        words = [word for word in sentence.split(" ") if word][:word_ind + 1]
        sentence = " ".join(words)
        word = words[word_ind]
        sentence = sentence.replace(word, "[MASK]")
        sentence = "[CLS] %s [SEP]"%sentence
        tokenized_text = cls.tokenizer.tokenize(sentence)
        masked_index = tokenized_text.index("[MASK]")
        indexed_tokens = cls.tokenizer.convert_tokens_to_ids(tokenized_text)

        word_ids = cls.tokenizer.convert_tokens_to_ids(cls.tokenizer.tokenize(word))

        tokens_tensor = torch.tensor([indexed_tokens])
        cls.model.eval()
        with torch.no_grad():
            outputs = cls.model(tokens_tensor)
            predictions = outputs[0]

        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)

        return float(-torch.log(probs[word_ids]).mean())

# the below functions are used in the SurprisalOnSafavi notebook

def surprisal_df_from_sentences(sentence_path: str, model: type[Surprisal] = Bolbol):
    with open(sentence_path) as file:
        lines = [line.rstrip() for line in file]
        proc_lines: list[tuple[int, str, str, int, str]] = []
        for i, line in enumerate(lines):
            t = i % 4
            clean_line = line.replace("_","")
            words = line.split(" ")
            noun = words[1]
            verb_ind, verb = next((i, word[:-1]) for i, word in enumerate(words) if "_" in word)
            proc_lines.append((t, clean_line, noun, verb_ind, verb))


    df_dict = {"Distance":[], "Probability":[], "Surprisal":[]}

    for i, line in enumerate(proc_lines):
        t, sent, _, verb_ind, _ = line
        surprisal = model.get_surprisal_at(sent, verb_ind)
        df_dict["Surprisal"].append(surprisal)
        
        if t == 0 or t == 1:
            df_dict["Distance"].append("Short" if t == 0 else "Long")
            df_dict["Probability"].append("High")
        if t == 2 or t == 3:
            df_dict["Distance"].append("Short" if t == 2 else "Long")
            df_dict["Probability"].append("Low")

    return pd.DataFrame(df_dict)

def get_surprisal_comparison_boxplot(df: pd.DataFrame, title:str = "Surprisal at the critical region (verb)"):
    med_vals = df.groupby(["Probability", "Distance"]).median().sort_values(by="Distance", ascending=False)["Surprisal"].reset_index()

    fig1 = px.box(df[df.Probability == "High"], x="Distance", y="Surprisal", color="Probability", boxmode='overlay', labels={"Probability":"Predictability"})
    fig2 = px.box(df[df.Probability == "Low"], x="Distance", y="Surprisal", color="Probability",color_discrete_sequence=["#EF553B"])

    fig3 = px.scatter(x=med_vals[(med_vals.Probability == "High")].Distance, y=med_vals[(med_vals.Probability == "High")].Surprisal,color_discrete_sequence=["#636EFA"])
    fig3.update_traces(mode='lines+markers', line_dash="dash")


    fig4 = px.scatter(x=med_vals[(med_vals.Probability == "Low")].Distance, y=med_vals[(med_vals.Probability == "Low")].Surprisal,color_discrete_sequence=["#EF553B"])
    fig4.update_traces(mode='lines+markers', line_dash="dash")

    fig1.add_traces(list(fig2.select_traces()))

    name = ['High','Low']

    for i in range(len(fig1.data)):
        fig1.data[i]['name'] = name[i]
        fig1.data[i]['showlegend'] = True

    fig1.add_trace(fig3.data[0])
    fig1.add_trace(fig4.data[0])

    fig1.update_layout(width=600, height=400)
    fig1.update_layout(boxgap=0.6)

    fig1.update_layout(title=title)
    fig1.show()