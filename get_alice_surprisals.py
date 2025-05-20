import pandas as pd

# data from https://doi.org/10.17605/OSF.IO/JHPX8
story_path = "../Literature/MomenianEtAl2024/Language Model/story.csv"
eeg_path = "../Literature/MomenianEtAl2024/Bayesian Analysis/Data.csv"

story_df = pd.read_csv(story_path)
eeg_df = pd.read_csv(eeg_path)
eeg_df = eeg_df[eeg_df.Group == "Monolingual"]


words = list(story_df.word)
og_surp = list(eeg_df.Surprisal_GPT2)
bolbol_surprisal = []
window = 17
ind_to_cut_surprisal = []
start_ind = 0
for i, word in enumerate(words):
    s = surprisal.Bolbol.get_surprisal_at(" ".join(words[start_ind:i + 1]), len(words[start_ind:i + 1]) - 1)
    if abs(s - og_surp[i]) > 0.1:
        start_ind = i
        ind_to_cut_surprisal.append(i)
        s = surprisal.Bolbol.get_surprisal_at(" ".join(words[start_ind:i + 1]), 0)
    bolbol_surprisal.append(s)

ind_to_cut_cp = ind_to_cut_surprisal.copy()
words = list(story_df.word)
fabert_surprisal = []
end_ind = ind_to_cut_cp.pop(0)
start_ind = 0
for i, word in enumerate(words):
    s = surprisal.FaBert.get_surprisal_at(" ".join(words[start_ind:i + 1]), len(words[start_ind:i + 1]) - 1)
    if i == end_ind:
        end_ind = ind_to_cut_cp.pop(0) if ind_to_cut_cp else len(words)
        start_ind = i
        s = surprisal.FaBert.get_surprisal_at(" ".join(words[start_ind:i + 1]), 0)
    fabert_surprisal.append(s)

ind_to_cut_cp = ind_to_cut_surprisal.copy()
words = list(story_df.word)
hoosh_surprisal = []
end_ind = ind_to_cut_cp.pop(0)
start_ind = 0
for i, word in enumerate(words):
    s = surprisal.Hoosh.get_surprisal_at(" ".join(words[start_ind:i + 1]), len(words[start_ind:i + 1]) - 1)
    if i == end_ind:
        end_ind = ind_to_cut_cp.pop(0) if ind_to_cut_cp else len(words)
        start_ind = i
        s = surprisal.Hoosh.get_surprisal_at(" ".join(words[start_ind:i + 1]), 0)
    hoosh_surprisal.append(s)

df = pd.DataFrame({"Bolbol": bolbol_surprisal})
df.insert(0, "orig", eeg_df.Surprisal_GPT2)
df.insert(0, "word", eeg_df.RealWord)

with_bert_df = df.copy()[0:len(fabert_surprisal)]
with_bert_df.insert(len(with_bert_df.columns), "FaBert", fabert_surprisal)
with_bert_df

with_hoosh_df = with_bert_df.copy()[0:len(hoosh_surprisal)]
with_hoosh_df.insert(len(with_hoosh_df.columns), "Hoosh", hoosh_surprisal)
with_hoosh_df

with_hoosh_df.to_csv('AliceWonderlandSurprisals.csv.csv', index=False)