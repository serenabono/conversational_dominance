import re
import pandas as pd
from src.llama import Llama


data_path = "../data/transcript_dominance.csv"

class TranscriptDominanceDataset:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.preprocess()

    def preprocess(self):
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.drop(columns=["Unnamed: 0"])
        self.df["text"] = self.df["file_content"].apply(lambda x: self.process_text(x))

    def process_text(self, t1):
        """
        Find all people and moderators and replace them with [P1], [P2], [MOD]
        """

        people_re = re.compile("P0[0-9][0-9]")
        moderator_re = re.compile("M0[0-9][0-9]_S[0-9][0-9]")
        people = list(set(people_re.findall(t1)))
        moderator = list(set(moderator_re.findall(t1)))

        print(people)
        print(moderator)

        assert len(people) == 2, print(f"{len(people)}")
        assert len(moderator) == 1, print(f"{len(moderator)}")

        t2 = f"{t1}"
        t2 = re.sub(re.compile(people[0]), "[P1]", t2)
        t2 = re.sub(re.compile(people[1]), "[P2]", t2)
        t2 = re.sub(re.compile(moderator[0]), "[MOD]", t2)

        return t2

    def get_probs(self, t2, llama_model):
        for val in re.finditer("\[P1\]", t2):
            start_idx = val.start()
            if start_idx!=0:
                context = t2[:start_idx]
                text, probs = llama_model.get_response(
                    prompt=context,
                    max_tokens=4,
                    temperature=0.5,
                    top_p=0.95,
                    top_k=50,
                    logprobs=10,
                    stop=['USER:'],
                    echo=True
                )
                print(text)
                print(probs)
                break



llama = Llama()
df = pd.read_csv(data_path)
df["text"]

