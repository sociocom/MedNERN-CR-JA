import mojimoji
import pandas as pd
from rapidfuzz import fuzz, process


class EntityDictionary:

    def __init__(self, path):
        self.df = pd.read_csv(path)

    def get_candidates_list(self):
        return self.df.iloc[:, 0].to_list()

    def get_normalization_list(self):
        return self.df.iloc[:, 2].to_list()

    def get_normalized_term(self, term):
        return self.df[self.df.iloc[:, 0] == term].iloc[:, 2].item()


class DiseaseDict(EntityDictionary):

    def __init__(self):
        super().__init__('dictionaries/disease_dict.csv')


class DrugDict(EntityDictionary):

    def __init__(self):
        super().__init__('dictionaries/drug_dict.csv')


class EntityNormalizer:

    def __init__(self, database: EntityDictionary, matching_method=fuzz.ratio, matching_threshold=0):
        self.database = database
        self.matching_method = matching_method
        self.matching_threshold = matching_threshold
        self.candidates = [mojimoji.han_to_zen(x) for x in self.database.get_candidates_list()]

    def normalize(self, term):
        term = mojimoji.han_to_zen(term)
        preferred_candidate = process.extractOne(term, self.candidates, scorer=self.matching_method)
        score = preferred_candidate[1]

        if score > self.matching_threshold:
            ret = self.database.get_normalized_term(preferred_candidate[0])
            return ('' if pd.isna(ret) else ret), score
        else:
            return '', score

