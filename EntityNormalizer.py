import mojimoji
import pandas as pd
from rapidfuzz import fuzz, process


class EntityDictionary:

    def __init__(self, path, candidate_column, normalization_column):
        if path is None:
            raise ValueError('Path to dictionary file is not specified.')
        if candidate_column is None:
            raise ValueError('Candidate column is not specified.')
        if normalization_column is None:
            raise ValueError('Normalization column is not specified.')

        self.df = pd.read_csv(path)
        self.candidate_column = candidate_column
        self.normalization_column = normalization_column

    def get_candidates_list(self):
        return self.df.iloc[:, self.candidate_column].to_list()

    def get_normalization_list(self):
        return self.df.iloc[:, self.normalization_column].to_list()

    def get_normalized_term(self, term):
        return self.df[self.df.iloc[:, self.candidate_column] == term].iloc[:, self.normalization_column].item()


class DefaultDiseaseDict(EntityDictionary):

    def __init__(self):
        super().__init__('dictionaries/disease_dict.csv', 0, 2)


class DefaultDrugDict(EntityDictionary):

    def __init__(self):
        super().__init__('dictionaries/drug_dict.csv', 0, 2)


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
