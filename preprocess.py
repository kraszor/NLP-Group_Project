import pandas as pd
import re

MENTAL_HEALTH = './data/Mental-Health-Twitter.csv'
SUSPICIOUS_COMMUNICATION = './data/Suspicious Communication on Social Platforms.csv'
DISASTER = './data/disaster_tweets/train.csv'


class Preprocess:
    """
    Class for preprocessing data, dedicated to dataframes from Tweeter
    """

    def __init__(self, path_to_file: str) -> None:
        """
        creates pandas.DataFrame from .csv file, normalizes column names
        :param path_to_file: path to .csv file with data
        """
        self.df = pd.read_csv(path_to_file)
        if path_to_file == MENTAL_HEALTH:
            self.df.rename(columns={"post_text": "text", "label": "target"},
                           inplace=True)
        elif path_to_file == SUSPICIOUS_COMMUNICATION:
            self.df.rename(columns={"comments": "text", "tagging": "target"},
                           inplace=True)

    def save_to_json(self, path: str) -> None:
        """
        method saving dataframe to .json file
        """
        self.df.to_json(path, orient="records")

# Links preprocessing methods

    @staticmethod
    def __find_links(row: pd.DataFrame) -> list:
        """
        method finds all links mentioned in text
        :param row: row of a dataframe
        :return: list which contains the links made up from tuples
        """
        links = re.findall("(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])",
                           row['text'])
        return links
    
    @staticmethod
    def __remove_links(row: pd.DataFrame) -> str:
        """
        method removes all links mentioned in text
        :param row: row from dataframe
        :return: new clean text without links
        """
        new_text = re.sub("(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])", "",
                          row['text'])
        return new_text

    def preprocess_links(self) -> None:
        """
        method removes links in 'text' column and 
        adds new column 'links' containing this links
        """
        self.df['links'] = self.df.apply(self.__find_links, axis=1)
        self.df['text'] = self.df.apply(self.__remove_links, axis=1)

# References preprocessing methods

    @staticmethod
    def __retweet_classify(row: pd.DataFrame) -> int:
        """
        method classifies if given text is retweet or not
        :return: 1 if is retweet, 0 if not
        """
        return 1 if re.search(r"^RT @\w+", row['text']) else 0

    @staticmethod
    def __retweet_author(row: pd.DataFrame):  # @TODO: complete return type
        """
        methos finds to whom tweet was referring
        :param row: row from given dataframe
        :return: 0 if given tweet weren't referring to no one,
                 or nickname of the account to whom tweet was referring
        """
        if row['is_retweet'] == 1:
            result = re.search(r"^RT @\w+", row['text']).group(0)
            return re.sub(r"^RT @", "", result)
        else:
            return 0

    @staticmethod
    def __find_references(row: pd.DataFrame) -> list:
        """
        method finds all references in text
        :param row: row from dataframe
        :return: list of all references
        """
        references = re.findall(r"@\w+", row['text'])
        all_references = [re.sub(r"^@", "", ref) for ref in references]
        return all_references if row['is_retweet'] == 0 else references[1:]

    @staticmethod
    def __remove_references(row: pd.DataFrame) -> str:
        """
        method removes all the references from text
        (both retweets and normal references)
        :param row: row from dataframe
        :return: list of all references
        """
        new_text = row['text']
        if row['is_retweet']:
            new_text = re.sub(r"^RT @\w+:", "", new_text)
        new_text = re.sub(r"@\w+", "", new_text)
        return new_text

    def preprocess_references(self) -> None:
        """
        method removes all kind of references from text 
        and adds new columns:
        * is_retweet - containing binary information if it was a retweet
        * retweet_of - with the nickname  to whom tweet was referring
        * references - list of all references which occured in text
        """
        self.df['is_retweet'] = self.df.apply(
            lambda row: self.__retweet_classify(row), axis=1)
        self.df['retweet_of'] = self.df.apply(
            lambda row: self.__retweet_author(row), axis=1)
        self.df['references'] = self.df.apply(
            lambda row: self.__find_references(row), axis=1)
        self.df['text'] = self.df.apply(
            lambda row: self.__remove_references(row), axis=1)

# Hashtag preprocessing methods

    @staticmethod
    def __find_hashtags(row: pd.DataFrame) -> list:
        """
        method finds all hashtags in text
        :param row: row from dataframe
        :return: list of all words mentioned as hashtags
        """
        hashtags = re.findall(r"#\w+", row['text'])
        all_hashtags = [re.sub(r"^#", "", hashtag) for hashtag in hashtags]
        return all_hashtags

    @staticmethod
    def __remove_hashtags(row: pd.DataFrame) -> str:
        """
        method removes all hashtags from text
        :param row: row from dataframe
        :return: clean text free of hashtags
        """
        new_text = re.sub(r"#", "", row['text'])
        return new_text

    def preprocess_hashtags(self) -> None:
        """
        method creates new column with list of all hashtags
        which occured in text and removes sign '#' from tweet text
        """
        self.df['hashtags'] = self.df.apply(
            lambda row: self.__find_hashtags(row), axis=1)
        self.df['text'] = self.df.apply(
            lambda row: self.__remove_hashtags(row), axis=1)