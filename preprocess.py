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

# Links preprocessing

    @staticmethod
    def __find_links(row: pd.DataFrame) -> list:
        """
        method finds all links mentioned in text
        :param row: row of a dataframe
        :return: list contains the links made up from tuples
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
