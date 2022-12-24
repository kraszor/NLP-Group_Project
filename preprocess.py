import pandas as pd

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
