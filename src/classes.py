# libraly imports
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string
import pandas as pd
from gender_guesser.detector import Detector


class Plot:
    def __init__(self, figsize=(8, 5), palette='coolwarm'):
        self.figsize = figsize
        self.palette = palette

    def barplot(self, data, x_col, y_col, title='Bar Plot', xlabel=None, ylabel=None, rotation=0):
        """
        Plots a bar chart using instance parameters.
        """
        plt.figure(figsize=self.figsize)
        sns.barplot(x=data[x_col], y=data[y_col], palette=self.palette)
        plt.title(title)
        plt.xlabel(xlabel if xlabel else x_col)
        plt.ylabel(ylabel if ylabel else y_col)
        plt.xticks(rotation=rotation)
        plt.show()

    def boxplot(self, data, x_col, y_col, title='Box Plot', xlabel=None, ylabel=None):
        """
        Plots a boxplot using instance parameters.
        """
        plt.figure(figsize=self.figsize)
        sns.boxplot(x=data[x_col], y=data[y_col], palette=self.palette)
        plt.title(title)
        plt.xlabel(xlabel if xlabel else x_col)
        plt.ylabel(ylabel if ylabel else y_col)
        plt.tight_layout()
        plt.show()

    def plot_wordcloud(self, data, text_column, title="Word Cloud", max_words=100, background_color="white"):
        """
        Generate and display a word cloud from a specified text column in a DataFrame.
        """
        text = ' '.join(data[text_column].dropna().apply(lambda x: x.translate(str.maketrans('', '', string.punctuation))))
        wordcloud = WordCloud(max_words=max_words, background_color=background_color, 
                              width=1600, height=800, min_font_size=10).generate(text)
        plt.figure(figsize=self.figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, weight='bold')
        plt.show()

class Auxiliaries:
    def __init__(self):
        self.gender_detector = Detector()


    def predict_gender(self, name):
        """
        Predicts the gender based on a given name using the gender_guesser library.

        Parameters:
            name (str): The name to analyze.

        Returns:
            str: Predicted gender ('male', 'female', 'unknown', etc.).
        """
        if isinstance(name, str):
            first_name = name.split()[0]  # Use the first name for better accuracy
            return self.gender_detector.get_gender(first_name)
        return 'unknown'  # Return 'unknown' for non-string values