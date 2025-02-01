# libraly imports
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string
import pandas as pd
from gender_guesser.detector import Detector
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import  r2_score,mean_absolute_error
from timeseriesmetrics import theil

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
        
    def boxplot(self, data, x_col=None, y_col=None, title='Box Plot', xlabel=None, ylabel=None):
        """
        Plots a boxplot using instance parameters.
        """
        plt.figure(figsize=self.figsize)
        
        if x_col:  # If x_col is provided, use it for the x-axis
            sns.boxplot(x=data[x_col], y=data[y_col], palette=self.palette)
        else:  # If x_col is None or '', plot just the numeric column
            sns.boxplot(y=data[y_col], palette=self.palette)
        
        plt.title(title)
        plt.xlabel(xlabel if xlabel else (x_col if x_col else ''))
        plt.ylabel(ylabel if ylabel else y_col)
        plt.tight_layout()
        plt.show()

    '''def boxplot(self, data, x_col, y_col, title='Box Plot', xlabel=None, ylabel=None):
        """
        Plots a boxplot using instance parameters.
        """
        plt.figure(figsize=self.figsize)
        sns.boxplot(x=data[x_col], y=data[y_col], palette=self.palette)
        plt.title(title)
        plt.xlabel(xlabel if xlabel else x_col)
        plt.ylabel(ylabel if ylabel else y_col)
        plt.tight_layout()
        plt.show()'''


    def plot_wordcloud(self, data, text_column, title="Word Cloud", max_words=100, background_color="white"):
        """
        Generate and display a word cloud from a specified text column in a DataFrame.
        Also returns a DataFrame with the most common words and their frequencies.
        """
        # Preprocess the text
        text = ' '.join(data[text_column].dropna().apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower()))
        
        # Tokenize and count words
        words = text.split()
        word_counts = Counter(words)
        
        # Create DataFrame with word frequencies
        df_words = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
        
        # Generate word cloud
        wordcloud = WordCloud(max_words=max_words, background_color=background_color, 
                            width=1600, height=800, min_font_size=10).generate(text)
        
        # Plot the word cloud
        plt.figure(figsize=self.figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, weight='bold')
        plt.show()
        
        return df_words  # Return DataFrame with word counts


class Auxiliaries:
    def __init__(self,plotter):
        """
        Class constructor that initializes stopwords and other settings.
        """
        # Download stopwords if not already available
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.gender_detector = Detector()
        self.plotter = plotter

    def predict_gender(self, name):
        """
        Predicts the gender based on a given name using the gender_guesser library.
        """
        if isinstance(name, str):
            first_name = name.split()[0]  # Use the first name for better accuracy
            return self.gender_detector.get_gender(first_name)
        return 'unknown'  # Return 'unknown' for non-string values
    
    def most_common_words(self, data, text_column, max_words=20):
        """
        Identifies the most common words in a text column of a DataFrame.
        """
        text = ' '.join(data[text_column].dropna().apply(
            lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower()
        ))
        words = text.split()
        common_words = Counter(words).most_common(max_words)
        return [word for word, _ in common_words]

    def plot_price_distribution(self, data, text_column, price_column, df_words, max_words=10):
            """
            Calculate and plot the distribution of average prices for the most common words 
            in the property name using the word frequency DataFrame.
            """
            # Select the top max_words * 2 most frequent words from df_words
            common_words = df_words.nlargest(max_words * 2, 'Frequency')['Word'].tolist()  # Get extra words for better filtering
            
            # Filter out stopwords, numbers, and short words (less than 3 characters)
            common_words = [
                word for word in common_words 
                if word not in self.stop_words and len(word) > 2 and not re.match(r'^\d+$', word)  # Remove stopwords, numbers, and short words
            ][:max_words]  # Select only the top max_words after filtering

            word_prices = []
            for word in common_words:
                # Filter rows where the text_column contains the word as a whole word (not a substring)
                filtered_data = data[data[text_column].str.contains(rf'\b{word}\b', case=False, na=False, regex=True)]
                
                # Calculate the average price for properties containing this word
                avg_price = filtered_data[price_column].mean()
                word_prices.append((word, avg_price))

            # Create a DataFrame with words and their corresponding average prices
            df_plot = pd.DataFrame(word_prices, columns=['Word', 'Average Price']).dropna()
            
            # Call the plotting function (adjust according to your implementation)
            self.plotter.boxplot(data=df_plot, x_col='Word', y_col='Average Price', title='Box Plot')

            return df_plot  # Return the DataFrame with words and average prices



class Modelling:
    def __init__(self):
        """
        Class constructor that initializes necessary attributes.
        """
        self.models_label_r = []
        self.df_test = pd.DataFrame()
        
        self.models_label = []
        self.models_R2 = []
        self.models_MAE = []
        self.models_theil = []

    def ResidualForModels(self, models, y_pred): 
        """
        Store residual predictions for different models.
        """
        for model in models:
            formalism = type(model).__name__
            self.models_label_r.append(formalism)
            self.df_test[formalism] = y_pred  # Assuming y_pred is correctly formatted

        return self.df_test

    def computeAccuracyModels(self, models, y_pred, y_test): 
        """
        Compute and display accuracy metrics (MSE, MAE) for different models.
        """
        self.models_label
        self.models_R2
        self.models_MAE
        self.models_theil
        
        for model in models:
            formalism = type(model).__name__
            self.models_label.append(formalism)

            # Compute error metrics
            
            self.models_MAE.append(mean_absolute_error(y_true=y_test.values.ravel(), y_pred=y_pred))
            self.models_R2.append(r2_score(y_true=y_test.values.ravel(), y_pred=y_pred))
            self.models_theil.append(theil(y_true=y_test.values.ravel(), y_pred=y_pred))

        # Create DataFrame with metrics
        df = pd.DataFrame({
            'model': self.models_label, 
            'MAE': self.models_MAE,
            'R2': self.models_R2,
            'theil': self.models_theil
        })
        
        print(" --------------------------- Error Metrics for Models --------------------------- ") 
        print(df.sort_values(by='MSE', ascending=True))

        return df  # Returning the DataFrame for further use

