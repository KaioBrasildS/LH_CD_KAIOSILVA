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
from sklearn.metrics import r2_score, mean_absolute_error
from timeseriesmetrics import theil
from sklearn.model_selection import GridSearchCV


class Plot:
    def __init__(self, figsize=(8, 5), palette='coolwarm'):
        self.figsize = figsize
        self.palette = palette

    def barplot(self, data, x_col, y_col, title='Bar Plot', xlabel=None,
                ylabel=None, rotation=0):
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

    def boxplot(self, data, x_col=None, y_col=None, title='Box Plot',
                xlabel=None, ylabel=None):
        """
        Plots a boxplot using instance parameters.
        """
        plt.figure(figsize=self.figsize)
        if x_col:
            sns.boxplot(x=data[x_col], y=data[y_col], palette=self.palette)
        else:
            sns.boxplot(y=data[y_col], palette=self.palette)
        plt.title(title)
        plt.xlabel(xlabel if xlabel else (x_col if x_col else ''))
        plt.ylabel(ylabel if ylabel else y_col)
        plt.tight_layout()
        plt.show()

    def plot_wordcloud(self, data, text_column, title='Word Cloud',
                       max_words=100, background_color='white'):
        """
        Generate and display a word cloud from a specified text column in a
        DataFrame. Also returns a DataFrame with the most common words and
        their frequencies.
        """
        text = ' '.join(
            data[text_column].dropna().apply(
                lambda x: x.translate(str.maketrans('', '',
                string.punctuation)).lower()
            )
        )
        words = text.split()
        word_counts = Counter(words)
        df_words = pd.DataFrame(word_counts.items(),
                                columns=['Word', 'Frequency'])
        df_words = df_words.sort_values(by='Frequency', ascending=False)
        wordcloud = WordCloud(
            max_words=max_words,
            background_color=background_color,
            width=1600,
            height=800,
            min_font_size=10
        ).generate(text)
        plt.figure(figsize=self.figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20, weight='bold')
        plt.show()
        return df_words


class Auxiliaries:
    def __init__(self, plotter):
        """
        Class constructor that initializes stopwords and other settings.
        """
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.gender_detector = Detector()
        self.plotter = plotter

    def predict_gender(self, name):
        """
        Predicts the gender based on a given name using the gender_guesser
        library.
        """
        if isinstance(name, str):
            first_name = name.split()[0]
            return self.gender_detector.get_gender(first_name)
        return 'unknown'

    def most_common_words(self, data, text_column, max_words=20):
        """
        Identifies the most common words in a text column of a DataFrame.
        """
        text = ' '.join(
            data[text_column].dropna().apply(
                lambda x: x.translate(str.maketrans('', '',
                string.punctuation)).lower()
            )
        )
        words = text.split()
        common_words = Counter(words).most_common(max_words)
        return [word for word, _ in common_words]

    def plot_price_distribution(self, data, text_column, price_column,
                                df_words, max_words=10):
        """
        Calculate and plot the distribution of average prices for the most
        common words in the property name using the word frequency DataFrame.
        """
        common_words = df_words.nlargest(max_words * 2, 'Frequency')
        common_words = common_words['Word'].tolist()
        common_words = [
            word for word in common_words
            if word not in self.stop_words and len(word) > 2 and
            not re.match(r'^\d+$', word)
        ][:max_words]
        word_prices = []
        for word in common_words:
            filtered_data = data[
                data[text_column].str.contains(
                    rf'\b{word}\b', case=False, na=False, regex=True
                )
            ]
            avg_price = filtered_data[price_column].mean()
            word_prices.append((word, avg_price))
        df_plot = pd.DataFrame(word_prices,
                               columns=['Word', 'Average Price']).dropna()
        self.plotter.boxplot(data=df_plot, x_col='Word',
                             y_col='Average Price', title='Box Plot')
        return df_plot


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
            self.df_test[formalism] = y_pred
        return self.df_test

    def computeAccuracyModels(self, models, y_pred, y_test):
        """
        Compute and display accuracy metrics (MSE, MAE) for different models.
        """
        for model in models:
            formalism = type(model).__name__
            self.models_label.append(formalism)
            self.models_MAE.append(
                mean_absolute_error(y_true=y_test.values.ravel(),
                                    y_pred=y_pred)
            )
            self.models_R2.append(
                r2_score(y_true=y_test.values.ravel(), y_pred=y_pred)
            )
            self.models_theil.append(
                theil(y_true=y_test.values.ravel(), y_pred=y_pred)
            )
        df = pd.DataFrame({
            'model': self.models_label,
            'MAE': self.models_MAE,
            'R2': self.models_R2,
            'theil': self.models_theil
        })
        print("-" * 30 + "Métricas de erro para os modelos" + "-" * 30)
        
        return df


    def train_and_evaluate_with_gridsearch(self, model, grid_search, X_train, y_train, X_val, y_val, X_test):
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        # Treinar o modelo com os melhores hiperparâmetros
        best_model = model.__class__(**best_params)
        best_model.fit(X_train, y_train)
        
        # Avaliação no conjunto de validação
        y_val_pred = best_model.predict(X_val)
        r2_val = r2_score(y_val, y_val_pred)
        print("-" * 30 + "Métricas de R2 para o modelo" + "-" * 30)
        print(f"R² para o conjunto de validação: {r2_val:.4f}")

        # Previsão no conjunto de teste
        y_test_pred = best_model.predict(X_test)
        
        # Computar resíduos e métricas de acurácia
        self.Modelling.ResidualForModels(models=[grid_search.best_estimator_], y_pred=y_test_pred)
        self.Modelling.computeAccuracyModels(models=[grid_search.best_estimator_], y_pred=y_test_pred)

        return best_model, grid_search.best_estimator_
