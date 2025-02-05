import re
import string
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from gender_guesser.detector import Detector
from nltk.corpus import stopwords
from sklearn.metrics import mean_absolute_error, r2_score
from timeseriesmetrics import theil
from wordcloud import WordCloud



class Plot:
    def __init__(self, figsize=(8, 5), palette='coolwarm'):
        """
        Initializes the Plotter with given figsize and color palette.

        Parameters:
        - figsize: Tuple (width, height) for the plot figure size.
        - palette: Color palette to use for the plot.
        """
        self.figsize = figsize
        self.palette = palette

    def barplot(self, data, x_col, y_col, title='Bar Plot', xlabel=None,
                    ylabel=None, rotation=0, show_percentage=True):
            """
            Plots a bar chart and displays either the percentage or absolute 
            values above each bar.
            
            Parameters:
            - data: DataFrame containing the data.
            - x_col: Column name for the X-axis.
            - y_col: Column name for the Y-axis.
            - title: Chart title (default: 'Bar Plot').
            - xlabel: X-axis label (default: None, uses x_col).
            - ylabel: Y-axis label (default: None, uses y_col).
            - rotation: Rotation of the X-axis labels (default: 0).
            - show_percentage: If True, displays percentages; otherwise,
            displays absolute values ​​(default: True).
            """
            
            plt.figure(figsize=self.figsize)
            ax = sns.barplot(x=data[x_col], y=data[y_col], palette=self.palette)

            total = data[y_col].sum()

            for p in ax.patches:
                height = p.get_height()
                label = (f"{(height / total) * 100:.1f}%" if show_percentage 
                        else f"{height:,.0f}")
                ax.annotate(label, (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=12, color='black')

            plt.title(title)
            plt.xlabel(xlabel if xlabel else x_col)
            plt.ylabel(ylabel if ylabel else y_col)
            plt.xticks(rotation=rotation)
            plt.show()
            
    def boxplot(self, data, x_col=None, y_col=None, title='Box Plot',
                xlabel=None, ylabel=None):
        """
        Plots a boxplot using the instance parameters and displays the
        quantile values ​​in the console.
        
        Parameters:
        - data: DataFrame containing the data.
        - x_col: Column name for the X axis (default: None).
        - y_col: Column name for the Y axis.
        - title: Title of the plot (default: 'Box Plot').
        - xlabel: Label for the X axis (default: None, uses x_col if available).
        - ylabel: Label for the Y axis (default: None, uses y_col).
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

        Parameters:
        - data: DataFrame containing the text data.
        - text_column: Column name containing the text for the word cloud.
        - title: Title of the word cloud plot (default: 'Word Cloud').
        - max_words: Maximum number of words to display in the word cloud
          (default: 100).
        - background_color: Background color of the word cloud (default: 'white').
        """
        text = ' '.join(
            data[text_column].dropna().apply(
                lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower()
            )
        )
        words = text.split()
        word_counts = Counter(words)
        df_words = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency'])
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

        Parameters:
        - plotter: Instance of the Plotter class used for visualizations.
        """
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.gender_detector = Detector()
        self.plotter = plotter

    def predict_gender(self, name):
        """
        Predicts the gender based on a given name using the gender_guesser
        library.

        Parameters:
        - name: The name to predict the gender for.
        """
        if isinstance(name, str):
            first_name = name.split()[0]
            return self.gender_detector.get_gender(first_name)
        return 'unknown'

    def most_common_words(self, data, text_column, max_words=20):
        """
        Identifies the most common words in a text column of a DataFrame.

        Parameters:
        - data: DataFrame containing the text data.
        - text_column: Column name containing the text to analyze.
        - max_words: Maximum number of common words to return (default: 20).
        """
        text = ' '.join(
            data[text_column].dropna().apply(
                lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower()
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

        Parameters:
        - data: DataFrame containing the data to plot.
        - text_column: Column name containing the text for analysis.
        - price_column: Column name containing the price values.
        - df_words: DataFrame with word frequencies.
        - max_words: Maximum number of words to include in the plot (default: 10).
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

        Parameters:
        - models: List of models to store residuals for.
        - y_pred: Predicted values from the models.
        """
        for model in models:
            formalism = type(model).__name__
            self.models_label_r.append(formalism)
            self.df_test[formalism] = y_pred
        return self.df_test

    def computeAccuracyModels(self, models, y_pred, y_test):
        """
        Compute and display accuracy metrics (MSE, MAE) for different models.

        Parameters:
        - models: List of models to compute accuracy for.
        - y_pred: Predicted values from the models.
        - y_test: True values for comparison.
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
        print("-" * 30 + "Métricas  para o modelo" + "-" * 30)
        print(df)
        return df

    def train_and_evaluate_with_gridsearch(
        self, model, grid_search, X_train, y_train,
        X_val, y_val, X_test, y_test
    ):
        """
        Trains and evaluates a model using GridSearch.

        Parameters:
        - model: The model to train.
        - grid_search: The grid search object to optimize hyperparameters.
        - X_train: Training feature data.
        - y_train: Training target data.
        - X_val: Validation feature data.
        - y_val: Validation target data.
        - X_test: Test feature data.
        - y_test: Test target data.
        """
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        # Print the best hyperparameters
        print("-" * 30 + "Melhores hiperparâmetros" + "-" * 30)
        for param, value in best_params.items():
            print("{}: {}".format(param, value))
        print("-" * 80)
        
        # Train the model with the best hyperparameters
        best_model = model.__class__(**best_params)
        best_model.fit(X_train, y_train)
        
        # Evaluate on the validation set
        y_val_pred = best_model.predict(X_val)
        r2_val = r2_score(y_val, y_val_pred)
        print("-" * 30 + "Métricas de R² para o modelo" + "-" * 30)
        print("R² para o conjunto de validação: {:.2f}\n".format(r2_val))
        
        # Predict on the test set
        y_test_pred = best_model.predict(X_test)
        
        # Compute residuals and accuracy metrics
        self.ResidualForModels(models=[best_model], y_pred=y_test_pred)
        self.computeAccuracyModels(
            models=[best_model], y_pred=y_test_pred, y_test=y_test
        )
        
        return best_model, grid_search.best_estimator_
