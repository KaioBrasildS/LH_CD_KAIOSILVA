# libraly imports
import matplotlib.pyplot as plt
import seaborn as sns

class Plot:
    @staticmethod
    def barplot(data, x_col, y_col, title='Bar Plot', xlabel=None, ylabel=None, 
                figsize=(8, 5), rotation=0, palette=None):
        """
        Plots a bar chart with optional color palette.

        Parameters:
            data (DataFrame): The dataset containing the values.
            x_col (str): Column name for x-axis.
            y_col (str): Column name for y-axis.
            title (str): Title of the plot.
            xlabel (str): Label for x-axis.
            ylabel (str): Label for y-axis.
            figsize (tuple): Figure size.
            rotation (int): Rotation angle for x-axis labels.
            palette (str or list): Color palette for the bars.
        """
        plt.figure(figsize=figsize)
        sns.barplot(x=data[x_col], y=data[y_col], palette=palette)
        plt.title(title)
        plt.xlabel(xlabel if xlabel else x_col)
        plt.ylabel(ylabel if ylabel else y_col)
        plt.xticks(rotation=rotation)
        plt.show()
    
    @staticmethod
    def boxplot(data, x_col, y_col, title='Box Plot', xlabel=None, ylabel=None, 
                figsize=(10, 6), palette=None):
        """
        Plots a boxplot with optional color palette.

        Parameters:
            data (DataFrame): The dataset containing the values.
            x_col (str): Column name for x-axis.
            y_col (str): Column name for y-axis.
            title (str): Title of the plot.
            xlabel (str): Label for x-axis.
            ylabel (str): Label for y-axis.
            figsize (tuple): Figure size.
            palette (str or list): Color palette for the boxplot.
        """
        plt.figure(figsize=figsize)
        sns.boxplot(x=data[x_col], y=data[y_col], palette=palette)
        plt.title(title)
        plt.xlabel(xlabel if xlabel else x_col)
        plt.ylabel(ylabel if ylabel else y_col)
        plt.tight_layout()
        plt.show()
