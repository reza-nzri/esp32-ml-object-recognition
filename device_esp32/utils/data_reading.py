import pandas as pd


class Data:
    """
    Class for reading data into dataframe.
    """

    def __init__(self):
        """
        Initializes empty Dataframe with columns distance (1) and step(2).
        """
        self.df = pd.DataFrame(columns=["distance", "step"])

    def reading(self, distance: float, step: int):
        """
        Reads arguments into DataFrame.

        Args:
            distance: float
            step: int

        Returns:
            None: (None).
        """
        new_row =pd.DataFrame({
            "distance": [distance],
            "step": [step]
        })
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def get_dataframe(self):
        """
        Returns Dataframe with measurements.
        Returns:
            pd.DataFrame: (pd.DataFrame).
        """
        return self.df


