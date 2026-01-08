import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class Data:
    """
    Class for reading and transforming data.
    """

    def __init__(self, steps_per_revolution=64, step_angle=5.625):
        """
        Initializes empty Dataframe with columns distance and step.
        """
        self.df = pd.DataFrame()
        self.steps_per_revolution = steps_per_revolution
        self.step_angle = step_angle

    def load_from_csv(self, filepath: str, has_header: bool = True):
        """
        Lädt Daten aus CSV-Datei.
        Erwartet Format: scan_index, step, angle_deg, distance_cm

        Args:
            filepath: Pfad zur CSV-Datei
            has_header: Ob CSV einen Header hat

        Returns:
            self für Method Chaining
        """
        if has_header:
            self.df = pd.read_csv(filepath)
        else:
            self.df = pd.read_csv(
                filepath,
                header=None,
                names=['scan_index', 'step', 'angle_deg', 'distance_cm']
            )

        # Spalten umbenennen für Konsistenz mit bestehenden Methoden
        column_mapping = {
            'scan_index': 'scan_id',
            'angle_deg': 'angle',
            'distance_cm': 'distance'
        }
        self.df.rename(columns=column_mapping, inplace=True)

        return self

    def read_from_csv(self, fname: str, has_header: bool=True):

        """
        Convert CSV-file to a pandas Dataframe

        :param fname: name of CSV-file
        :return: pd.Dataframe
        """

        if has_header:
            self.df = pd.read_csv(fname)

        else:
            self.df = pd.read_csv(
                fname,
                header=None,
                names=["scan_index","step","angle_deg","distance_cm"]
                                  )
        return self

    def get_info(self):

        info={
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": dict(self.df.dtypes),
            "memory_usage": self.df.memory_usage(deep=True).sum()
        }

        if "scan_id" in self.df.columns:
            info["num_scans"] = self.df["scan_id"].nunique()
            info['scans'] = sorted(self.df['scan_id'].unique())
            info['measurements_per_scan'] = dict(self.df.groupby('scan_id').size())

        return info

    def fix_index_column(self, scan_column='scan_id', index_column='step'):
        """
        Korrigiert Index-Spalte: Startet bei 1 für jeden Scan und erhöht in 1er-Schritten.

        Args:
            scan_column: Name der Scan-ID-Spalte
            index_column: Name der Index-Spalte

        Returns:
            self für Method Chaining
        """
        if scan_column in self.df.columns:
            self.df[index_column] = self.df.groupby(scan_column).cumcount() + 1
        else:
            self.df[index_column] = range(1, len(self.df) + 1)

        return self

    def get_dataframe(self):
        """
        Returns Dataframe with measurements
        :return: pd.DataFrame.
        """
        return self.df

    def detect_outliers_iqr(self, column, factor=1.5):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outliers_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        outliers = self.df[outliers_mask]

        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "count": len(outliers),
            "percentage": len(outliers) / len(self.df) * 100,
            "outliers": outliers
        }

    def detect_outliers_zscore(self, column, threshold=3):
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        outliers_mask = z_scores > threshold
        outliers = self.df[outliers_mask]

        return {
            "threshold": threshold,
            "count": len(outliers),
            "percentage": len(outliers) / len(self.df) * 100,
            "outliers": outliers,
            "z_scores": z_scores
        }

    def check_duplicates(self, subset=None):
        """
        Checks for duplicates

        Args:
            subset: List of columns for duplicate checking

        Returns:
            int: Count of duplicates
        """
        return self.df.duplicated(subset=subset).sum()

    def remove_duplicates(self, subset=None, keep='first'):
        """
        Deletes duplicates

        :param subset: list of columns for duplicate checking
        :param keep: "first", "last", or False

        Returns:
            self
        """
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        return self


    def clip_outliers(self, column, method="iqr", **kwargs):

        if method == "iqr":
            outlier_info = self.detect_outliers_iqr(column, **kwargs)
            self.df[column] = self.df[column].clip(
                lower=outlier_info["lower_bound"],
                upper=outlier_info["upper_bound"]
            )
        elif method == "manual":
            lower = kwargs.get("lower", self.df[column].min())
            upper = kwargs.get("upper", self.df[column].max())
            self.df[column] = self.df[column].clip(lower=lower, upper=upper)

        return self

    def normalize_relative(self):
        """
        Relative normal: division by mean
        add column "normalized" to dataframe

        Returns:
            Dataframe with additional column 'normalized'
        """
        mean_distance = self.df["distance"].mean()
        self.df["normalized"] = self.df["distance"] / mean_distance

        return self

    def correct_center_and_normalize(self):
        """
        Corrects eccentric position and normalizes

        Returns:
            Dataframe with corrected normal values
        """


        self.df["angle"] = pd.to_numeric(self.df["angle"], errors="coerce")
        self.df["distance"] = pd.to_numeric(self.df["distance"], errors='coerce')

        self.df = self.df.dropna(subset=["angle", "distance"])
        # conversion to cartesian coordinates
        angles_rad = np.radians(self.df["angle"])
        x_coords = self.df["distance"] * np.cos(angles_rad)
        y_coords = self.df["distance"] * np.sin(angles_rad)

        # calculating geometric centre
        centre_x = x_coords.mean()
        centre_y = y_coords.mean()

        # calculate corrected distances
        delta_x = x_coords - centre_x
        delta_y = y_coords - centre_y
        corrected_distances = np.sqrt(delta_x**2 + delta_y**2)

        # add corrected distance to dataframe
        self.df["distance_corrected"] = corrected_distances

        # apply normal to corrected data
        mean_distance = corrected_distances.mean()

        if mean_distance != 0:
            self.df["normalized"] = corrected_distances / mean_distance
        else:
            self.df["normalized"] = 0
        return self

    def normalize(self, columns, method="minmax", per_scan=False, scan_column="scan_id"):
        """
        Normalizes columns with standard practice ML-methods

        :param columns: List of columns or column name
        :param method: "minmax", "standard" (z-score), or "robust"
        :param per_scan: If True, normalizes each scan group separately
        :param scan_column: Name of scan id column (only relevant if per_scan=True)

        Returns:
            self
        """
        #ensures columns is of type list
        columns = [columns] if isinstance(columns, str) else columns

        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unbekannte Methode: {method}")


        if per_scan and scan_column in self.df.columns:
            #normalize per scan
            for scan_id in self.df[scan_column].unique():
                mask = self.df[scan_column] == scan_id
                self.df.loc[mask, columns] = scaler.fit_transform(
                    self.df.loc[mask, columns]
                )

        else:
            self.df[columns] = scaler.fit_transform(self.df[columns])

        return self

    def add_normalized_columns(self, columns, method='minmax', suffix='_norm',
                               per_scan=False, scan_column='scan_id'):
        """
        Fügt normalisierte Spalten hinzu ohne die Originale zu überschreiben.


        :param columns: List of all columns of column name
            method: 'minmax', 'standard', or 'robust'
            suffix: Suffix for new columns
            per_scan: If True, normalizes for each scan id separate
            scan_column: Name of scan id column

        Returns:
            self
        """
        if isinstance(columns, str):
            columns = [columns]

        # Scaler auswählen
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}")

        new_columns = [f"{col}{suffix}" for col in columns]

        if per_scan and scan_column in self.df.columns:
            for col, new_col in zip(columns, new_columns):
                self.df[new_col] = np.nan
                for scan_id in self.df[scan_column].unique():
                    mask = self.df[scan_column] == scan_id
                    self.df.loc[mask, new_col] = scaler.fit_transform(
                        self.df.loc[mask, [col]]
                    ).flatten()
        else:
            self.df[new_columns] = scaler.fit_transform(self.df[columns])

        return self


    def save_to_csv(self, filepath, index=False):
        """
        Saves DataFrame as CSV.

        :param filepath: destination path
        :param index: True if indices are saved as well
        """
        self.df.to_csv(filepath, index=index)
        return self

    def copy(self):
        """
        Creates a copy of the data object

        Returns:
            New data instance with copied DataFrame
        """
        new_data = Data(
            steps_per_revolution=self.steps_per_revolution,
            step_angle=self.step_angle
        )
        new_data.df = self.df.copy()
        return new_data