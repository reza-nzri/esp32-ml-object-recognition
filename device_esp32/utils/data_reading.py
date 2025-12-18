import pandas as pd
import numpy as np


class Data:
    """
    Class for reading and transforming data.
    """

    def __init__(self, steps_per_revolution=64, step_angle=5.625):
        """
        Initializes empty Dataframe with columns distance and step.
        """
        self.df = pd.DataFrame(columns=["step", "angle", "distance"])
        self.steps_per_revolution = steps_per_revolution
        self.step_angle = step_angle


    def reading(self, distance: float, step: int):
        """
        Reads arguments into DataFrame.

        Args:
            distance: float
            step: int

        Returns:
            None: (None).
        """
        angle = step * self.step_angle
        new_row =pd.DataFrame({
            "step": [step],
            "angle": [angle],
            "distance": [distance]
        })
        self.df = pd.concat([self.df, new_row], ignore_index=True)


    def get_dataframe(self):
        """
        Returns Dataframe with measurements
        :return: pd.DataFrame.
        """
        return self.df

    def normalize_relative(self):
        """
        Relative normal: division by mean
        add column "normalized" to dataframe

        Returns:
            Dataframe with additional column 'normalized'
        """
        mean_distance = self.df['distance'].mean()
        self.df['normalized'] = self.df['distance'] / mean_distance

        return self.df

    def correct_center_and_normalize(self):
        """
        Corrects eccentric position and normalizes

        Returns:
            Dataframe with corrected normal values
        """
        # conversion to cartesian coordinates
        angles_rad = np.radians(self.df['angle'])
        x_coords = self.df['distance'] * np.cos(angles_rad)
        y_coords = self.df['distance'] * np.sin(angles_rad)

        # calculating geometric centre
        centre_x = x_coords.mean()
        centre_y = y_coords.mean()

        # calculate corrected distances
        delta_x = x_coords - centre_x
        delta_y = y_coords - centre_y
        corrected_distances = np.sqrt(delta_x ** 2 + delta_y ** 2)

        # add corrected distance to dataframe
        self.df['distance_corrected'] = corrected_distances

        # apply normal to corrected data
        mean_distance = corrected_distances.mean()
        self.df['normalized'] = corrected_distances / mean_distance

        return self.df


if __name__ == "__main__":
    data = Data(steps_per_revolution=64, step_angle=5.625)

    for step in range(64):
        distance = 20 + np.random.rand() * 5
        data.reading(distance=distance, step=step)

    data.normalize_relative()
    print(data.df.head())