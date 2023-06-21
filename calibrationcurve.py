import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import typing
from abc import ABC, abstractmethod

st.title("Calibration Curve")

@dataclass
class Experiment:
    data: pd.DataFrame = None
    test_replicates: int = None
    cal_line_points: int = None 
    x: str = None # read in the column name from the edited df and use this variable for indexing
    y: str = None # same here


    # create df that can be edited to allow for user input
    # todo: add option to upload csv

    def generate_dataframe(self):
        df = pd.DataFrame(
            [
                {"predictor": 1, "response": 1 },
                
            ]
        )
        self.data = st.data_editor(df, num_rows="dynamic")
        self.x = self.data.columns[0]
        self.y = self.data.columns[1]

    # user input functions to accept reps and cal line points(maybe just read from data/non na rows)

    def get_test_replicates(self):
        self.test_replicates = st.number_input("Enter number of test replicates", min_value=1, value=1, step=1)
        st.write("Number of test replicates: ", self.test_replicates)

    def get_cal_line_points(self):
        # self.cal_line_points = st.number_input("Enter number of calibration line points", min_value=1, value=1, step=1)
        # st.write("Number of calibration line points: ", self.cal_line_points)
        self.cal_line_points = len(self.data)
        assert len(self.data) > 1, "Calibration line must have at least 2 points"

    def run(self):
        self.generate_dataframe()
        self.get_test_replicates()
        self.get_cal_line_points()

  

exp = Experiment()

exp.run()
exp.tabulate_fit_data()

@dataclass
class CalibrationCurve:
    experiment: Experiment
    fitted_model: sm.OLS = None
    slope: float = None
    intercept: float = None
    fitted_values: np.ndarray = None


    def fit_ols(self):
        X = self.experiment.data[self.experiment.x]
        X = sm.add_constant(X)
        y = self.experiment.data[self.experiment.y]
        self.fitted_model = sm.OLS(y, X).fit()

    def get_params(self):
        self.slope = self.fitted_model.params[1]
        self.intercept = self.fitted_model.params[0]
    
    def get_fitted_values(self):
        self.fitted_values = self.slope * self.experiment.data[self.experiment.x] + self.intercept

    def run(self):
        self.fit_ols()
        self.get_params()
        self.get_fitted_values()

    def tabulate_fit_data(self):
        fit_data = pd.DataFrame(
            [
                {"name": "Test Replicates", "value": self.test_replicates},
                {"name": "Calibration Line Points", "value": self.cal_line_points},
                {"name": "Predictor", "value": self.x},
                {"name": "Response", "value": self.y},
                {"name": "Slope", "value": self.slope},
                {"name": "Intercept", "value": self.intercept},
                {"name": "R Squared", "value": self.fitted_model.r_squared},
            ]
        )
        st.write(fit_data)

cal = CalibrationCurve(exp)
cal.run()
cal.tabulate_fit_data()


@dataclass 
class Stats:
    experiment: Experiment
    curve: CalibrationCurve
    sse: float = None
    syx: float = None
    uncertainty: float = None
    t_value: float = None
    df_resid: int = None
    r_squared: float = None

    def inverse_prediction(self, unknown):
        unknown = st.number_input("Enter unknown value", min_value=0.0, value=0.0, step=0.1)
        if len(unknown) > 1:
            y = np.mean(unknown)
        else:
            y = unknown[0]
        
        return (y - self.intercept)/self.slope

    def calculate_sse(self):
        self.curve.get_fitted_values()
        return np.sum((self.curve.fitted_values - self.experiment.data[self.experiment.y]) **2)

    def calculate_syx(self):
        return np.sqrt((self.calculate_sse())/(len(self.experiment.data[self.experiment.x])-2))

    # def get_t_value(self,alpha=0.05):
        # return sp.stats.t.ppf(1 - alpha/2, self.df_resid)

    # def calculate_uncertainty(self):
        # return self.calculate_sxhat() * self.get_t_value(0.05)

    def calculate_sxhat(self):
        return (self.calculate_syx() / self.curve.slope) * np.sqrt( 1/ self.experiment.test_replicates + 1 / self.experiment.cal_line_points) 

    def tabulate_stats(self):
        self.sse = self.calculate_sse()
        self.syx = self.calculate_syx()
        # self.uncertainty = self.calculate_uncertainty()
        # self.t_value = self.get_t_value(0.05)
        self.df_resid = self.curve.fitted_model.df_resid
        self.r_squared = self.curve.fitted_model.rsquared

        stats_table = pd.DataFrame(
            [
                {"name": "SSE", "value": self.sse},
                {"name": "Syx", "value": self.syx},
                {"name": "Uncertainty", "value": self.uncertainty},
                # {"name": "t-value", "value": self.t_value},
                {"name": "df_resid", "value": self.df_resid},
                {"name": "R-squared", "value": self.r_squared}
            ]
        )
        st.write(stats_table)

stats = Stats(exp, cal)
stats.tabulate_stats()






# stats = pd.DataFrame(
#     [
#         {"name": "R-squared", "value": model.rsquared},
#         {"name": "Slope", "value": model.params[1]},
#         {"name": "Intercept", "value": model.params[0]}

#     ]
# )

# def calculate_fitted_values(self):
        
#         self.fitted_values = self.slope * self.x + self.intercept

   












