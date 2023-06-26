import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy as sp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import typing
import seaborn as sns

st.title("ChemCal Python")

st.write("Analyse your calibration curve data with ChemCal Python. Select manual data entry to see an example. ")
# markdown link format

st.write("## Input Data")

@dataclass
class Experiment:
    df: pd.DataFrame = None
    data: pd.DataFrame = None
    test_replicates: int = None
    cal_line_points: int = None 
    x: str = None # read in the column name from the edited df and use this variable for indexing
    y: str = None # same here

    def user_input(self):
        """function to select to upload a csv or enter data manually. """

        st.write("Enter your data into the field below or upload a csv file.")
        option = st.selectbox(
            'Do you want to enter your data or upload a CSV?',
            ('Enter the data', 'Upload a CSV'))

        st.write("Enter the column names for your predictor and response variables exactly as they appear in your csv file.")


  

        st.write('You selected:', option)

        if option == 'Enter the data':
            self.generate_dataframe()
        elif option == 'Upload a CSV':
            self.x = st.text_input("Enter predictor column name")
            self.y = st.text_input("Enter response column name")
            self.get_uploaded_file()
            


    def generate_dataframe(self):
        self.df = pd.DataFrame(
            [
                {"predictor": 0.2, "response": 0.221 },
                {"predictor": 0.05, "response": 0.057 },
                {"predictor": 0.1, "response": 0.119 },
                {"predictor": 0.8, "response": 0.73 },
                {"predictor": 0.6, "response": 0.599 },
                {"predictor": 0.4, "response": 0.383 }

                
            ]
        )
        self.data = st.data_editor(self.df, num_rows="dynamic")
        self.x = self.data.columns[0]
        self.y = self.data.columns[1]


    def get_uploaded_file(self):
        uploaded_file = st.file_uploader("Choose a file")

        st.write("If your data has uploaded correctly it will appear below.")
        
        if uploaded_file is not None:
            self.df = pd.read_csv(uploaded_file)
            self.data = st.data_editor(self.df, num_rows="dynamic")
            self.pred_data = self.data[str(self.x)]
            

    def get_test_replicates(self):
        """function to get the number of test replicates."""
        self.test_replicates = st.number_input("Enter number of test replicates", min_value=1, value=1, step=1)
        st.write("Number of test replicates: ", self.test_replicates)

    def get_cal_line_points(self):
        """function to get the number of points on the calibration line."""
        self.cal_line_points = len(self.data)
        assert len(self.data) > 1, "Calibration line must have at least 2 points"

    def run(self):
        self.get_test_replicates()
        self.get_cal_line_points()

  

@dataclass
class CalibrationCurve:
    experiment: Experiment
    fitted_model: sm.OLS = None
    slope: float = None
    intercept: float = None
    fitted_values: np.ndarray = None
    r_squared: float = None


    def fit_ols(self):
        X = self.experiment.data[self.experiment.x]
        X = sm.add_constant(X)
        y = self.experiment.data[self.experiment.y]
        self.fitted_model = sm.OLS(y, X).fit()
        self.r_squared = self.fitted_model.rsquared

    def get_params(self):
        self.slope = self.fitted_model.params[1]
        self.intercept = self.fitted_model.params[0]
    
    def get_fitted_values(self):
        self.fitted_values = self.slope * self.experiment.data[self.experiment.x] + self.intercept


    def tabulate_fit_data(self):
        fit_data = pd.DataFrame(
            [
                {"name": "Predictor", "value": self.experiment.x},
                {"name": "Response", "value": self.experiment.y},
                {"name": "Slope", "value": self.slope},
                {"name": "Intercept", "value": self.intercept},
                {"name": "R Squared", "value": self.r_squared},
            ]
        )
        st.write(fit_data)


    def cal_curve_plot(self):
            pred_ols = self.fitted_model.get_prediction()
            iv_l = pred_ols.summary_frame()["obs_ci_lower"]
            iv_u = pred_ols.summary_frame()["obs_ci_upper"]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title("Calibration Curve")
            ax.set_xlabel(self.experiment.x)
            ax.set_ylabel(self.experiment.y)
            ax.plot(self.experiment.data[self.experiment.x], self.experiment.data[self.experiment.y], "o", label="data")
            ax.plot(self.experiment.data[self.experiment.x], self.fitted_model.fittedvalues, "b--.", label="OLS")
            ax.plot(self.experiment.data[self.experiment.x], iv_u, "r--")
            ax.plot(self.experiment.data[self.experiment.x], iv_l, "r--")
    
            ax.text(
                0.95,
                0.05,
                f"y = {self.slope:.3f}x + {self.intercept:.3f}",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=ax.transAxes,
            )
        
            ax.text(
                0.95,
                0.01,
                f"$R^2$ = {self.r_squared:.3f}",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=ax.transAxes,
            )
            ax.legend(loc="best")
            st.pyplot(fig)


    def run(self):
        self.fit_ols()
        self.get_params()
        self.get_fitted_values()
        self.tabulate_fit_data()
        # self.cal_curve_plot()

    def inverse_prediction(self):
        unknown = st.number_input("Enter unknown value")
        pred = (unknown - self.intercept)/self.slope
        return st.write(pred)
    

    

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
    sumsquares: float = None
    sr: float = None
    test_repeats: int = None
    cal_line_points: int = None
      

    def calculate_sse(self):
        self.curve.get_fitted_values()
        return np.sum((self.curve.fitted_values - self.experiment.data[self.experiment.y]) **2)

    def calculate_syx(self):
        return np.sqrt((self.calculate_sse())/(len(self.experiment.data[self.experiment.x])-2))

    def get_t_value(self, alpha):
        return sp.stats.t.ppf(1 - alpha/2, self.df_resid)

    def calculate_uncertainty(self):
        return self.calculate_sxhat() * self.get_t_value(0.05)

    def calculate_sxhat(self):
        return (self.calculate_syx() / self.curve.slope) * np.sqrt( 1/ self.experiment.test_replicates + 1 / self.experiment.cal_line_points) 

    def calculate_sum_square(self):
        return np.sum((self.experiment.data[self.experiment.x] - 
                       self.experiment.data[self.experiment.x].mean())**2)
    
    # y0: mean of replicate observations
    # new input value for obs values
    
    def calculate_hibbert_uncertainty(self, sr, test_repeats, syx, cal_points, ybar, y0):
        return (1/ self.curve.slope) * np.sqrt(((sr**2)/test_repeats) + ((syx**2)/cal_points) 
                                              + 
                                              (((syx**2)*((y0-ybar)**2))/((self.curve.slope**2)*(self.sumsquares))))
    
    
    
    def tabulate_stats(self):
        self.sse = self.calculate_sse()
        self.syx = self.calculate_syx()
        self.df_resid = self.curve.fitted_model.df_resid
        self.t_value = self.get_t_value(0.05)
        self.uncertainty = self.calculate_uncertainty()
        self.r_squared = self.curve.fitted_model.rsquared
        self.sumsquares = self.calculate_sum_square()

        stats_table = pd.DataFrame(
            [
                {"name": "SSE", "value": self.sse},
                {"name": "Syx", "value": self.syx},
                {"name": "Uncertainty", "value": self.uncertainty},
                {"name": "t-value", "value": self.t_value},
                {"name": "df_resid", "value": self.df_resid},
                {"name": "R-squared", "value": self.r_squared},
                {"name": "Sum of squares", "value": self.sumsquares},
            ]
        )
        st.write(stats_table)

@dataclass
class InversePrediction:
    experiment: Experiment
    curve: CalibrationCurve
    stats: Stats
    pred = None
    unknowns: pd.DataFrame = pd.DataFrame(
        [
            {"Observation": 0.490},
            {"Observation": 0.471},
            {"Observation": 0.484},
            {"Observation": 0.473},
            {"Observation": 0.479},
            {"Observation": 0.492},

        ]
    )
    edited_unknowns = None
    hibbert_uncertainty = None
    test_repeats = None
    mean_replicate_observations = None
    average_response_cal_line = None
    sr = None



    def user_input(self):
        self.edited_unknowns = st.data_editor(self.unknowns, num_rows="dynamic")
        self.test_repeats = len(self.edited_unknowns)
        self.mean_replicate_observations = np.mean(self.edited_unknowns["Observation"])
        self.sr = np.std(self.edited_unknowns["Observation"])
        self.average_response_cal_line = np.mean(self.experiment.data[self.experiment.y])


    def calculate_hibbert_uncertainty(self):
        self.hibbert_uncertainty =  (1/self.curve.slope) * np.sqrt(((self.sr**2)/self.test_repeats) + ((self.stats.syx**2)/self.experiment.cal_line_points) + 
                                                                   (((self.stats.syx**2)*((self.mean_replicate_observations-self.average_response_cal_line)**2))/((self.curve.slope**2)*(self.stats.sumsquares))))

    

    def inverse_prediction_hibbert(self):
        self.calculate_hibbert_uncertainty()

        # unknown_h = st.number_input("Enter unknown value", key="unknown_h")
        self.pred = (self.mean_replicate_observations - self.curve.intercept)/self.curve.slope
        return st.write(f"{self.pred: 3f} +/- {self.hibbert_uncertainty * self.stats.get_t_value(0.05): 2f}")
    
    def plot_inverse_prediction(self):
        fig, ax = plt.subplots()
        sns.regplot(x=self.experiment.data[self.experiment.x], y=self.experiment.data[self.experiment.y], ci=95)
        ax.set_title("Calibration curve")
        ax.set_xlabel("Predictor")
        ax.set_ylabel("Response")
        ax.annotate(f"y = {self.curve.slope: 3f}x + {self.curve.intercept: 3f}", xy=(0.1, 0.9), xycoords="axes fraction")
        ax.annotate(f"R-squared = {self.stats.r_squared: 3f}", xy=(0.1, 0.8), xycoords="axes fraction")
        # plot pred value on the plot
        ax.axvline(x=self.pred, color="red", linestyle="--")
        # plot uncertainty on the plot
        # ax.axvline(x=self.pred + self.hibbert_uncertainty * self.stats.get_t_value(0.05), color="black", linestyle="--")
        # ax.axvline(x=self.pred - self.hibbert_uncertainty * self.stats.get_t_value(0.05), color="black", linestyle="--")
        # annotate pred value on plot
        ax.annotate(f"Predicted value = {self.pred: 3f}", xy=(0.09, 0.6), xycoords="data")
        st.pyplot(fig)


def main():
    exp = Experiment()
    exp.user_input()
    exp.run()



    st.header("Calibration Curve") 
    col1, col2 = st.columns(2)
    cal = CalibrationCurve(exp)

    with col1:
        cal.run()
    # cal = CalibrationCurve(exp)
    # cal.run()
    with col2:
        stat = Stats(exp, cal)
        stat.tabulate_stats()
    cal.cal_curve_plot()


    st.header("Inverse Prediction")

    ip = InversePrediction(exp, cal, stat)
    st.write("Enter your observations for the unknown value in the field below and press enter after each value.") 
    st.write("The uncertainty of this prediction is calculated using the equation below:")

    st.latex(r'''
    s_{\hat{x}_0}=\frac{1}{b} \sqrt{\frac{s_r^2}{m}+\frac{s_{y / x}^2}{n}+\frac{s_{y / x}^2\left(y_0-\bar{y}\right)^2}{b^2 \sum_{i=1}^n\left(x_i-\bar{x}\right)^2}})
    ''')

    ip.user_input()
    ip.inverse_prediction_hibbert()
    ip.plot_inverse_prediction()
    


if __name__ == "__main__":
    main()

   












