import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

from scipy.integrate import odeint
from scipy.interpolate import interp1d


# Parameters

# Mechanistic friction parameters
viscosity = 0.1        # [Pa s]
r_barrel = 0.00735     # [m]

# Injection dynamics parameters
m = 0.012              # [kg]
k = 1200               # [N/m]
l0 = 0.045             # [m]
x_end = 0.035          # [m]


# File paths
file_training1 = r"C:\Users\Andrea\OneDrive - University College London (1)\Desktop\Ph.D\Data\Hypack04_9281826\GSK_DATA_Hypack_04_928_192mm_test1.csv"
file_test_pred = r"C:\Users\Andrea\OneDrive - University College London (1)\Desktop\Ph.D\Data\Hypack04_9281826\GSK_DATA_Hypack_04_928_120mm_test1.csv"


# Load data
data_train1 = pd.read_csv(file_training1)
data_test_pred = pd.read_csv(file_test_pred)

# Friction
friction_training1 = data_train1['Friction [N]'].values
friction_test_pred = data_test_pred['Friction [N]'].values

# Travel
travel_training1 = data_train1['Travel [mm]'].values
travel_test_pred = data_test_pred['Travel [mm]'].values

# Speed
speed_training1 = data_train1['Target Speed [mm/min]'].values
speed_test_pred = data_test_pred['Target Speed [mm/min]'].values


# Mechanistic model for the friction force
def friction_mechanistic(speed, theta):
    """
    Mechanistic friction model
    """
    friction_force = (2 * math.pi * viscosity * (r_barrel/2) * theta) * speed
    return friction_force


def calibration_function(theta, speed, friction_measured):
    """
    Objective function for calibration
    """
    friction_predicted = friction_mechanistic(speed, theta)
    residual = (friction_predicted - friction_measured) ** 2
    return residual


# Calibration of theta

initial_guess = 20610 # this initial guess is obtained from experiments

optimal_parameter_cal_training1 = np.zeros(len(friction_training1))
optimal_parameter_cal_testing = np.zeros(len(friction_test_pred))

for i in range(len(friction_training1)):
    calibration_train = minimize(
        calibration_function,
        initial_guess,
        args=(speed_training1[i], friction_training1[i])
    )
    optimal_parameter_cal_training1[i] = calibration_train.x[0]

for i in range(len(friction_test_pred)):
    calibration_test = minimize(
        calibration_function,
        initial_guess,
        args=(speed_test_pred[i], friction_test_pred[i])
    )
    optimal_parameter_cal_testing[i] = calibration_test.x[0]

# Plot calibrated theta (training theta VS testing theta)

plt.figure()
plt.scatter(travel_training1, optimal_parameter_cal_training1, label='Training data')
plt.scatter(travel_test_pred, optimal_parameter_cal_testing, label='Testing data')
plt.xlabel('Travel [mm]')
plt.ylabel('Calibrated parameter θ')
plt.legend()
plt.show()

# ANN dataset
# Here I defined the training and testing datasets usid to train and validate the data-driven model.

Y_training_ANN = optimal_parameter_cal_training1
Y_test_ANN = optimal_parameter_cal_testing

X_train = travel_training1.reshape(-1, 1)
X_test = travel_test_pred.reshape(-1, 1)

# Scaling
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

y_train_scaled = scaler_y.fit_transform(Y_training_ANN.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(Y_test_ANN.reshape(-1, 1)).flatten()

# Build deterministic ANN
# Initialization of the ANN

# ANN structure
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# ANN training procedure
history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_scaled),
    verbose=1
)

# ANN performance evaluation
loss, mae = model.evaluate(X_test_scaled, y_test_scaled)
print(f'Validation Loss: {loss:.4f}, Validation MAE: {mae:.4f}')

# Now that the ANN is trained we can use it to predict the values of theta 
Y_pred_scaled = model.predict(X_test_scaled).flatten()
Y_pred_rescaled = scaler_y.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()

# Hybrid model prediction
# Here we used the predictions obtained with the ANN in the mechanistic model (Rathore model)
friction_predicted_hybrid = np.zeros(len(friction_test_pred))

for i in range(len(friction_test_pred)):
    friction_predicted_hybrid[i] = friction_mechanistic(
        speed_test_pred[i],
        Y_pred_rescaled[i]
    )

# Plot final comparison: comparison of the profiles of the measured friction & the friction obtained using the hybrid model
meas_std = [0.400] * len(friction_test_pred)

plt.figure(figsize=(8, 5))
plt.plot(travel_test_pred, friction_test_pred, label='Measured friction')
plt.fill_between(
    travel_test_pred,
    friction_test_pred - meas_std,
    friction_test_pred + meas_std,
    alpha=0.2,
    label="Measurement variance"
)
plt.plot(travel_test_pred, friction_predicted_hybrid, label='Hybrid prediction')
plt.axvline(x=5, color="red", linestyle="dotted")
plt.axvline(x=30, color="red", linestyle="dotted")
plt.xlabel('Travel [mm]', fontsize=14)
plt.ylabel('Friction [N]', fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()



# Hydrodynamic parameters 
mu_oil = 0.1           # [Pa s]
d_oil = 1.5e-9         # [m]
mu_F = 0.003           # [Pa s]
Ln = 0.02              # [m]
rn = 0.00015           # [m]
rb = r_barrel
l_stopper = 0.007      # [m]


# Injection dynamics ODE (the momentum balance is used to evaluate the injection time)

def injection_time_model(state, t, m, k, l0, friction_fun):
    x, v = state

    Kf = (2 * np.pi * mu_oil * rb * l_stopper) / d_oil
    Kh = (8 * np.pi * mu_F * Ln * rb**4) / rn**4
    K_hydro = Kf + Kh

    dxdt = v
    dvdt = (k * (l0 - x) - friction_fun(x) - K_hydro * v) / m
    return [dxdt, dvdt]



# Friction interpolator (IDENTICAL logic to Rathore_ann4)

def make_friction_interpolator(travel_mm, friction_profile):
    travel_m = travel_mm / 1000

    travel_m_unique, idx = np.unique(travel_m, return_index=True)
    friction_unique = friction_profile[idx]

    if len(travel_m_unique) < 2:
        return lambda x: friction_unique[0]

    return interp1d(
        travel_m_unique,
        friction_unique,
        bounds_error=False,
        fill_value=(friction_unique[0], friction_unique[-1])
    )



# Injection time computation 

def compute_injection_time(travel_mm, friction_profile, label=""):
    friction_fun = make_friction_interpolator(travel_mm, friction_profile)

    t = np.linspace(0, 80.0, 6000)
    state0 = [0.0, 0.0]

    sol = odeint(
        injection_time_model,
        state0,
        t,
        args=(m, k, l0, friction_fun)
    )

    x = sol[:, 0]
    idx = np.where(x >= x_end)[0]

    if len(idx) == 0:
        print(f"⚠ Injection not completed: {label}")
        return np.nan, t, x

    return t[idx[0]], t, x



# Compute injection time using the hybrid friction obtained in the first part of the code

t_hybrid, tgrid, x_hybrid = compute_injection_time(
    travel_test_pred,
    friction_predicted_hybrid,
    label="Hybrid"
)

print(f"\nInjection time (Hybrid model): {t_hybrid:.4f} s")



# Also mechanistic-only for comparison

# Build mechanistic friction profile using calibrated theta from test
friction_mech_test = np.zeros(len(travel_test_pred))
for i in range(len(travel_test_pred)):
    friction_mech_test[i] = friction_mechanistic(
        speed_test_pred[i],
        optimal_parameter_cal_testing[i]
    )

t_mech, _, x_mech = compute_injection_time(
    travel_test_pred,
    friction_mech_test,
    label="Mechanistic"
)

print(f"Injection time (Mechanistic model): {t_mech:.4f} s")



# Plot plunger dynamics
plt.figure(figsize=(8,5))
plt.plot(tgrid, x_mech*1000, label="Mechanistic", linewidth=2)
plt.plot(tgrid, x_hybrid*1000, label="Hybrid", linewidth=2)
plt.axhline(x_end*1000, color="red", linestyle="--", label="End of injection")
plt.xlabel("Time [s]")
plt.ylabel("Plunger position [mm]")
plt.title("Injection dynamics – mechanistic vs hybrid")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
