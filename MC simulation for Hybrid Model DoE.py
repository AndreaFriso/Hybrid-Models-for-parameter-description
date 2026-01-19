import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session



# Parameters

# Mechanistic friction parameters
viscosity = 0.1        # [Pa s]
r_barrel = 0.00535     # [m]

# Injection dynamics parameters
m = 0.012              # [kg]
k = 1200               # [N/m]
l0 = 0.045             # [m]
x_end = 0.035          # [m]

# Hydrodynamic parameters (from Rathore)
mu_oil = 0.1           # [Pa s]
d_oil = 1.5e-9         # [m]
mu_F = 0.003           # [Pa s]
Ln = 0.02              # [m]
rn = 0.00015           # [m]
rb = r_barrel
l_stopper = 0.007      # [m]



# File paths
file_training1 = r"C:\Users\Andrea\OneDrive - University College London (1)\Desktop\Ph.D\Data\Hypack04_9281826\GSK_DATA_Hypack_04_928_192mm_test1.csv"
file_test_pred = r"C:\Users\Andrea\OneDrive - University College London (1)\Desktop\Ph.D\Data\Hypack04_9281826\GSK_DATA_Hypack_04_928_120mm_test1.csv"



# Load data

data_train1 = pd.read_csv(file_training1)
data_test_pred = pd.read_csv(file_test_pred)

friction_training1 = data_train1['Friction [N]'].values
friction_test_pred = data_test_pred['Friction [N]'].values

travel_training1 = data_train1['Travel [mm]'].values
travel_test_pred = data_test_pred['Travel [mm]'].values

speed_training1 = data_train1['Target Speed [mm/min]'].values
speed_test_pred = data_test_pred['Target Speed [mm/min]'].values

max_interference_training1 = data_train1['Max interference [μm]'].values
max_interference_test_pred = data_test_pred['Max interference [μm]'].values


# Mechanistic model

def friction_mechanistic(speed, theta):
    return (2 * math.pi * viscosity * r_barrel * theta) * speed


def calibration_function(theta, speed, friction_measured):
    friction_predicted = friction_mechanistic(speed, theta)
    return (friction_predicted - friction_measured) ** 2



# Calibration of theta
initial_guess = 20610

optimal_parameter_cal_training1 = np.zeros(len(friction_training1))
optimal_parameter_cal_testing = np.zeros(len(friction_test_pred))

for i in range(len(friction_training1)):
    res = minimize(calibration_function, initial_guess, args=(speed_training1[i], friction_training1[i]))
    optimal_parameter_cal_training1[i] = res.x[0]

for i in range(len(friction_test_pred)):
    res = minimize(calibration_function, initial_guess, args=(speed_test_pred[i], friction_test_pred[i]))
    optimal_parameter_cal_testing[i] = res.x[0]



# ANN dataset (travel + interference)

Y_training_ANN = optimal_parameter_cal_training1
Y_test_ANN = optimal_parameter_cal_testing

X_train = np.column_stack((travel_training1, max_interference_training1))
X_test  = np.column_stack((travel_test_pred, max_interference_test_pred))

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled  = scaler_x.transform(X_test)

y_train_scaled = scaler_y.fit_transform(Y_training_ANN.reshape(-1, 1)).flatten()
y_test_scaled  = scaler_y.transform(Y_test_ANN.reshape(-1, 1)).flatten()



# Build random ANN (stile Rathore)

def build_ann_random(input_dim):
    n1 = np.random.choice([32, 64, 128])
    n2 = np.random.choice([32, 64, 128])
    lr = 10 ** np.random.uniform(-4, -2.5)

    model = Sequential([
        Dense(n1, activation='relu', input_shape=(input_dim,)),
        Dense(n2, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(lr), loss='mse')
    return model



# MC over ANN hyperparameters → theta
def mc_hyperparam_theta_prediction(X_train_scaled, y_train_scaled, X_test_scaled, n_mc=500, epochs=100):
    theta_preds = []

    for i in range(n_mc):
        print(f"ANN MC {i+1}/{n_mc}")

        clear_session()
        model = build_ann_random(X_train_scaled.shape[1])
        model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=32, verbose=0)

        y_pred_scaled = model.predict(X_test_scaled).flatten()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        theta_preds.append(y_pred)

    return np.array(theta_preds)


n_mc = 500   # debug: 5 | for the paper at least 100
epochs_mc = 100

theta_mc_samples = mc_hyperparam_theta_prediction(
    X_train_scaled,
    y_train_scaled,
    X_test_scaled,
    n_mc=n_mc,
    epochs=epochs_mc
)



# Uncertainty propagation from the theta to the friction (MC simulation)

def compute_friction_from_theta(theta_samples, speed):
    n_mc, N = theta_samples.shape
    friction_samples = np.zeros((n_mc, N))

    for i in range(n_mc):
        for j in range(N):
            friction_samples[i, j] = friction_mechanistic(speed[j], theta_samples[i, j])

    return friction_samples


friction_mc_samples = compute_friction_from_theta(theta_mc_samples, speed_test_pred)


# Injection time model (ODE)

def injection_time_model(state, t, m, k, l0, friction_fun):
    x, v = state

    Kf = (2 * np.pi * mu_oil * rb * l_stopper) / d_oil
    Kh = (8 * np.pi * mu_F * Ln * rb**4) / rn**4
    K_hydro = Kh

    dxdt = v
    dvdt = (k * (l0 - x) - friction_fun(x) - K_hydro * v) / m

    return [dxdt, dvdt]


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


def compute_injection_time(travel_mm, friction_profile, label=""):
    friction_fun = make_friction_interpolator(travel_mm, friction_profile)

    t = np.linspace(0, 80.0, 6000)
    state0 = [0.0, 0.0]

    sol = odeint(injection_time_model, state0, t, args=(m, k, l0, friction_fun))

    x = sol[:, 0]
    idx = np.where(x >= x_end)[0]

    if len(idx) == 0:
        print(f"Injection not completed: {label}")
        return np.nan, t, x

    return t[idx[0]], t, x



# Uncertainty propagation from the friction to the injection time (MC simulation #2)

def mc_injection_time_from_friction(travel_mm, friction_mc_samples):
    t_inj_samples = []

    n_mc = friction_mc_samples.shape[0]

    for i in range(n_mc):
        print(f"Injection MC {i+1}/{n_mc}")

        friction_profile = friction_mc_samples[i, :]
        t_inj, _, _ = compute_injection_time(travel_mm, friction_profile, label=f"MC {i}")

        if not np.isnan(t_inj):
            t_inj_samples.append(t_inj)

    return np.array(t_inj_samples)


t_inj_mc_samples = mc_injection_time_from_friction(travel_test_pred, friction_mc_samples)



# Final statistics

t_inj_mean = np.mean(t_inj_mc_samples)
t_inj_std  = np.std(t_inj_mc_samples)
t_inj_min  = np.min(t_inj_mc_samples)
t_inj_max  = np.max(t_inj_mc_samples)

print("\n========== FINAL RESULTS ==========")
print(f"Mean injection time = {t_inj_mean:.4f} s")
print(f"Std  injection time = {t_inj_std:.4f} s")
print(f"Min  injection time = {t_inj_min:.4f} s")
print(f"Max  injection time = {t_inj_max:.4f} s")



# Plot distribution: distributions of the ijection times obtained using the hybrid models and the MC simulation 

plt.figure(figsize=(7,5))
plt.hist(t_inj_mc_samples, bins=15, alpha=0.75, edgecolor='k')
plt.axvline(t_inj_mean, color='red', linestyle='--', linewidth=2, label=f"Mean = {t_inj_mean:.3f} s")
plt.xlabel("Injection time [s]")
plt.ylabel("Frequency")
plt.title("Injection time distribution (ANN hyperparameter MC)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# Plot plunger dynamics envelope

x_trajs = []

for i in range(friction_mc_samples.shape[0]):
    _, tgrid, x = compute_injection_time(travel_test_pred, friction_mc_samples[i, :], label=f"MC {i}")
    x_trajs.append(x)

x_trajs = np.array(x_trajs)
x_min = np.min(x_trajs, axis=0)
x_max = np.max(x_trajs, axis=0)
x_mean = np.mean(x_trajs, axis=0)

plt.figure(figsize=(8,5))
plt.plot(tgrid, x_mean*1000, label="Mean trajectory", linewidth=2)
plt.fill_between(tgrid, x_min*1000, x_max*1000, alpha=0.3, label="MC envelope")
plt.axhline(x_end*1000, color="red", linestyle="--", label="End of injection")
plt.xlabel("Time [s]")
plt.ylabel("Plunger position [mm]")
plt.title("Plunger dynamics – ANN epistemic uncertainty propagation")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()


