# Iran Regime Fall Probability Model

This repository contains a synthetic probabilistic model that simulates the likelihood of a regime fall in Iran under different scenarios. The model integrates internal dynamics within Iran, external geopolitical actions, and a logistic hazard function to estimate probabilities of regime instability and fall over a simulated period.

The model is implemented as an interactive [Streamlit](https://streamlit.io/) app that allows users to explore the impact of various internal and external parameters on regime stability.

---

## Features

- **Internal State Evolution**: Simulates the evolution of core state variables such as economic stress, protest pressure, repression, elite cohesion, and information blackout dynamics over time.
- **External Actor Choices**: Models geopolitical interactions using a **quantal response** framework, where external actors (e.g., US, Israel, Russia, GCC states) choose actions based on scenario parameters.
- **Logistic Hazard Estimation**: Estimates the probability of regime fall in each month and aggregates survival probabilities over time.
- **Monte Carlo Simulations**: Computes probabilities and risk estimates using thousands of independent simulation trajectories to capture uncertainty.
- **Interactive Dashboard**: Runs as a Streamlit app, offering sliders and inputs for scenario configurations and generating interactive visualizations.

---

## Model Design Overview

### 1. Internal State Variables

The **internal state** of the Iran regime includes five core variables, each scaled to [-1, +1]:

- **Economic Stress (econ)**: Higher values represent worse economic conditions.
- **Protest Pressure (protest)**: Higher values indicate stronger societal unrest.
- **Repression (repression)**: Higher values indicate greater use of coercive repression.
- **Elite Cohesion (elite)**: Higher values reflect stronger political unity among regime elites.
- **Information Blackout (blackout)**: Higher values indicate heavy information suppression and control.

### 2. External Actors and Actions

The model incorporates the influence of external actors through a **quantal response** (logit choice) framework. Each actor (e.g., the US, Israel, GCC, etc.) chooses from a set of discrete actions that are mapped to continuous intensities affecting Iran’s stability.

### 3. Monthly Hazard Calculation

The probability of regime fall in a given month is estimated using a **logit function**:

```math
h_t = sigmoid(η_t)
```

The **log-odds (η_t)** combines contributions from internal state variables, external actions, and their interactions.

### 4. Stochastic Dynamics

The internal state evolves stochastically over time, incorporating:
- **Drift**: Systematic monthly trends.
- **Coupling Effects**: Dependencies between variables (e.g., economic stress affects protests).
- **Gaussian Noise**: Random shocks.

### 5. Monte Carlo Simulations

The simulation engine generates independent trajectories for each input configuration and computes:
- Fall probability within the simulation horizon.
- Survival curves.
- Importance metrics for internal and external drivers.

---

## Installation

### Prerequisites

Ensure you have the following installed:
- Python (>= 3.8)
- [pip](https://pypi.org/project/pip/)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies Overview

- **Streamlit**: For building the interactive dashboard.
- **Altair**: For generating interactive plots.
- **Pandas**: For data manipulation.
- **Matplotlib** (optional): For additional data exploration.

---

## How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/iteimouri/ir-scenarios.git
    cd ir-scenarios
    ```

2. Start the Streamlit application:
    ```bash
    streamlit run main.py
    ```

3. Open a browser and navigate to the URL displayed in the terminal (typically `http://localhost:8501`).

---

## Interactive Dashboard: Parameters and Outputs

### Input Parameters

#### **Core Simulation Settings**
- **Horizon (months)**: Total number of months to simulate.
- **Number of Simulations**: Number of Monte Carlo trajectories (higher yields smoother estimates but slower runtime).
- **Random Seed**: For reproducibility of simulation results.
- **External Actor Rationality (λ)**: Higher values result in deterministic actions; lower values introduce stochastic decision-making.

#### **Initial Internal State**
All initial internal variables are set between `-1.0` (low intensity) and `1.0` (high intensity).

- `econ₀`: Initial economic stress.
- `protest₀`: Initial protest pressure.
- `repression₀`: Initial repression intensity.
- `elite₀`: Initial elite cohesion.
- `blackout₀`: Initial information blackout intensity.

#### **Scenario Levers**
Adjust scenario-specific inputs like:
- **US Restraint**
- **Gulf Anti-War Pressure**
- **Israel Perceived Threat**
- **Economic Alignment with Russia and China**
- **Sanctions Enforcement by Europe**

#### **Hazard Coefficients**
Modify weights affecting the hazard model (e.g., contribution of protests, economic stress, and their interactions).

#### **Internal Dynamics**
Fine-tune the drift and noise parameters defining internal state evolution.

---

### Outputs and Visualizations

- **P(fall within horizon)**: Probability of regime fall within the specified time horizon.
- **Survival Curve**: Probability of no fall by each month.
- **Monthly Hazard**: Average hazard (fall probability) for each month.
- **Importance Metrics**: Contribution of internal and external drivers to the probability of fall.
- **State Evolution**: Sample paths of internal and external states over time.

---

## Limitations and Responsible Use

### **What this Model Is**
- A **scenario exploration tool** that allows users to analyze the impact of hypothetical interventions in a synthetic model of regime stability.
- Useful for exploring sensitivities, understanding how dynamics interact, and communicating insights in educational or research contexts.

### **What this Model Is Not**
- **Not a prediction tool**: The model is not calibrated to actual data and does not provide real-world forecasts.
- **Not causal analysis**: Although interactions are specified, the model does not establish causal relationships empirically.
- **Not empirical**: Variables and weights are synthetic and stylized, aimed at qualitative understanding rather than quantitative accuracy.

### Limitations
- **Simplistic hazard estimation**: The hazard model assumes linear log-odds with a small set of interactions.
- **Synthetic variables**: No real-world calibration of variables or coefficients.
- **No non-linear dynamics**: Real systems may exhibit regime shifts or non-stationarity that are outside this model's scope.

---

## Contributing

We welcome contributions to this project! Feel free to submit pull requests or open issues for bugs, feature requests, or questions.

1. Fork the repository.
2. Create a new branch for your changes:
    ```bash
    git checkout -b <feature-branch>
    ```
3. Commit and push your changes.
4. Submit a pull request.

---

## Author

- **Ilia Teimouri**  
- Year: **2026**  

You can contribute or raise issues/questions about the model via the repository's GitHub page:
[GitHub Repository](https://github.com/iteimouri/ir-scenarios)

---

## License

This project is licensed under the terms of the MIT license.****
