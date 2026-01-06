# README

This document serves as a preliminary README and outlines the steps taken so far in this project (as of **December 2025**).

---

## Project Overview

This project aims to train a machine learning model to **nowcast and forecast neutral temperature and wind** using **Fabry–Pérot Interferometer (FPI)** measurements in the **Scandinavian region**.

The FPI data used are restricted to altitudes between **120–250 km**, corresponding to atomic oxygen airglow emissions:

### Atomic Oxygen Emissions

- **Green line emission**
  - Wavelength: **557.7 nm**
  - Origin: Atomic oxygen
  - Typical altitude: **~120 km**

- **Red line emission**
  - Wavelength: **630.0 nm** (from excited atomic oxygen, O(¹D))
  - Typical peak emission altitude: **~240 km**

These altitude constraints define the vertical range of the thermospheric parameters modelled in this study.

---

## Data Sources

### FPI Data

- **Source:** UCL Indra server
- **Available temporal coverage:** 1970–2025
- **Current usage:** 2013–2025
- **Future plan:** Extend the dataset back by one additional solar cycle to better capture solar variability and improve model generalisability across different solar conditions.

### Auxiliary Space Weather Data

In addition to the FPI observations, the following datasets are incorporated:

- **Geomagnetic data**  
  Source: https://kp.gfz.de/en/hp30-hp60/data
- **Solar radio flux data**  
  Source: https://spaceweather.cls.fr/services/radioflux/
- **Interplanetary Magnetic Field (IMF) data**  
  Source: https://omniweb.gsfc.nasa.gov/form/dx1.html

---

## Data Integration and Processing

Each auxiliary dataset is temporally aligned with the FPI measurements by:

1. Identifying the closest datetime between each FPI observation and the corresponding external data streams.
2. Joining the datasets based on this temporal proximity.

Additionally:

- Time-series data covering **X hours prior** to each FPI observation are included.
- The sampling rate (**Y**) and time-window length are user-defined inputs.
- These historical features are appended to the main dataset.

After preprocessing, a **Graph Neural Network (GNN)** data object is created using the `data_processing.py` script.

---

## Model Training

Model training is performed using the `run_main.py` script, which orchestrates:

- `gnn_model.py`
- `train_{model_type}.py`

The GNN is trained to predict:

- **Neutral temperature**
- **Neutral wind speed**

at the **location and time of the FPI measurement**.

---

## Validation Strategy

Model validation is performed using **independent measurement sources**, prioritising **non-FPI** and preferably **in-situ** observations. Current validation datasets include:

- **WINDII (UARS satellite)**
  - Time span: **1991–1997**
  - Measurement type: Limb-scanning (quasi in-situ)

- **Atmospheric Explorer missions**
  - Time span: **1975–1981**
  - Measurement type: **Direct in-situ measurements**

- **Dynamics Explorer missions**
  - Time span: **1981–1985**
  - Measurement type: **Direct in-situ measurements**

These datasets provide independent validation across different time periods, measurement techniques, and solar conditions, and can be treated as a **gold standard** for model comparison.

---

## Model Benchmarking

To assess the performance of the trained GNN relative to the current state of the field, the model will be compared against established empirical and semi-empirical models:

### Thermospheric Density and Temperature Models
- **NRLMSIS-2.0**
- **JB2008** (subject to data availability)
- **DTM2020** (subject to data availability)

### Neutral Wind Models
- **HWM2014**
- **HL-TWIM** (if resources permit)

These models are widely used by satellite operators and are commonly adopted as benchmarks when evaluating new thermospheric modelling approaches.

---

## Analysis and Visualisation

The primary analysis outputs will include:

- Time-series comparisons during **geomagnetically quiet and active periods**
- Direct comparisons between the GNN and benchmark models
- Statistical metrics such as:
  - Coefficient of determination (**R²**)
  - Regression coefficients
- **Taylor diagrams** to summarise model performance concisely

The use of Taylor plots was inspired by work from **Sean Elvidge**.

---

## Current Issues and Limitations

One key limitation is that **ap30 data are only available from 1985 onwards**, which complicates validation using direct in-situ measurements at earlier times.

A potential mitigation strategy is to use the **ap index** for earlier periods, under the assumption that it provides sufficient proxy information for geomagnetic activity. While this is not ideal, it may be the most practical option.

A detailed review of thermospheric wind measurements is provided in the following paper:  
https://doi.org/10.3389/fspas.2022.1050586

Based on this review, the available validation datasets outlined above represent the most realistic options for this study.
