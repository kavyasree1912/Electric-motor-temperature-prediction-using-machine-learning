---

# Electric Motor Temperature Prediction Using Machine Learning

## 📌 Project Overview

Electric motors are critical components in industrial and automation systems. Overheating can lead to reduced efficiency, equipment damage, and unexpected failures. This project uses **Machine Learning techniques** to predict the operating temperature of an electric motor based on sensor data, enabling **early fault detection and predictive maintenance**.

---

## 🎯 Objectives

* Predict electric motor temperature accurately using ML models
* Prevent overheating and sudden motor failures
* Improve motor efficiency and lifespan
* Support predictive maintenance strategies

---

## 🧠 Machine Learning Models Used

* Linear Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

The models are trained on historical sensor data such as motor speed, torque, current, voltage, and environmental parameters.

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn, NumPy, Pandas
* **Web Interface:** HTML, CSS
* **Backend:** Flask
* **Visualization:** Matplotlib / Seaborn

---

## 📁 Project Structure

```
Electric Motor Temperature Prediction/
│
├── Code/                     # Frontend HTML files
│   ├── index.html
│   ├── sensor.html
│   └── Manual.html
│
├── Main/                     # Backend & ML code
│   ├── app.py
│   ├── train_model.py
│   ├── sensor_model_train.py
│
├── Output/                   # Project screenshots
│   ├── Home.png
│   ├── Sensor.png
│   ├── Manual.png
│   └── Prediction_Result.png
│
├── data/                     # Dataset (ignored in GitHub)
│
├── .gitignore
└── README.md
```

---

## 🖼️ Screenshots

**Home Page**
![Home](Output/Home.png)

**Manual Input Page**
![Manual](Output/Manual.png)

**Sensor Input Page**
![Sensor](Output/Sensor.png)

**Prediction Result**
![Result](Output/Senor_Prediction_result.png)

---

## 📊 Dataset & Model Files

⚠️ Due to GitHub size limitations, **dataset (`.csv`) and trained model files (`.pkl`) are not included** in this repository.

* Dataset used: `measures_v2.csv`
* Model files are generated locally after training

To generate models locally, run:

```bash
python train_model.py
python sensor_model_train.py
```

---

## 🚀 How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/nehashinde8836/Electric-Motor-Temperature-Prediction-Using-Machine-Learning.git
```

2. Install required libraries

```bash
pip install -r requirements.txt
```

3. Run the Flask application

```bash
python app.py
```

4. Open browser and go to

```
http://127.0.0.1:5000
```

---

## ✅ Applications

* Industrial motor monitoring
* Predictive maintenance systems
* Electric vehicles
* Smart manufacturing
* Automation industries

---

## 🔮 Future Enhancements

* Real-time sensor data integration
* Cloud-based deployment
* Advanced deep learning models
* Live dashboards and analytics

---

## 📌 Conclusion

This project demonstrates how **Machine Learning can be effectively used to predict electric motor temperature**, helping industries prevent failures, reduce downtime, and improve operational efficiency. It showcases the practical use of ML in real-world predictive maintenance applications.

---
