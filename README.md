# 🇹🇳 Tunisia Tourism Recommendation & Visualization App

This project is a Python desktop application that analyzes tourism data and provides personalized travel recommendations in Tunisia using data science, machine learning, and interactive visualization.

The application is developed with **Tkinter** for the graphical interface and integrates data analysis, recommendation logic, and interactive maps in a single desktop environment.

![Architecture Diagram](https://img.shields.io/badge/Architecture-CountVectorizer%20%2B%20Cosine_Similarity-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)


## ✨ Features

### 📊 Tourism Data Analysis
* **Comprehensive Analysis:** Analysis of Tunisian and world tourism datasets.
* **Visual Insights:** Generates charts, comparisons, and trends using Matplotlib.

### 🎯 Smart Recommendation System
* **User Preferences:** Users can select:
    * Category
    * Subcategory
    * Sub-subcategory
    * Price range
* **Hybrid Logic:** Recommendations are generated based on:
    * **Text Similarity:** Matching user preferences vs. destinations.
    * **Geographic Proximity:** Calculating distances using the Haversine formula.

### 🗺️ Interactive Map Inside the App
* **Embedded Maps:** Recommended destinations are displayed on a map rendered directly inside the application (not in a browser).
* **Route Planning:** Routes are drawn automatically between selected locations.
* **Smart Suggestions:** The nearest restaurant is automatically added to the trip.

### 🖥️ User-Friendly Desktop Interface
* **GUI Based:** No command-line interaction required.
* **Easy Controls:** Dropdowns and input fields for preferences.
* **Interactive:** Buttons to generate recommendations and visual reports instantly.

## 🧠 Technologies Used
* **Python**
* **Tkinter** – Desktop Graphical User Interface (GUI)
* **Pandas & NumPy** – Data processing and manipulation
* **Matplotlib** – Data visualization and plotting
* **Scikit-learn** – Recommendation system logic (text similarity)
* **Folium + HTML rendering** – Interactive map integration
* **Geographic calculations** – Haversine distance formula

## 💻 Installation & Usage

### 1. Clone the Repository
Open your terminal or command prompt and run the following command to download the project:

```bash
git clone [https://github.com/MohamedAliWerda/Tunisia-Tourism-Recommendation-Visualization-App.git](https://github.com/MohamedAliWerda/Tunisia-Tourism-Recommendation-Visualization-App.git)
cd Tunisia-Tourism-Recommendation-Visualization-App
```
### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn folium
```
### 3. Run the Application

```bash
python tourism.py
```
