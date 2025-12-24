import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from math import radians, cos, sin, sqrt, atan2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import folium
import webbrowser


# Load data from CSV files instead of MongoDB
try:
    df_tunis = pd.read_csv("tunis_tourism.csv")
except Exception:
    df_tunis = pd.DataFrame()

try:
    df_world = pd.read_csv("world_tourism.csv")
except Exception:
    df_world = pd.DataFrame()

if not df_tunis.empty and 'subsubcategory' in df_tunis.columns and 'price' in df_tunis.columns:
    df_tunis['destinations_features'] = df_tunis['subsubcategory'].astype(str) + ' ' + df_tunis['price'].astype(str)
else:
    df_tunis['destinations_features'] = ''


def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
    return R * c


def nearest_restaurant(target_lon, target_lat, df, n=1):
    distances = {i: haversine_distance(target_lon, target_lat, row['longitude'], row['latitude'])
                 for i, row in df.iterrows() if pd.notna(row['longitude']) and pd.notna(row['latitude'])}
    nearest_indices = sorted(distances, key=distances.get)[:n]
    return df.loc[[nearest_indices[0]]]

def recommend_destinations(user_preferences_list, df, top_n_destinations=3):
    user_preferences = " ".join(user_preferences_list)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['destinations_features'])
    user_vec = vectorizer.transform([user_preferences])
    sim = cosine_similarity(user_vec, X)[0]
    df["text_score"] = sim
    top_text = df.sort_values("text_score", ascending=False).head(10)
    
    closeness_scores = []
    for i, row_i in top_text.iterrows():
        total_distance = 0
        for j, row_j in top_text.iterrows():
            if i != j:
                total_distance += haversine_distance(row_i["longitude"], row_i["latitude"],
                                                     row_j["longitude"], row_j["latitude"])
        closeness_scores.append((i, total_distance))
    
    closeness_scores = sorted(closeness_scores, key=lambda x: x[1])
    selected_indices = [idx for idx,_ in closeness_scores[:top_n_destinations]]
    return df.loc[selected_indices]


class TunisDashboard(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.df = df_tunis
        self.pack(fill='both', expand=True)
        
        # Layout
        self.left_frame = tk.Frame(self, width=300)
        self.left_frame.pack(side="left", fill="y")
        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side="right", fill="both", expand=True)
        
        tk.Label(self.left_frame, text="Select Graph", font=("Arial", 14)).pack(pady=10)
        self.graph_var = tk.StringVar()
        graphs = [
            "KDE Heatmap", "Cluster Scatter", "Category Count", "Avg Rating per Category",
            "Subcategory Count", "Rating Distribution", "Price Distribution", "Duration Distribution",
            "Top 10 Rated", "GoodFor Distribution", "Subsubcategory Distribution"
        ]
        ttk.Combobox(self.left_frame, textvariable=self.graph_var, values=graphs).pack(pady=5)
        tk.Button(self.left_frame, text="Generate", command=self.render_graph).pack(pady=20)
        
        tk.Label(self.left_frame, text="Category Filter", font=("Arial", 12)).pack(pady=5)
        self.category_var = tk.StringVar()
        categories = ["All"] + sorted(self.df["category_name"].dropna().unique())
        ttk.Combobox(self.left_frame, textvariable=self.category_var, values=categories).pack()
        
        
        self.canvas_frame = tk.Frame(self.right_frame)
        self.canvas_frame.pack(fill="both", expand=True)
    
    def apply_filters(self):
        df = self.df.copy()
        if self.category_var.get() != "All":
            df = df[df["category_name"]==self.category_var.get()]
        return df
    
    def render_graph(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        df = self.apply_filters()
        fig = plt.figure(figsize=(8,6))
        ax = None
        g = self.graph_var.get()
        
        if g == "KDE Heatmap":
            sns.kdeplot(data=df, x="longitude", y="latitude", fill=True, cmap="inferno", thresh=0.05)
        elif g == "Cluster Scatter":
            sns.scatterplot(data=df, x="longitude", y="latitude", hue="cluster", palette="tab10", s=40)
        elif g == "Category Count":
            df['category_name'].value_counts().plot(kind='bar')
        elif g == "Avg Rating per Category":
            df.groupby('category_name')['rating'].mean().sort_values().plot(kind='bar')
        elif g == "Subcategory Count":
            df['subcategory_name'].value_counts().plot(kind='barh')
        elif g == "Rating Distribution":
            plt.hist(df['rating'], bins=5, edgecolor='black')
        elif g == "Price Distribution":
            df['price'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        elif g == "Duration Distribution":
            df['Duration'].value_counts().plot(kind='bar')
        elif g == "Top 10 Rated":
            top10 = df.sort_values(by='rating', ascending=False).head(10)
            plt.barh(top10['name'], top10['rating'])
            plt.gca().invert_yaxis()
        elif g == "GoodFor Distribution":
            df['GoodFor'].value_counts().plot(kind='bar')
        elif g == "Subsubcategory Distribution":
            df['subsubcategory'].value_counts().plot(kind='barh')
        try:
            ax = plt.gca()
            xticks = ax.get_xticklabels()
            if xticks:
                plt.setp(xticks, rotation=45, ha='right')
                fig.subplots_adjust(bottom=0.25)
        except Exception:
            pass

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

class WorldDashboard(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.df = df_world
        self.pack(fill='both', expand=True)
        
        self.left_frame = tk.Frame(self, width=300)
        self.left_frame.pack(side="left", fill="y")
        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side="right", fill="both", expand=True)
        
        tk.Label(self.left_frame, text="Select Graph", font=("Arial", 14)).pack(pady=10)
        self.graph_var = tk.StringVar()
        graphs = [
            "Locations per Country", "Country Distribution", "Category Count", "Category Distribution",
            "Accommodation Available", "Accommodation per Country", "Revenue by Category", "Revenue by Country/Category"
        ]
        ttk.Combobox(self.left_frame, textvariable=self.graph_var, values=graphs).pack(pady=5)
        tk.Button(self.left_frame, text="Generate", command=self.render_graph).pack(pady=20)
        
        tk.Label(self.left_frame, text="Country Filter", font=("Arial", 12)).pack(pady=5)
        self.country_var = tk.StringVar()
        countries = ["All"] + sorted(self.df["Country"].dropna().unique())
        ttk.Combobox(self.left_frame, textvariable=self.country_var, values=countries).pack()
        
        tk.Label(self.left_frame, text="Category Filter", font=("Arial", 12)).pack(pady=5)
        self.category_var = tk.StringVar()
        categories = ["All"] + sorted(self.df["Category"].dropna().unique())
        ttk.Combobox(self.left_frame, textvariable=self.category_var, values=categories).pack()
        
        
        self.canvas_frame = tk.Frame(self.right_frame)
        self.canvas_frame.pack(fill="both", expand=True)
    
    def apply_filters(self):
        df = self.df.copy()
        if self.country_var.get() != "All":
            df = df[df["Country"]==self.country_var.get()]
        if self.category_var.get() != "All":
            df = df[df["Category"]==self.category_var.get()]
        return df
    
    def render_graph(self):
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        df = self.apply_filters()
        fig = plt.figure(figsize=(8,6))
        g = self.graph_var.get()
        
        if g == "Locations per Country":
            data = df.groupby('Country')['Category'].count().reset_index(name='Count')
            axis = sns.barplot(x='Country', y='Count', data=data)
            axis.bar_label(axis.containers[0])
            plt.ylabel('Number of locations')
            plt.title('Number of locations by Country')
        elif g == "Country Distribution":
            plt.pie(df['Country'].value_counts(), labels=df['Country'].value_counts().index, autopct='%1.1f%%')
            plt.title('Country Distribution')
        elif g == "Category Count":
            axis = sns.countplot(data=df, x="Category")
            axis.bar_label(axis.containers[0])
        elif g == "Category Distribution":
            plt.pie(df['Category'].value_counts(), labels=df['Category'].value_counts().index, autopct='%1.1f%%')
            plt.title('Category Distribution')
        elif g == "Accommodation Available":
            sns.countplot(x=df["Accommodation_Available"])
            plt.title("Accommodation Available Count")
        elif g == "Accommodation per Country":
            sns.histplot(df['Country'], color='teal')
            plt.title('Accommodation Available vs Country')
            plt.xlabel('Country')
            plt.ylabel('Accommodation Available')
        elif g == "Revenue by Category":
            revenue_per_Category = df.groupby('Category')['Revenue'].sum().reset_index()
            sns.lineplot(x='Category', y='Revenue', data=revenue_per_Category, marker='o', color='red')
            plt.title('Revenue by Category')
            plt.xlabel('Category')
            plt.ylabel('Revenue')
            plt.xticks(rotation=45)
            plt.grid(True)
        elif g == "Revenue by Country/Category":
            revenue_per_country = df.groupby(['Country', 'Category'])['Revenue'].sum().reset_index()
            sns.lineplot(x='Country', y='Revenue', hue="Category", data=revenue_per_country, marker='o')
            plt.title('Revenue by Country and Category')
            plt.xlabel('Country')
            plt.ylabel('Revenue')
            plt.xticks(rotation=45)
            plt.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

class RecommendationDashboard(ttk.Frame):
    def __init__(self, parent, df):
        super().__init__(parent)
        self.df = df
        self.pack(fill='both', expand=True)

        # Header
        tk.Label(self, text="Get Recommendations", font=("Arial", 14)).pack(pady=8)

        frm = ttk.Frame(self)
        frm.pack(fill='x', padx=12, pady=6)

        # Category combobox
        ttk.Label(frm, text="Category:").grid(row=0, column=0, sticky='w')
        self.category_var = tk.StringVar()
        cats = [c for c in sorted(self.df['category_name'].dropna().unique())] if not self.df.empty else []
        self.category_cb = ttk.Combobox(frm, textvariable=self.category_var, values=[''] + cats, state='readonly')
        self.category_cb.grid(row=0, column=1, sticky='ew', padx=6, pady=2)
        self.category_cb.bind('<<ComboboxSelected>>', self._on_category_change)

        # Subcategory combobox
        ttk.Label(frm, text="Subcategory:").grid(row=1, column=0, sticky='w')
        self.subcategory_var = tk.StringVar()
        self.subcategory_cb = ttk.Combobox(frm, textvariable=self.subcategory_var, values=[''], state='readonly')
        self.subcategory_cb.grid(row=1, column=1, sticky='ew', padx=6, pady=2)
        self.subcategory_cb.bind('<<ComboboxSelected>>', self._on_subcategory_change)

        # Subsubcategory combobox
        ttk.Label(frm, text="Subsubcategory:").grid(row=2, column=0, sticky='w')
        self.subsubcategory_var = tk.StringVar()
        self.subsubcategory_cb = ttk.Combobox(frm, textvariable=self.subsubcategory_var, values=[''], state='readonly')
        self.subsubcategory_cb.grid(row=2, column=1, sticky='ew', padx=6, pady=2)

        # Price combobox 
        ttk.Label(frm, text="Price:").grid(row=3, column=0, sticky='w')
        self.price_var = tk.StringVar()
        self.price_cb = ttk.Combobox(frm, textvariable=self.price_var, values=[''], state='readonly')
        self.price_cb.grid(row=3, column=1, sticky='ew', padx=6, pady=2)

        # Number of destinations
        ttk.Label(frm, text="Number of results:").grid(row=4, column=0, sticky='w')
        self.top_n_spin = tk.Spinbox(frm, from_=1, to=20, width=5)
        self.top_n_spin.grid(row=4, column=1, sticky='w', padx=6, pady=6)

        frm.columnconfigure(1, weight=1)

        # Buttons
        btns = ttk.Frame(self)
        btns.pack(fill='x', padx=12)
        ttk.Button(btns, text="Get Recommendations", command=self.recommend).pack(side='left')
        ttk.Button(btns, text="Open last map", command=self._open_last_map).pack(side='left', padx=8)

        # Results area
        ttk.Label(self, text="Recommendations:").pack(anchor='w', padx=12, pady=(10,0))
        self.output_text = tk.Text(self, height=10)
        self.output_text.pack(fill="both", padx=12, pady=(0,12), expand=True)

        self.last_recommended = pd.DataFrame()
    
    def recommend(self):
        # Build preference list from inputs
        prefs = []
        cat = self.category_var.get().strip()
        sub = self.subcategory_var.get().strip()
        subsub = self.subsubcategory_var.get().strip()
        price = self.price_var.get().strip()
        if cat:
            prefs.append(str(cat))
        if sub:
            prefs.append(str(sub))
        if subsub:
            prefs.append(str(subsub))
        if price:
            prefs.append(str(price))

        if not prefs:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "Please choose at least a category or provide a price.")
            return

        try:
            top_n = int(self.top_n_spin.get())
        except Exception:
            top_n = 3

        # Calling recommendation 
        recommended_destinations = recommend_destinations(prefs, self.df, top_n)

        # Find nearby restaurant 
        restaurants_df = self.df[self.df.get('subcategory_name', pd.Series()) == 'Restaurants'] if not self.df.empty else pd.DataFrame()
        if not recommended_destinations.empty and recommended_destinations['longitude'].notna().any():
            center_lon = recommended_destinations['longitude'].astype(float).mean()
            center_lat = recommended_destinations['latitude'].astype(float).mean()
        else:
            center_lon = 0.0
            center_lat = 0.0

        nearby_restaurants = pd.DataFrame()
        if not restaurants_df.empty and center_lon != 0.0:
            nearby_restaurants = nearest_restaurant(center_lon, center_lat, restaurants_df, n=1)

        combined_df = pd.concat([recommended_destinations, nearby_restaurants]).drop_duplicates()

        # Save last recommended
        self.last_recommended = combined_df.copy()

        # Create map 
        try:
            if not combined_df.empty and combined_df['latitude'].notna().any():
                m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                points = []
                for _, row in combined_df.iterrows():
                    lat = row.get('latitude')
                    lon = row.get('longitude')
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    points.append([lat, lon])
                    folium.Marker(
                        location=[lat, lon],
                        popup=str(row.get('name', '')),
                        icon=folium.Icon(color='green' if row.get('subcategory_name') == 'Restaurants' else 'blue')
                    ).add_to(m)
                if len(points) > 1:
                    folium.PolyLine(points, color='blue', weight=3, opacity=0.7).add_to(m)
                m.save("recommendation_map.html")
                webbrowser.open("recommendation_map.html")
        except Exception:
            pass

        # Display 
        self.output_text.delete("1.0", tk.END)
        if recommended_destinations.empty:
            self.output_text.insert(tk.END, "No recommendations found for the selected inputs.")
        else:
            try:
                cols = [c for c in ['name', 'category_name', 'subcategory_name', 'subsubcategory', 'price', 'rating'] if c in recommended_destinations.columns]
                self.output_text.insert(tk.END, recommended_destinations[cols].to_string(index=False))
            except Exception:
                self.output_text.insert(tk.END, recommended_destinations.to_string(index=False))
        if not nearby_restaurants.empty:
            self.output_text.insert(tk.END, "\n\nNearby Restaurant:\n")
            try:
                self.output_text.insert(tk.END, nearby_restaurants[['name','subcategory_name']].to_string(index=False))
            except Exception:
                self.output_text.insert(tk.END, nearby_restaurants.to_string(index=False))

    def _on_category_change(self, *_):
        sel = self.category_var.get()
        if not sel or self.df.empty:
            self.subcategory_cb['values'] = ['']
            self.subsubcategory_cb['values'] = ['']
            return
        subs = sorted(self.df[self.df['category_name'] == sel]['subcategory_name'].dropna().unique())
        self.subcategory_cb['values'] = [''] + subs
        self.subcategory_var.set('')
        self.subsubcategory_cb['values'] = ['']
        self.subsubcategory_var.set('')

    def _on_subcategory_change(self, *_):
        cat = self.category_var.get()
        sub = self.subcategory_var.get()
        if not cat or not sub or self.df.empty:
            self.subsubcategory_cb['values'] = ['']
            return
        subsubs = sorted(self.df[(self.df['category_name'] == cat) & (self.df['subcategory_name'] == sub)]['subsubcategory'].dropna().unique())
        self.subsubcategory_cb['values'] = [''] + subsubs
        self.subsubcategory_var.set('')
        self.price_cb['values'] = ['']
        self.price_var.set('')
        self.subsubcategory_cb.bind('<<ComboboxSelected>>', self._on_subsubcategory_change)

    def _on_subsubcategory_change(self, *_):
        cat = self.category_var.get()
        sub = self.subcategory_var.get()
        subsub = self.subsubcategory_var.get()
        if not cat or not sub or not subsub or self.df.empty:
            self.price_cb['values'] = ['']
            self.price_var.set('')
            return
        prices = self.df[(self.df['category_name'] == cat) & (self.df['subcategory_name'] == sub) & (self.df['subsubcategory'] == subsub)]['price'].dropna().unique()
        price_vals = sorted([str(p) for p in prices]) if len(prices) > 0 else []
        self.price_cb['values'] = [''] + price_vals
        self.price_var.set('')

    def _open_last_map(self):
        try:
            import os
            if os.path.exists('recommendation_map.html'):
                webbrowser.open('recommendation_map.html')
            else:
                self.output_text.delete('1.0', tk.END)
                self.output_text.insert(tk.END, 'No map file found. Generate recommendations first.')
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Complete Tourism Dashboard")
    root.geometry("1400x800")
    
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)
    
    tab_tunisia = ttk.Frame(notebook)
    tab_world = ttk.Frame(notebook)
    tab_recommend = ttk.Frame(notebook)
    
    notebook.add(tab_tunisia, text="Tunisia Tourism")
    notebook.add(tab_world, text="World Tourism")
    notebook.add(tab_recommend, text="Tunisia Recommendations")
    
    TunisDashboard(tab_tunisia)
    WorldDashboard(tab_world)
    RecommendationDashboard(tab_recommend, df_tunis)
    
    root.mainloop()
