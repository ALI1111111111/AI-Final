import pandas as pd
df = pd.read_csv('netflix_titles.csv')
# Explore the first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())
# Handle missing values
df['date_added'].fillna('Unknown', inplace=True)
df.dropna(subset=['rating', 'duration'], inplace=True)
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a box plot to visualize the distribution of ratings
plt.figure(figsize=(12, 8))
sns.boxplot(x='rating', y='release_year', data=df, palette="viridis")
plt.title('Box Plot of Movie Ratings by Release Year')
plt.xlabel('Rating')
plt.ylabel('Release Year')
plt.xticks(rotation=45)
plt.show()

# Perform clustering (example using KMeans)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Encode the categorical ratings to numeric values
le = LabelEncoder()
df['rating_encoded'] = le.fit_transform(df['rating'])

# Perform clustering
kmeans = KMeans(n_clusters=5)
df['cluster'] = kmeans.fit_predict(df[['rating_encoded']])

