import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans,DBSCAN

# 03
df = pd.read_csv('Mall_Customers.csv')
print(df)
df.info()


print()
df = df.drop('CustomerID',axis=1)
print(df)

print()
df.Gender.value_counts()



df.Gender.value_counts().plot(kind='pie',autopct="%.1f%%")
plt.title('Gender Distributions Data')
plt.legend()
plt.show()



sns.kdeplot(df['Age'])
plt.show()



sns.kdeplot(df['Annual Income (k$)'])
plt.show()



sns.kdeplot(df['Spending Score (1-100)'])
plt.show()


df['Gender'] = df['Gender'].replace({'Female':0,'Male':1})
df.head()




# missed 20

plt.figure(figsize=(10, 10))
i = 1
for col in df.columns:
    plt.subplot(2, 2, i)
    sns.histplot(df[col])
    plt.title(f'Histogram of {col}')
    i += 1
    if i > 4:  # Ensure not to exceed the number of subplots
        break

plt.figure(figsize=(10, 10))
i = 1
for col in df.columns:
    plt.subplot(2, 2, i)
    df[[col]].boxplot()
    plt.title(f'Boxplot of {col}')
    i += 1
    if i > 4:  # Ensure not to exceed the number of subplots
        break

plt.show()





df.head()




# 25
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled.shape


# 26
df_scaled_data = pd.DataFrame(df_scaled,columns=df.columns)
df_scaled_data

# 28
import warnings
warnings.filterwarnings('ignore')




# 31
inertia_data = []
for k in range(1,15):
    model = KMeans(n_clusters=k)
    model.fit(df_scaled_data)
    inertia_data.append(model.inertia_)


# 32
inertia_data



# 47
plt.plot(range(1,15),inertia_data,color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12)
plt.title("Elbow Method")
plt.xlabel("Number Of Clusters")
plt.ylabel("Inertia Of Model")
plt.show()



# 48
from yellowbrick.cluster import KElbowVisualizer

model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(1,15))

visualizer.fit(df_scaled_data)
visualizer.show()
plt.show()


# 50
from yellowbrick.cluster import KElbowVisualizer

model = KMeans(random_state=1)
visualizer = KElbowVisualizer(model, k=(2,15),metric='silhouette')

visualizer.fit(df_scaled_data)
visualizer.show()
plt.show()



# 51

from yellowbrick.cluster import SilhouetteVisualizer
model = KMeans(n_clusters=5,)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(df_scaled_data)
visualizer.show()
plt.show()



# 52
from yellowbrick.cluster import SilhouetteVisualizer
model = KMeans(n_clusters=5,)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(df_scaled_data)
visualizer.show()
plt.show()


# 53
# Calculate silhouette score for different values of k
silhouette_scores = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(df_scaled_data)
    silhouette_avg = silhouette_score(df_scaled_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)


    # 59
    silhouette_scores


# 67
plt.plot(silhouette_scores,range(2,15),color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12)
plt.title("Silhouette Score and Cluster Distributions")
plt.xlabel("Silhouette Score")
plt.ylabel("No. Of Cluster")
plt.show()


# 68
m = KMeans(n_clusters=5)
y_pred = m.fit_predict(df_scaled_data)

# 69
y_pred    # Predicted Labels if K = 5

# Add the following lines outside of any loops or conditional statements
# These lines will display the array
print(y_pred)

print()
print("Second Array ")
print()


# 70
m1 = KMeans(n_clusters=10)
y_pred1 = m1.fit_predict(df_scaled_data)
y_pred1    # Predicted Labels if K = 10
print(y_pred1)


print()
print()

# 72
df['K_5_y_pred'] = y_pred
df['K_10_y_pred'] = y_pred1
print(df)



# 78
sns.scatterplot(x=df['Age'],y=df['Annual Income (k$)'],hue=df['K_5_y_pred'],palette=['Red','Blue','Green','Yellow','Violet'])
plt.title(f"No. Of Cluster:{5} Distributions Labels")
plt.show()



# 80
sns.scatterplot(x=df['Age'],y=df['Annual Income (k$)'],hue=df['K_10_y_pred'])#palette=['Red','Blue','Green','Yellow','Violet'])
plt.title(f"No. Of Cluster:{10} Distributions Labels")
plt.show()