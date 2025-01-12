from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
df = pd.DataFrame(data= heart_disease.data.features)

df["target"] = heart_disease.data.targets

# drop missing value this is a preprocess
if df.isna().any().any():
    df.dropna(inplace=True)
    print("nan")

# Prepare the data for model: X are the features, y is the target
X = df.drop(["target"], axis=1).values
y = df.target.values

# Split the data into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize and train the logistic regression model
log_reg = LogisticRegression(penalty="l2", C=1, solver="lbfgs", max_iter=100)
log_reg.fit(X_train, y_train)

y_pred= log_reg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix'i görselleştirelim
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek Değer")
plt.show()

print("Confusion Matris")
print(cm)
train_accuracy= log_reg.score(X_train, y_train);
train_accuracy = log_reg.score(X_train, y_train)
print("Training Accuracy Score:", train_accuracy)

# Evaluate the model using accuracy score
accuracy = log_reg.score(X_test, y_test)
print("Test Accuracy Score:", accuracy)
