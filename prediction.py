# Importing required libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from scipy.stats import randint, uniform
import random

# Implementation of HyperbandSearchCV
class HyperbandSearchCV:
    def __init__(self, estimator, param_distributions, max_iter=81, eta=3, cv=5, scoring='accuracy', random_state=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.max_iter = max_iter
        self.eta = eta
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        self.best_estimator_ = None
        self.best_score_ = -np.inf
        self.best_params_ = None
        
    def _sample_params(self):
        params = {}
        for param, distribution in self.param_distributions.items():
            if hasattr(distribution, 'rvs'):
                params[param] = distribution.rvs(random_state=self.random_state)
            elif isinstance(distribution, list):
                params[param] = random.choice(distribution)
            elif isinstance(distribution, tuple) and len(distribution) == 2:
                # Handle (low, high) tuples for integers
                if all(isinstance(x, int) for x in distribution):
                    params[param] = randint(distribution[0], distribution[1]).rvs(random_state=self.random_state)
                # Handle (low, high) tuples for floats
                elif all(isinstance(x, float) for x in distribution):
                    params[param] = uniform(distribution[0], distribution[1]).rvs(random_state=self.random_state)
            else:
                params[param] = distribution
        return params
    
    def _try_params(self, n_iterations, params):
        model = clone(self.estimator)
        model.set_params(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=self.cv, scoring=self.scoring, n_jobs=-1)
        return np.mean(scores)
    
    def fit(self, X, y):
        global X_train_scaled, y_train
        X_train_scaled, y_train = X, y
        s_max = int(np.log(self.max_iter) / np.log(self.eta))
        B = (s_max + 1) * self.max_iter
        
        for s in reversed(range(s_max + 1)):
            n = int(np.ceil(B / self.max_iter * (self.eta ** s) / (s + 1)))
            r = self.max_iter * (self.eta ** (-s))
            
            for i in range(n):
                params = self._sample_params()
                score = self._try_params(int(r), params)
                
                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params
                    self.best_estimator_ = clone(self.estimator)
                    self.best_estimator_.set_params(**params)
                    self.best_estimator_.fit(X, y)
        
        return self

# Load and preprocess the dataset
heart = pd.read_csv("heart_cleveland.csv")

# Renaming columns for consistency
heart = heart.rename(columns={'condition': 'target'})

# Display basic information about the dataset
print("Dataset Overview:\n")
print(heart.info())
print("\nFirst 5 rows of the dataset:")
print(heart.head())
print("\nDataset Statistics:")
print(heart.describe())

# Check for missing values
print("\nChecking for missing values in the dataset:")
print(heart.isnull().sum())

# Correlation heatmap visualization
plt.figure(figsize=(12, 10))
corr = heart.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# Splitting features and target
X = heart.drop(columns='target')
y = heart['target']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize base models for stacking
rf = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()
base_models = [('Random Forest', rf), ('K-Nearest Neighbors', knn)]

# Create a stacking classifier with Logistic Regression as the final estimator
stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    stack_method='auto',
    n_jobs=-1,
    passthrough=False
)

# Update models dictionary to include the stacked model
models = {
    'Random Forest': rf,
    'K-Nearest Neighbors': knn,
    'Stacked RF+KNN': stacked_model,
    'Support Vector Machine': SVC(random_state=42, probability=True),  # Added probability=True for ROC curve
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Evaluate models
print("\nEvaluating Models:\n")
results = {}
for model_name, model in models.items():
    # Perform cross-validation
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    avg_score = np.mean(scores)
    results[model_name] = avg_score
    print(f"{model_name} - Cross-Validation Accuracy: {avg_score:.4f}")

# Selecting the best model based on accuracy
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\nBest Model Based on Cross-Validation: {best_model_name}")
print(f"Cross-Validation Accuracy of Best Model: {results[best_model_name]:.4f}")

# Train the best model on the training data
best_model.fit(X_train_scaled, y_train)

# Hyperparameter tuning for the best model using Hyperband
print(f"\nPerforming Hyperparameter Tuning for {best_model_name} using Hyperband...")

if best_model_name == "Random Forest":
    param_distributions = {
        'n_estimators': (50, 300),
        'max_depth': (1, 50),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
        'max_features': ['sqrt', 'log2', None]
    }
    hyperband = HyperbandSearchCV(best_model, param_distributions, max_iter=81, eta=3, cv=5, scoring='accuracy', random_state=42)
    hyperband.fit(X_train_scaled, y_train)
    best_model = hyperband.best_estimator_
    print(f"Best Parameters for {best_model_name}: {hyperband.best_params_}")
    print(f"Best Cross-Validation Score: {hyperband.best_score_:.4f}")

elif best_model_name == "K-Nearest Neighbors":
    param_distributions = {
        'n_neighbors': (3, 15),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    hyperband = HyperbandSearchCV(best_model, param_distributions, max_iter=81, eta=3, cv=5, scoring='accuracy', random_state=42)
    hyperband.fit(X_train_scaled, y_train)
    best_model = hyperband.best_estimator_
    print(f"Best Parameters for {best_model_name}: {hyperband.best_params_}")
    print(f"Best Cross-Validation Score: {hyperband.best_score_:.4f}")

elif best_model_name == "Stacked RF+KNN":
    # Tune base models and final estimator using Hyperband
    print("\nTuning base models for Stacked RF+KNN using Hyperband...")
    
    # Tune Random Forest
    rf_param_distributions = {
        'n_estimators': (50, 300),
        'max_depth': (1, 50),
        'min_samples_split': (2, 10)
    }
    rf_hyperband = HyperbandSearchCV(RandomForestClassifier(random_state=42), rf_param_distributions, max_iter=81, eta=3, cv=5, scoring='accuracy')
    rf_hyperband.fit(X_train_scaled, y_train)
    best_rf = rf_hyperband.best_estimator_
    print(f"Best RF Parameters: {rf_hyperband.best_params_}")
    print(f"Best RF Score: {rf_hyperband.best_score_:.4f}")
    
    # Tune KNN
    knn_param_distributions = {
        'n_neighbors': (3, 15),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    knn_hyperband = HyperbandSearchCV(KNeighborsClassifier(), knn_param_distributions, max_iter=81, eta=3, cv=5, scoring='accuracy')
    knn_hyperband.fit(X_train_scaled, y_train)
    best_knn = knn_hyperband.best_estimator_
    print(f"Best KNN Parameters: {knn_hyperband.best_params_}")
    print(f"Best KNN Score: {knn_hyperband.best_score_:.4f}")
    
    # Create a function to validate solver-penalty combinations
    def get_valid_solver_penalty_combos():
        return [
            {'penalty': 'l1', 'solver': 'liblinear'},
            {'penalty': 'l2', 'solver': 'lbfgs'},
            {'penalty': 'l2', 'solver': 'newton-cg'},
            {'penalty': 'l2', 'solver': 'sag'},
            {'penalty': 'elasticnet', 'solver': 'saga'},
            {'penalty': None, 'solver': 'lbfgs'}
        ]
    
    # Sample valid combinations
    valid_combos = get_valid_solver_penalty_combos()
    sampled_combo = random.choice(valid_combos)
    
    # Tune the final estimator (Logistic Regression)
    final_param_distributions = {
        'final_estimator__C': (0.001, 100),
        'final_estimator__penalty': ['l2', None],  # Simplified to avoid conflicts
        'final_estimator__solver': ['lbfgs', 'newton-cg', 'sag']  # Compatible solvers
    }
    
    # Create new stacked model with tuned base models
    tuned_stacked_model = StackingClassifier(
        estimators=[('Random Forest', best_rf), ('K-Nearest Neighbors', best_knn)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_jobs=-1
    )
    
    print("\nTuning final estimator for Stacked RF+KNN using Hyperband...")
    final_hyperband = HyperbandSearchCV(tuned_stacked_model, final_param_distributions, 
                                      max_iter=81, eta=3, cv=5, scoring='accuracy')
    final_hyperband.fit(X_train_scaled, y_train)
    best_model = final_hyperband.best_estimator_
    print(f"Best Parameters for final estimator: {final_hyperband.best_params_}")
    print(f"Best Stacked Model Score: {final_hyperband.best_score_:.4f}")

elif best_model_name == "Gradient Boosting":
    param_distributions = {
        'n_estimators': (50, 300),
        'learning_rate': (0.001, 0.3),
        'max_depth': (3, 9),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4),
        'max_features': ['sqrt', 'log2', None]
    }
    hyperband = HyperbandSearchCV(best_model, param_distributions, max_iter=81, eta=3, cv=5, scoring='accuracy', random_state=42)
    hyperband.fit(X_train_scaled, y_train)
    best_model = hyperband.best_estimator_
    print(f"Best Parameters for {best_model_name}: {hyperband.best_params_}")
    print(f"Best Cross-Validation Score: {hyperband.best_score_:.4f}")

# Evaluate the tuned or selected model
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Accuracy of {best_model_name} on Test Data: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f"{best_model_name} - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve visualization
plt.figure(figsize=(8, 6))
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
else:  # For models like SVC that need probability=True
    y_prob = best_model.decision_function(X_test_scaled)
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())  # Scale to [0,1]
    
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'{best_model_name} - Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Feature Importance visualization (for models that support it)
if hasattr(best_model, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    importances = best_model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    
    plt.bar(range(X.shape[1]), importances[indices], align='center', color='skyblue')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.title(f'{best_model_name} - Feature Importance')
    plt.tight_layout()
    plt.show()
elif hasattr(best_model, 'coef_'):
    plt.figure(figsize=(10, 6))
    coef = best_model.coef_[0]
    feature_names = X.columns
    indices = np.argsort(np.abs(coef))[::-1]
    
    plt.bar(range(X.shape[1]), coef[indices], align='center', color='salmon')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title(f'{best_model_name} - Feature Coefficients')
    plt.tight_layout()
    plt.show()

# Save the best model to a file
filename = f'heart-disease-prediction-{best_model_name.lower().replace(" ", "-")}-model.pkl'
pickle.dump(best_model, open(filename, 'wb'))
print(f"\nBest model saved as '{filename}'.")