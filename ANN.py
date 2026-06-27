import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)


df = pd.read_csv('alzheimers_disease_data.csv')

print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFeature Names:")
print(df.columns.tolist())
print("\nData Types:")
print(df.dtypes.to_string())
print("\nMissing Values Per Column:")
print(df.isnull().sum().to_string())
print(f"\nDuplicate Records: {df.duplicated().sum()}")
print("\nStatistical Summary:")
print(df.describe().T.to_string())
print("\nTarget Variable (Diagnosis) Distribution:")
vc = df['Diagnosis'].value_counts()
print(vc)
print(f"Class Balance → {df['Diagnosis'].mean()*100:.1f}% Positive (Alzheimer's)")

df_eda = df.drop(columns=['PatientID', 'DoctorInCharge'])

# Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['Diagnosis'].value_counts()
bars = ax.bar(["No Alzheimer's (0)", "Alzheimer's (1)"], counts.values,
              color=['#2196F3', '#F44336'], edgecolor='black', width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10, str(val),
            ha='center', fontsize=11, fontweight='bold')
ax.set_title("Target Class Distribution", fontsize=13, fontweight='bold')
ax.set_ylabel("Number of Patients")
ax.set_ylim(0, max(counts.values) * 1.2)
plt.tight_layout()
plt.savefig('plots/01_class_distribution.png', dpi=120)
plt.close()

# Histograms – continuous features
cont_cols = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
             'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
             'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
             'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL']

fig, axes = plt.subplots(3, 5, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(cont_cols):
    axes[i].hist(df_eda[col], bins=30, color='#5C85D6', edgecolor='black', alpha=0.8)
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].set_xlabel('Value', fontsize=7)
    axes[i].set_ylabel('Frequency', fontsize=7)
plt.suptitle("Histograms – Continuous Features", fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plots/02_histograms.png', dpi=120, bbox_inches='tight')
plt.close()

# Boxplots: key features vs diagnosis
key_cont = ['Age', 'BMI', 'MMSE', 'FunctionalAssessment',
            'ADL', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(key_cont):
    sns.boxplot(data=df_eda, x='Diagnosis', y=col,
                palette=['#2196F3', '#F44336'], ax=axes[i])
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Diagnosis (0=No, 1=Yes)', fontsize=8)
plt.suptitle("Boxplots: Key Features vs Alzheimer's Diagnosis", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/03_boxplots.png', dpi=120)
plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(18, 14))
corr = df_eda.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, ax=ax, annot_kws={'size': 6})
ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/04_correlation_heatmap.png', dpi=110)
plt.close()

# Feature importance (correlation with target)
target_corr = df_eda.corr()['Diagnosis'].drop('Diagnosis').abs().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 10))
colors = ['#F44336' if v > 0.1 else '#2196F3' for v in target_corr.values]
ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor='black')
ax.set_xlabel('|Pearson Correlation with Diagnosis|', fontsize=11)
ax.set_title("Feature Importance (Correlation-based)", fontsize=13, fontweight='bold')
ax.axvline(0.1, color='orange', linestyle='--', linewidth=2, label='0.1 threshold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/05_feature_importance.png', dpi=120)
plt.close()

print("EDA complete — 5 plots saved to plots/")

X = df.drop(columns=['PatientID', 'DoctorInCharge', 'Diagnosis','Gender', 'Ethnicity', 'BMI', 'Smoking', 'DietQuality'])
y = df['Diagnosis']

print(f"\nFeature matrix shape : {X.shape}")
print(f"Target vector shape  : {y.shape}")
print(f"\nFeatures used ({X.shape[1]}):")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test  set: {X_test.shape[0]} samples")
print(f"Train class balance: {y_train.mean()*100:.1f}% positive")
print(f"Test  class balance: {y_test.mean()*100:.1f}% positive")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

n_features = X_train_sc.shape[1]

def build_ann(layers=[128, 64, 32], dropout=0.3, lr=0.001, activation='elu'):
    """
    Build a configurable ANN for binary classification.

    Parameters
    ----------
    layers     : list of ints — number of neurons in each hidden layer
    dropout    : float        — dropout rate applied after each hidden layer
    lr         : float        — learning rate for Adam optimiser
    activation : str          — hidden-layer activation function

    Returns
    -------
    Compiled Keras Sequential model
    """
    model = Sequential()
    model.add(Dense(layers[0], input_dim=n_features, activation=activation))
    model.add(Dropout(dropout))

    for units in layers[1:]:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


model = build_ann(layers=[128, 64, 32], dropout=0.3, lr=0.001, activation='elu')
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_sc, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Learning curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='Train Loss', color='blue',  linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss',   color='red',   linewidth=2)
axes[0].set_title('Loss Curve', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Binary Cross-Entropy Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'],     label='Train Acc', color='blue', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val Acc',   color='red',  linewidth=2)
axes[1].set_title('Accuracy Curve', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("ANN Learning Curves (Best Model)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/06_learning_curves.png', dpi=120)
plt.close()


y_pred_prob = model.predict(X_test_sc).flatten()
y_pred      = (y_pred_prob >= 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n{'Metric':<15} {'Score':>8}")
print("-" * 25)
print(f"{'Accuracy':<15} {acc:>8.4f}")
print(f"{'Precision':<15} {prec:>8.4f}")
print(f"{'Recall':<15} {rec:>8.4f}")
print(f"{'F1 Score':<15} {f1:>8.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["No Alzheimer's", "Alzheimer's"]))

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred No', 'Pred Yes'],
            yticklabels=['True No', 'True Yes'])
ax.set_title("Confusion Matrix", fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.savefig('plots/07_confusion_matrix.png', dpi=120)
plt.close()


final_train_acc = history.history['accuracy'][-1]
final_val_acc   = history.history['val_accuracy'][-1]
gap             = final_train_acc - final_val_acc

print(f"\nFinal Training Accuracy  : {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Gap (Train - Val)        : {gap:.4f}")

if gap > 0.10:
    print("WARNING: OVERFITTING detected (gap > 10%)")
    print("Suggestions: increase Dropout, reduce model size, add L2 regularisation")
elif final_train_acc < 0.75:
    print("WARNING: UNDERFITTING detected (training accuracy too low)")
    print("Suggestions: increase neurons, reduce Dropout, train longer")
else:
    print("OK: Model is well generalised — train and val accuracies are close")

configs = [
    {'name': 'Baseline',          'layers': [128, 64, 32],      'dropout': 0.3, 'lr': 0.001,  'batch': 32, 'activation': 'relu'},
    {'name': 'Deeper (4 layers)', 'layers': [256, 128, 64, 32], 'dropout': 0.3, 'lr': 0.001,  'batch': 32, 'activation': 'relu'},
    {'name': 'Smaller LR',        'layers': [128, 64, 32],      'dropout': 0.3, 'lr': 0.0001, 'batch': 32, 'activation': 'relu'},
    {'name': 'High Dropout',      'layers': [128, 64, 32],      'dropout': 0.5, 'lr': 0.001,  'batch': 32, 'activation': 'relu'},
    {'name': 'Large Batch',       'layers': [128, 64, 32],      'dropout': 0.3, 'lr': 0.001,  'batch': 64, 'activation': 'relu'},
    {'name': 'ELU Activation',    'layers': [128, 64, 32],      'dropout': 0.3, 'lr': 0.001,  'batch': 32, 'activation': 'elu'},
    {'name': 'Wide+Low Dropout',  'layers': [256, 128],         'dropout': 0.2, 'lr': 0.001,  'batch': 32, 'activation': 'relu'},
]

results = []
for cfg in configs:
    m = build_ann(layers=cfg['layers'], dropout=cfg['dropout'],
                  lr=cfg['lr'], activation=cfg['activation'])
    m.fit(
        X_train_sc, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=cfg['batch'],
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )
    yp = (m.predict(X_test_sc, verbose=0).flatten() >= 0.5).astype(int)
    results.append({
        'Model':      cfg['name'],
        'Layers':     str(cfg['layers']),
        'Dropout':    cfg['dropout'],
        'LR':         cfg['lr'],
        'Batch':      cfg['batch'],
        'Activation': cfg['activation'],
        'Accuracy':   round(accuracy_score(y_test, yp), 4),
        'Precision':  round(precision_score(y_test, yp), 4),
        'Recall':     round(recall_score(y_test, yp), 4),
        'F1':         round(f1_score(y_test, yp), 4),
    })
    print(f"  {cfg['name']:25s} | Acc={results[-1]['Accuracy']:.4f} | F1={results[-1]['F1']:.4f}")

results_df = pd.DataFrame(results)
best_idx   = results_df['F1'].idxmax()
best_row   = results_df.loc[best_idx]

print(f"\n{'='*80}")
print("HYPERPARAMETER COMPARISON TABLE")
print(f"{'='*80}")
print(results_df[['Model', 'Layers', 'Dropout', 'LR', 'Batch', 'Activation',
                   'Accuracy', 'Precision', 'Recall', 'F1']].to_string(index=False))
print(f"\nBEST MODEL: {best_row['Model']} (F1={best_row['F1']})")

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(results_df))
w = 0.2
ax.bar(x - w*1.5, results_df['Accuracy'],  w, label='Accuracy',  color='#2196F3')
ax.bar(x - w*0.5, results_df['Precision'], w, label='Precision', color='#4CAF50')
ax.bar(x + w*0.5, results_df['Recall'],    w, label='Recall',    color='#FF9800')
ax.bar(x + w*1.5, results_df['F1'],        w, label='F1 Score',  color='#F44336')
ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=25, ha='right', fontsize=8)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('Score')
ax.set_title('Hyperparameter Tuning – Model Comparison', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('plots/08_hyperparameter_comparison.png', dpi=120)
plt.close()


model.save('alzheimers_ann_model.keras')
joblib.dump(scaler, 'scaler.pkl')

print("Model saved → alzheimers_ann_model.keras")
print("Scaler saved → scaler.pkl")

from tensorflow.keras.models import load_model

loaded_model  = load_model('alzheimers_ann_model.keras')
loaded_scaler = joblib.load('scaler.pkl')

new_patient = pd.DataFrame([{
    'Age': 78, 'EducationLevel': 2,
    'AlcoholConsumption': 5.2,
    'PhysicalActivity': 3.1, 'SleepQuality': 6.5,
    'FamilyHistoryAlzheimers': 1, 'CardiovascularDisease': 0,
    'Diabetes': 0, 'Depression': 1, 'HeadInjury': 0, 'Hypertension': 1,
    'SystolicBP': 140, 'DiastolicBP': 88, 'CholesterolTotal': 210.0,
    'CholesterolLDL': 130.0, 'CholesterolHDL': 55.0,
    'CholesterolTriglycerides': 180.0, 'MMSE': 22.0,
    'FunctionalAssessment': 4.0, 'MemoryComplaints': 1,
    'BehavioralProblems': 1, 'ADL': 3.5, 'Confusion': 1,
    'Disorientation': 1, 'PersonalityChanges': 0,
    'DifficultyCompletingTasks': 1, 'Forgetfulness': 1
}])

new_patient_sc = loaded_scaler.transform(new_patient)
probability    = loaded_model.predict(new_patient_sc, verbose=0)[0][0]
prediction     = "Alzheimer's Detected" if probability >= 0.5 else "No Alzheimer's"

print(f"\nNew Patient Prediction:")
print(f"  Probability : {probability:.4f}")
print(f"  Diagnosis   : {prediction}")

print("\nPipeline complete — all outputs saved to plots/")