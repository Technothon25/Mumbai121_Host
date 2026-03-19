"""
Train ML Model for Mumbai121 Candidate Ranking
===============================================
This script trains a Random Forest model on your training data
"""

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

print("="*70)
print("🎓 MUMBAI121 ML MODEL TRAINER")
print("="*70)

# ============================================================================
# STEP 1: LOAD TRAINING DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 1: LOADING TRAINING DATA")
print("="*70)

training_file = './mumbai121_training_data_real.json'

if not os.path.exists(training_file):
    print(f"❌ {training_file} not found!")
    print("   Run 'python generate_training_data.py' first")
    exit(1)

with open(training_file, 'r', encoding='utf-8') as f:
    training_data = json.load(f)

print(f"✅ Loaded {len(training_data)} training examples")

# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 2: PREPARING DATA FOR TRAINING")
print("="*70)

# Create DataFrame
df = pd.DataFrame(training_data)

# Combine job description and candidate skills into one text
df['combined_text'] = df['job_description'] + " " + df['candidate_skills']

print(f"✅ Prepared {len(df)} examples")
print(f"   Features: job description + candidate skills")
print(f"   Target: relevance score (0-1)")

# ============================================================================
# STEP 3: FEATURE EXTRACTION (TEXT TO NUMBERS)
# ============================================================================

print("\n" + "="*70)
print("STEP 3: CONVERTING TEXT TO NUMBERS (TF-IDF)")
print("="*70)

print("🔄 Extracting features using TF-IDF...")

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=200,       # Use top 200 most important words
    ngram_range=(1, 2),     # Use single words and word pairs
    min_df=2,               # Word must appear in at least 2 documents
    max_df=0.8              # Ignore words that appear in >80% of docs
)

# Convert text to numbers
X = vectorizer.fit_transform(df['combined_text']).toarray()
y = df['relevance_score'].values

print(f"✅ Created {X.shape[1]} numerical features from text")
print(f"   Each example is now a vector of {X.shape[1]} numbers")
print(f"   Most important words: {', '.join(vectorizer.get_feature_names_out()[:10])}")

# ============================================================================
# STEP 4: SPLIT DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 4: SPLITTING DATA (TRAIN/TEST)")
print("="*70)

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Training set: {len(X_train)} examples (80%)")
print(f"📊 Test set: {len(X_test)} examples (20%)")

# ============================================================================
# STEP 5: TRAIN THE MODEL
# ============================================================================

print("\n" + "="*70)
print("STEP 5: TRAINING RANDOM FOREST MODEL")
print("="*70)

print("🌲 Training Random Forest with 100 trees...")

model = RandomForestRegressor(
    n_estimators=100,       # 100 decision trees
    max_depth=15,           # Maximum depth of each tree
    min_samples_split=5,    # Minimum samples to split a node
    min_samples_leaf=2,     # Minimum samples in leaf node
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)

# Train the model
model.fit(X_train, y_train)

print("✅ Training complete!")

# ============================================================================
# STEP 6: EVALUATE PERFORMANCE
# ============================================================================

print("\n" + "="*70)
print("STEP 6: EVALUATING MODEL PERFORMANCE")
print("="*70)

# Predictions on training set
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Predictions on test set
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"\n📊 TRAINING SET PERFORMANCE:")
print(f"   R² Score: {train_r2:.3f}")
print(f"   Mean Absolute Error: {train_mae:.3f}")

print(f"\n📊 TEST SET PERFORMANCE (UNSEEN DATA):")
print(f"   R² Score: {test_r2:.3f}")
print(f"     → How well predictions match reality")
print(f"     → 1.0 = perfect, 0.8+ = excellent, 0.7+ = good")
print(f"\n   Mean Absolute Error: {test_mae:.3f}")
print(f"     → Average error in predictions")
print(f"     → Lower is better!")
print(f"\n   Root Mean Squared Error: {test_rmse:.3f}")

# Performance assessment
if test_r2 >= 0.80:
    print(f"\n✅ EXCELLENT! Your model performs very well!")
elif test_r2 >= 0.70:
    print(f"\n✅ GOOD! Model is reliable for production use")
elif test_r2 >= 0.60:
    print(f"\n✅ DECENT! Model will work, but could be better")
else:
    print(f"\n⚠️  Model needs more training data to improve")

# Check for overfitting
if train_r2 - test_r2 > 0.15:
    print(f"\n⚠️  Warning: Possible overfitting (train score much higher than test)")
else:
    print(f"\n✅ Model generalizes well (no overfitting)")

# ============================================================================
# STEP 7: TEST WITH REAL EXAMPLES
# ============================================================================

print("\n" + "="*70)
print("STEP 7: TESTING WITH REAL-WORLD EXAMPLES")
print("="*70)

test_cases = [
    {
        'job': 'Need fullstack developer with React and Node.js experience',
        'candidates': [
            'React, Node.js, MongoDB, Express, JavaScript, Redux',
            'Python, Django, REST API, PostgreSQL',
            'React, Angular, TypeScript, CSS, HTML',
            'Java, Spring Boot, MySQL, Hibernate'
        ],
        'expected_order': [0, 2, 3, 1]  # Index 0 should rank highest
    },
    {
        'job': 'Backend Python developer with Django needed',
        'candidates': [
            'Python, Django, REST API, PostgreSQL, Docker',
            'Python, Flask, MongoDB, REST API',
            'Java, Spring Boot, MySQL',
            'React, JavaScript, HTML, CSS'
        ],
        'expected_order': [0, 1, 2, 3]
    },
    {
        'job': 'B2B sales executive for pharma industry',
        'candidates': [
            'B2B sales, pharma industry, client relations, territory management',
            'B2B sales, healthcare, account management',
            'retail sales, customer service, POS',
            'digital marketing, SEO, social media'
        ],
        'expected_order': [0, 1, 2, 3]
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"Test Case {i}: {test['job']}")
    print(f"{'-'*70}")
    
    scores = []
    for candidate_skills in test['candidates']:
        # Create combined text
        text = test['job'] + " " + candidate_skills
        # Transform to features
        features = vectorizer.transform([text]).toarray()
        # Predict score
        score = model.predict(features)[0]
        scores.append(score)
    
    # Rank candidates by score
    ranked_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    
    print(f"\n📊 Rankings:")
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"  {rank}. Score: {scores[idx]:.3f} | {test['candidates'][idx][:60]}...")
    
    # Check if ranking is correct
    if ranked_indices[0] == test['expected_order'][0]:
        print(f"\n✅ CORRECT! Top candidate is the best match")
    else:
        print(f"\n❌ Rankings could be better")

# ============================================================================
# STEP 8: FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*70)
print("STEP 8: UNDERSTANDING WHAT THE MODEL LEARNED")
print("="*70)

# Get feature importance
feature_names = vectorizer.get_feature_names_out()
importances = model.feature_importances_

# Sort by importance
indices = np.argsort(importances)[::-1][:20]  # Top 20

print(f"\n📈 Top 20 Most Important Features (Words/Phrases):")
for i, idx in enumerate(indices, 1):
    print(f"  {i:2d}. {feature_names[idx]:30s} → {importances[idx]:.4f}")

# ============================================================================
# STEP 9: SAVE THE MODEL
# ============================================================================

print("\n" + "="*70)
print("STEP 9: SAVING YOUR TRAINED MODEL")
print("="*70)

# Package everything together
model_package = {
    'model': model,
    'vectorizer': vectorizer,
    'training_size': len(training_data),
    'train_r2': train_r2,
    'test_r2': test_r2,
    'test_mae': test_mae,
    'feature_count': X.shape[1],
    'trained_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Save to file
model_filename = 'mumbai121_candidate_ranker.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_package, f)

file_size = os.path.getsize(model_filename) / 1024  # KB

print(f"✅ Model saved to: {model_filename}")
print(f"✅ File size: {file_size:.1f} KB")
print(f"\n📦 Model package includes:")
print(f"   - Trained Random Forest model")
print(f"   - TF-IDF vectorizer")
print(f"   - Training metadata")
print(f"   - Performance metrics")

# ============================================================================
# STEP 10: USAGE INSTRUCTIONS
# ============================================================================

print("\n" + "="*70)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*70)

print(f"\n📊 Final Performance Summary:")
print(f"   Training examples: {len(training_data)}")
print(f"   Features extracted: {X.shape[1]}")
print(f"   Test R² Score: {test_r2:.3f}")
print(f"   Test MAE: {test_mae:.3f}")

print(f"\n🚀 Next Steps:")
print(f"   1. Your model is ready to use!")
print(f"   2. Run 'python app10_with_ml.py' to use it in production")
print(f"   3. The model will automatically rank candidates")

print(f"\n💡 To improve your model over time:")
print(f"   - Collect feedback from companies")
print(f"   - Add more training examples")
print(f"   - Retrain monthly with: python train_ml_model.py")

print(f"\n📈 Expected Performance Growth:")
print(f"   Current ({len(training_data)} examples): R² ~ {test_r2:.2f}")
print(f"   With 500 examples: R² ~ 0.80-0.85")
print(f"   With 1000+ examples: R² ~ 0.85-0.90")

print("\n" + "="*70)
print("Your ML model is ready to rank candidates! 🎊")
print("="*70)
