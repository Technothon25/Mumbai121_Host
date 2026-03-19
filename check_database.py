"""
Database Diagnostic Script
===========================
Run this to see what's actually in your database and why rankings might be off
"""

import pymongo
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client.Mumbai121

print("="*70)
print("🔍 MUMBAI121 DATABASE DIAGNOSTIC")
print("="*70)

# Check Freshers collection
print("\n📊 FRESHERS COLLECTION:")
print("-"*70)

freshers = list(db.Freshers.find().limit(5))
print(f"Total freshers: {db.Freshers.count_documents({})}")

if freshers:
    print(f"\n🔎 Sample Fresher Record:")
    sample = freshers[0]
    print(f"  Name: {sample.get('name')}")
    print(f"  Job Preference: {sample.get('jobPreference')}")
    
    # Check all possible skill field names
    print(f"\n  Checking skill fields:")
    for field in ['skills', 'skill', 'keySkills', 'KeySkills', 'Skills']:
        value = sample.get(field)
        if value:
            print(f"    ✅ {field}: {str(value)[:80]}")
        else:
            print(f"    ❌ {field}: (empty)")
    
    print(f"\n  All fields in this record:")
    for key in sorted(sample.keys()):
        if key != '_id':
            value = str(sample.get(key))[:50]
            print(f"    - {key}: {value}")

# Check IT Engineer freshers specifically
print("\n\n💻 IT ENGINEER FRESHERS:")
print("-"*70)

it_freshers = list(db.Freshers.find({"jobPreference": "IT Engineer"}).limit(10))
print(f"Total IT Engineers: {db.Freshers.count_documents({'jobPreference': 'IT Engineer'})}")

if it_freshers:
    print(f"\n📋 First 10 IT Engineers:")
    for i, fresher in enumerate(it_freshers, 1):
        name = fresher.get('name', 'Unknown')
        skills = (
            fresher.get('keySkills') or 
            fresher.get('skills') or 
            fresher.get('skill') or 
            '(no skills found)'
        )
        print(f"\n  {i}. {name}")
        print(f"      Skills: {str(skills)[:100]}")

# Check Requirements collection
print("\n\n📄 REQUIREMENTS COLLECTION:")
print("-"*70)

requirements = list(db.Requirements.find().limit(3))
print(f"Total requirements: {db.Requirements.count_documents({})}")

if requirements:
    print(f"\n📋 Sample Requirement:")
    sample = requirements[0]
    print(f"  Company: {sample.get('company')}")
    print(f"  Job: {sample.get('jobPreference')}")
    print(f"  Description: {sample.get('workDescription', '')[:100]}")
    print(f"  Processed: {sample.get('processed')}")
    print(f"  AI Processed: {sample.get('aiProcessed')}")
    
    sent_count = len(sample.get('sentFresherIds', []))
    print(f"  Candidates sent: {sent_count}")

# Check if there are Java candidates
print("\n\n☕ JAVA CANDIDATES:")
print("-"*70)

java_keywords = ['java', 'Java', 'JAVA', 'spring', 'Spring', 'mysql', 'MySQL']

java_candidates = []
all_freshers = db.Freshers.find({"jobPreference": "IT Engineer"})

for fresher in all_freshers:
    skills = (
        fresher.get('keySkills') or 
        fresher.get('skills') or 
        fresher.get('skill') or 
        ''
    )
    
    skills_str = str(skills).lower()
    
    # Check if has Java
    if 'java' in skills_str:
        has_mysql = 'mysql' in skills_str or 'sql' in skills_str
        java_candidates.append({
            'name': fresher.get('name'),
            'skills': str(skills)[:100],
            'has_mysql': has_mysql
        })

print(f"Found {len(java_candidates)} candidates with Java skills:")

for i, cand in enumerate(java_candidates[:10], 1):
    mysql_icon = "✅" if cand['has_mysql'] else "❌"
    print(f"\n  {i}. {cand['name']} {mysql_icon}")
    print(f"      {cand['skills']}")

# Analysis
print("\n\n🎯 ANALYSIS:")
print("="*70)

total_it = db.Freshers.count_documents({'jobPreference': 'IT Engineer'})
print(f"Total IT Engineer freshers: {total_it}")
print(f"Java candidates found: {len(java_candidates)}")
print(f"Java + MySQL candidates: {sum(1 for c in java_candidates if c['has_mysql'])}")

if len(java_candidates) == 0:
    print("\n⚠️  WARNING: No Java candidates found!")
    print("   Possible reasons:")
    print("   1. Skills field is named differently")
    print("   2. Skills are stored in a different format")
    print("   3. No Java candidates in database")
elif len(java_candidates) < 10:
    print(f"\n⚠️  WARNING: Only {len(java_candidates)} Java candidates available")
    print("   For a Java+MySQL job, you need more Java developers!")

# Check field name consistency
print("\n\n🔧 FIELD NAME CHECK:")
print("-"*70)

field_usage = {
    'keySkills': 0,
    'skills': 0,
    'skill': 0,
    'Skills': 0,
    'KeySkills': 0
}

sample_freshers = db.Freshers.find().limit(100)
for fresher in sample_freshers:
    for field in field_usage.keys():
        if field in fresher and fresher[field]:
            field_usage[field] += 1

print("Field usage in first 100 records:")
for field, count in field_usage.items():
    if count > 0:
        print(f"  ✅ '{field}': used in {count} records")
    else:
        print(f"  ❌ '{field}': not used")

print("\n" + "="*70)
print("Diagnostic complete!")
print("="*70)
