"""
Generate Training Data for Mumbai121 ML Model
==============================================
This creates realistic training data based on Mumbai job market
"""

import json
import random

def generate_comprehensive_training_data():
    """
    Generate 100+ training examples covering all job categories in Mumbai121
    """
    
    training_data = []
    
    # ========================================================================
    # IT ENGINEER - FULLSTACK DEVELOPERS
    # ========================================================================
    
    fullstack_jobs = [
        "Need fullstack developer with React and Node.js for e-commerce platform",
        "Looking for MERN stack developer with MongoDB experience",
        "Fullstack engineer needed - React, Node.js, Express, PostgreSQL",
        "Web developer with React frontend and Node.js backend skills required",
    ]
    
    fullstack_perfect = [
        "React, Node.js, MongoDB, Express, JavaScript, Redux, REST API, Git",
        "MERN Stack, React, Node.js, MongoDB, Express, Redux, TypeScript",
        "React, Node.js, PostgreSQL, Express, REST API, Docker, AWS",
        "React, Vue.js, Node.js, Express, MySQL, GraphQL, Docker",
    ]
    
    fullstack_good = [
        "React, Angular, Node.js, Express, JavaScript, CSS, HTML",
        "Vue.js, Node.js, Express, MongoDB, REST API, Git",
        "React, TypeScript, Node.js, MySQL, Bootstrap, jQuery",
        "Angular, Node.js, PostgreSQL, Express, TypeScript",
    ]
    
    fullstack_okay = [
        "React, JavaScript, HTML, CSS, Bootstrap, jQuery",
        "Node.js, Express, MongoDB, REST API, Postman",
        "Angular, TypeScript, RxJS, NgRx, Material UI",
        "Vue.js, JavaScript, HTML5, CSS3, Vuex",
    ]
    
    fullstack_weak = [
        "Python, Django, REST API, PostgreSQL",
        "Java, Spring Boot, MySQL, Hibernate",
        "PHP, Laravel, MySQL, Apache",
        "C#, .NET, SQL Server, Entity Framework",
    ]
    
    for job in fullstack_jobs:
        for skills in fullstack_perfect:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.90, 0.98)
            })
        
        for skills in fullstack_good:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.75, 0.88)
            })
        
        for skills in fullstack_okay:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.55, 0.72)
            })
        
        for skills in fullstack_weak:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.25, 0.45)
            })
    
    # ========================================================================
    # IT ENGINEER - BACKEND DEVELOPERS
    # ========================================================================
    
    backend_jobs = [
        "Backend Python developer needed with Django and REST API experience",
        "Python developer for fintech startup - Django, PostgreSQL, Redis",
        "Backend engineer needed - Python, FastAPI, microservices",
        "Server-side developer - Python, Django, Docker, AWS required",
    ]
    
    backend_perfect = [
        "Python, Django, REST API, PostgreSQL, Docker, Redis, Celery",
        "Python, Django, DRF, PostgreSQL, AWS, Docker, CI/CD",
        "Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes",
        "Python, Flask, SQLAlchemy, PostgreSQL, Redis, RabbitMQ",
    ]
    
    backend_good = [
        "Python, Django, MySQL, REST API, Git, Linux",
        "Python, FastAPI, MongoDB, Docker, pytest",
        "Python, Flask, PostgreSQL, SQLAlchemy, Redis",
        "Python, Django, DRF, MySQL, Celery, Redis",
    ]
    
    backend_okay = [
        "Python, Flask, SQLite, basic REST API",
        "Python, Django, basic database, Git",
        "Java, Spring Boot, MySQL, REST API, Maven",
        "Node.js, Express, MongoDB, REST API",
    ]
    
    backend_weak = [
        "React, JavaScript, HTML, CSS, Bootstrap",
        "Angular, TypeScript, RxJS, Material UI",
        "Java, Android, Kotlin, XML",
        "PHP, WordPress, MySQL, Apache",
    ]
    
    for job in backend_jobs:
        for skills in backend_perfect:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.92, 0.99)
            })
        
        for skills in backend_good:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.78, 0.90)
            })
        
        for skills in backend_okay:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.50, 0.70)
            })
        
        for skills in backend_weak:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.15, 0.35)
            })
    
    # ========================================================================
    # IT ENGINEER - DATA SCIENCE
    # ========================================================================
    
    ds_jobs = [
        "Data scientist needed for ML projects - Python, scikit-learn, pandas",
        "ML engineer required - Python, TensorFlow, deep learning experience",
        "Data analyst position - Python, pandas, data visualization, SQL",
        "AI/ML developer - Python, machine learning, NLP, computer vision",
    ]
    
    ds_perfect = [
        "Python, scikit-learn, pandas, numpy, machine learning, statistics, Tableau",
        "Python, TensorFlow, Keras, deep learning, NLP, computer vision, pandas",
        "Python, PyTorch, scikit-learn, pandas, data science, ML algorithms",
        "Python, machine learning, pandas, numpy, statistics, data visualization, SQL",
    ]
    
    ds_good = [
        "Python, pandas, numpy, matplotlib, data analysis, SQL",
        "Python, scikit-learn, pandas, statistics, data visualization",
        "R, statistics, data analysis, ggplot2, dplyr, SQL",
        "Python, pandas, seaborn, matplotlib, Jupyter, SQL",
    ]
    
    for job in ds_jobs:
        for skills in ds_perfect:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.90, 0.98)
            })
        
        for skills in ds_good:
            training_data.append({
                'job_description': job,
                'job_category': 'IT Engineer',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.70, 0.85)
            })
    
    # ========================================================================
    # SALES AND MARKETING - B2B SALES
    # ========================================================================
    
    sales_jobs = [
        "B2B sales executive needed for pharma industry with client relations experience",
        "Field sales representative - FMCG sector, territory management required",
        "Sales manager for IT products - B2B sales, lead generation, CRM",
        "Business development executive - B2B sales, cold calling, negotiation",
    ]
    
    sales_perfect = [
        "B2B sales, pharma industry, client relations, negotiation, territory management, CRM",
        "field sales, FMCG, territory management, client acquisition, sales targets",
        "B2B sales, IT products, lead generation, CRM, sales presentations, negotiation",
        "business development, B2B sales, cold calling, lead generation, negotiation, closing",
    ]
    
    sales_good = [
        "B2B sales, healthcare, account management, client relations",
        "sales, FMCG, retail, customer service, targets",
        "inside sales, lead generation, CRM, email marketing",
        "business development, sales, presentations, client meetings",
    ]
    
    sales_okay = [
        "retail sales, customer service, POS systems, cash handling",
        "telecalling, lead generation, basic CRM",
        "customer support, complaint handling, customer satisfaction",
        "marketing, social media, content creation",
    ]
    
    for job in sales_jobs:
        for skills in sales_perfect:
            training_data.append({
                'job_description': job,
                'job_category': 'Sales And Marketing',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.88, 0.97)
            })
        
        for skills in sales_good:
            training_data.append({
                'job_description': job,
                'job_category': 'Sales And Marketing',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.68, 0.83)
            })
        
        for skills in sales_okay:
            training_data.append({
                'job_description': job,
                'job_category': 'Sales And Marketing',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.35, 0.58)
            })
    
    # ========================================================================
    # SALES AND MARKETING - DIGITAL MARKETING
    # ========================================================================
    
    digital_jobs = [
        "Digital marketing specialist - SEO, Google Ads, social media marketing",
        "Performance marketer needed - Google Ads, Facebook Ads, analytics",
        "Content marketer - SEO, content writing, social media, WordPress",
        "Social media manager - Instagram, Facebook, content creation, analytics",
    ]
    
    digital_perfect = [
        "SEO, Google Ads, Facebook Ads, social media marketing, analytics, content marketing",
        "Google Ads, Facebook Ads, Instagram Ads, PPC, conversion optimization, analytics",
        "SEO, content writing, WordPress, social media, keyword research, Google Analytics",
        "social media management, Instagram, Facebook, content creation, Canva, analytics",
    ]
    
    digital_good = [
        "SEO, content writing, basic WordPress, social media",
        "social media marketing, Facebook, Instagram, content creation",
        "Google Ads, basic analytics, email marketing",
        "content writing, blogging, SEO basics, social media",
    ]
    
    for job in digital_jobs:
        for skills in digital_perfect:
            training_data.append({
                'job_description': job,
                'job_category': 'Sales And Marketing',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.90, 0.98)
            })
        
        for skills in digital_good:
            training_data.append({
                'job_description': job,
                'job_category': 'Sales And Marketing',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.65, 0.82)
            })
    
    # ========================================================================
    # ACCOUNTING AND FINANCE
    # ========================================================================
    
    accounting_jobs = [
        "Junior accountant needed - Tally ERP, GST filing, accounting principles",
        "Accounts executive - Tally, bookkeeping, GST, accounts payable/receivable",
        "Finance intern - Excel, Tally, basic accounting, financial statements",
        "Accountant position - Tally ERP 9, GST, TDS, income tax, bookkeeping",
    ]
    
    accounting_perfect = [
        "Tally ERP 9, GST filing, accounting principles, Excel, bookkeeping, financial statements",
        "Tally ERP, accounts payable, accounts receivable, GST, TDS, reconciliation",
        "Tally, GST, income tax, TDS, accounting, Excel, QuickBooks",
        "accounting, Tally ERP, bookkeeping, GST returns, financial reporting, Excel",
    ]
    
    accounting_good = [
        "Tally ERP, basic accounting, GST, Excel",
        "accounting principles, bookkeeping, Tally, basic Excel",
        "QuickBooks, accounts payable, Excel, basic GST",
        "Excel, accounting, data entry, basic Tally",
    ]
    
    accounting_weak = [
        "MS Office, data entry, typing, basic Excel",
        "customer service, communication, teamwork",
        "sales, marketing, client relations",
    ]
    
    for job in accounting_jobs:
        for skills in accounting_perfect:
            training_data.append({
                'job_description': job,
                'job_category': 'Accounting And Finance',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.90, 0.98)
            })
        
        for skills in accounting_good:
            training_data.append({
                'job_description': job,
                'job_category': 'Accounting And Finance',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.68, 0.85)
            })
        
        for skills in accounting_weak:
            training_data.append({
                'job_description': job,
                'job_category': 'Accounting And Finance',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.20, 0.42)
            })
    
    # ========================================================================
    # HUMAN RESOURCES
    # ========================================================================
    
    hr_jobs = [
        "HR recruiter for tech hiring - LinkedIn sourcing, screening, ATS",
        "HR generalist needed - recruitment, onboarding, employee relations, HRIS",
        "Talent acquisition specialist - IT recruitment, LinkedIn, Boolean search",
        "HR coordinator - payroll, attendance, employee engagement, HR policies",
    ]
    
    hr_perfect = [
        "technical recruiting, LinkedIn sourcing, Boolean search, ATS, candidate screening, IT hiring",
        "HR generalist, recruitment, onboarding, employee relations, HRIS, HR policies",
        "talent acquisition, IT recruitment, LinkedIn, sourcing, screening, offer negotiation",
        "HR coordination, payroll, attendance management, employee engagement, HR software",
    ]
    
    hr_good = [
        "recruitment, LinkedIn, screening, HR basics",
        "HR, employee relations, onboarding, basic payroll",
        "sourcing, recruiting, ATS, communication",
        "payroll, attendance, Excel, HR administration",
    ]
    
    for job in hr_jobs:
        for skills in hr_perfect:
            training_data.append({
                'job_description': job,
                'job_category': 'Human Resources',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.88, 0.97)
            })
        
        for skills in hr_good:
            training_data.append({
                'job_description': job,
                'job_category': 'Human Resources',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.65, 0.82)
            })
    
    # ========================================================================
    # CUSTOMER SERVICE
    # ========================================================================
    
    cs_jobs = [
        "Customer support executive for SaaS product - technical support, ticketing",
        "Call center agent - customer service, CRM, complaint resolution",
        "Customer success manager - client onboarding, retention, support",
        "Technical support engineer - troubleshooting, ticketing, customer service",
    ]
    
    cs_perfect = [
        "technical support, SaaS products, Zendesk, troubleshooting, customer service, ticketing",
        "call center, customer service, CRM, complaint resolution, communication, patience",
        "customer success, client onboarding, retention, account management, SaaS",
        "technical support, troubleshooting, ticketing systems, remote support, customer service",
    ]
    
    cs_good = [
        "customer service, communication, CRM, problem-solving",
        "call handling, complaint resolution, basic CRM",
        "customer support, email support, ticketing",
        "technical support basics, troubleshooting, customer service",
    ]
    
    for job in cs_jobs:
        for skills in cs_perfect:
            training_data.append({
                'job_description': job,
                'job_category': 'Customer Service',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.88, 0.96)
            })
        
        for skills in cs_good:
            training_data.append({
                'job_description': job,
                'job_category': 'Customer Service',
                'candidate_skills': skills,
                'relevance_score': random.uniform(0.62, 0.80)
            })
    
    print(f"\n✅ Generated {len(training_data)} training examples")
    print(f"\n📊 Distribution by category:")
    
    categories = {}
    for item in training_data:
        cat = item['job_category']
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in categories.items():
        print(f"  - {cat}: {count} examples")
    
    print(f"\n📈 Score distribution:")
    scores = [item['relevance_score'] for item in training_data]
    print(f"  - Perfect (0.9-1.0): {sum(1 for s in scores if s >= 0.9)}")
    print(f"  - Good (0.7-0.9): {sum(1 for s in scores if 0.7 <= s < 0.9)}")
    print(f"  - Okay (0.5-0.7): {sum(1 for s in scores if 0.5 <= s < 0.7)}")
    print(f"  - Weak (0.0-0.5): {sum(1 for s in scores if s < 0.5)}")
    
    return training_data


if __name__ == "__main__":
    print("="*70)
    print("🎯 GENERATING TRAINING DATA FOR MUMBAI121 ML MODEL")
    print("="*70)
    
    training_data = generate_comprehensive_training_data()
    
    # Save to file
    with open('mumbai121_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved to: mumbai121_training_data.json")
    print(f"\n🎓 Next step: Run 'python train_ml_model.py' to train your model!")
    print("="*70)
