# ============================================================
#  Mumbai121 – FastAPI Backend
#  Stack: FastAPI + PyMongo (sync) + Gunicorn + Uvicorn workers
# ============================================================

import json
import pickle
import smtplib
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import gridfs
import pymongo
from bson.objectid import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from werkzeug.utils import secure_filename
import os

# ── ENV ───────────────────────────────────────────────────────────────────────
load_dotenv()

MONGO_URI      = os.getenv("MONGO_URI")
EMAIL_ADDRESS  = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# ── MONGODB ───────────────────────────────────────────────────────────────────
client = pymongo.MongoClient(
    MONGO_URI,
    # Connection pool tuned for 4 Gunicorn workers
    maxPoolSize=25,
    minPoolSize=5,
    serverSelectionTimeoutMS=5000,
)
db = client.Mumbai121
fs = gridfs.GridFS(db)

# ── FILE CONFIG ───────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE      = 5 * 1024 * 1024  # 5 MB

# ── ML MODEL ──────────────────────────────────────────────────────────────────
ML_MODEL      = None
ML_VECTORIZER = None

def load_ml_model():
    global ML_MODEL, ML_VECTORIZER
    model_path = "mumbai121_candidate_ranker.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            ML_MODEL      = model_data["model"]
            ML_VECTORIZER = model_data["vectorizer"]
            print("✅ ML model loaded successfully!")
            print(f"   Training size : {model_data.get('training_size', 'Unknown')}")
            print(f"   Test R² Score : {model_data.get('test_r2', 0):.3f}")
            print(f"   Features      : {model_data.get('feature_count', 0)}")
            return True
        except Exception as e:
            print(f"⚠️  Error loading ML model: {e} — using fallback ranking")
            return False
    print("⚠️  ML model file not found — using fallback ranking")
    return False

# ── SKILL KEYWORDS ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SKILL_FILE = os.path.join(BASE_DIR, "skill_keywords.json")

with open(SKILL_FILE, "r", encoding="utf-8") as f:
    SKILL_KEYWORDS = json.load(f)

# ── LIFESPAN (startup / shutdown) ─────────────────────────────────────────────
# The change stream runs in ONE dedicated daemon thread.
# Gunicorn spawns multiple worker PROCESSES — to avoid duplicate emails,
# only the *first* worker that acquires the lock starts the watcher.
# The lock is stored in MongoDB so it works across processes.

WORKER_LOCK_KEY = "change_stream_lock"

def try_acquire_worker_lock() -> bool:
    """Only one Gunicorn worker should run the change stream."""
    try:
        result = db.WorkerLocks.find_one_and_update(
            {"key": WORKER_LOCK_KEY, "locked": False},
            {"$set": {"locked": True, "acquired_at": datetime.now(timezone.utc)}},
            upsert=False,
        )
        if result:
            return True
        # If doc doesn't exist yet, create and acquire it
        db.WorkerLocks.update_one(
            {"key": WORKER_LOCK_KEY},
            {"$setOnInsert": {"key": WORKER_LOCK_KEY, "locked": True,
                              "acquired_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        # Check if we won the race
        doc = db.WorkerLocks.find_one({"key": WORKER_LOCK_KEY})
        return doc is not None
    except Exception:
        return False

def release_worker_lock():
    try:
        db.WorkerLocks.update_one(
            {"key": WORKER_LOCK_KEY},
            {"$set": {"locked": False}},
        )
    except Exception:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    load_ml_model()
    # Start change stream watcher in exactly one worker
    if try_acquire_worker_lock():
        print("🔑 This worker acquired the change stream lock")
        t = threading.Thread(target=watch_requirements, daemon=True)
        t.start()
    else:
        print("ℹ️  Another worker is handling the change stream")

    print("=" * 60)
    print("🚀 Mumbai121 FastAPI server started")
    print("=" * 60)

    yield  # Application runs here

    # Shutdown
    release_worker_lock()
    print("🛑 Server shutting down — lock released")

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mumbai121 API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS  (identical logic to Flask version)
# ══════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_resume_to_gridfs(file_bytes: bytes, filename: str,
                           candidate_name: str, candidate_email: str) -> str | None:
    try:
        safe_name = secure_filename(f"{candidate_name}_{candidate_email}_{filename}")
        file_id = fs.put(
            file_bytes,
            filename=safe_name,
            content_type="application/pdf",
            metadata={
                "candidate_name": candidate_name,
                "candidate_email": candidate_email,
                "upload_date": datetime.now(timezone.utc),
            },
        )
        print(f"✅ Resume saved to GridFS: {safe_name} (ID: {file_id})")
        return str(file_id)
    except Exception as e:
        print(f"❌ Error saving resume: {e}")
        return None


def get_resume_from_gridfs(file_id: str):
    try:
        return fs.get(ObjectId(file_id))
    except Exception as e:
        print(f"❌ Error retrieving resume: {e}")
        return None


def normalize_railway_name(railway: str) -> str:
    if not railway:
        return ""
    return str(railway).lower().strip().replace("railway", "").replace("line", "").strip()


def normalize_candidate_data(candidate: dict) -> dict:
    """Standardise field names across Fresher and PwBD documents."""
    normalized = dict(candidate)
    if "fullName" in normalized and "name" not in normalized:
        normalized["name"] = normalized["fullName"]
    if "railwayPreference" in normalized and "railways" not in normalized:
        normalized["railways"] = normalized["railwayPreference"]
    return normalized


def match_railway_lines(company_railways, candidate_railways) -> bool:
    if not company_railways or not candidate_railways:
        return False
    if not isinstance(company_railways, list):
        company_railways = [company_railways]
    if not isinstance(candidate_railways, list):
        candidate_railways = [candidate_railways]
    company_set   = {normalize_railway_name(r) for r in company_railways if r}
    candidate_set = {normalize_railway_name(r) for r in candidate_railways if r}
    return bool(company_set & candidate_set)


def get_matching_candidates(company_job: str, company_railways: list,
                             collection) -> list:
    matches = []
    for candidate in collection.find():
        nc = normalize_candidate_data(candidate)
        if nc.get("jobPreference") == company_job and \
                match_railway_lines(company_railways, nc.get("railways", [])):
            matches.append(nc)
    return matches


def rank_candidates_with_ml(work_description: str, candidates: list,
                              job_category: str, candidate_type: str) -> list:
    if not candidates:
        return []
    print(f"🤖 Ranking {len(candidates)} {candidate_type} candidates...")

    # Try custom ML model
    if ML_MODEL and ML_VECTORIZER:
        try:
            features = [
                f"{work_description} {c.get('skills','')} "
                f"{c.get('course','')} {c.get('college','')}"
                for c in candidates
            ]
            X      = ML_VECTORIZER.transform(features)
            scores = ML_MODEL.predict(X)
            ranked = [c for c, _ in sorted(zip(candidates, scores),
                                           key=lambda x: x[1], reverse=True)]
            print(f"✅ ML ranked {len(ranked)} candidates | "
                  f"top={scores.max():.3f} bottom={scores.min():.3f}")
            return ranked
        except Exception as e:
            print(f"⚠️  ML ranking failed: {e} — falling back to TF-IDF")

    # Fallback: TF-IDF cosine similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = [work_description] + [
            f"{c.get('skills','')} {c.get('course','')} {c.get('college','')}"
            for c in candidates
        ]
        vectorizer   = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        sims         = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        ranked       = [c for c, _ in sorted(zip(candidates, sims),
                                             key=lambda x: x[1], reverse=True)]
        print(f"✅ TF-IDF ranked {len(ranked)} candidates")
        return ranked
    except Exception as e:
        print(f"⚠️  Fallback ranking error: {e} — returning original order")
        return candidates


def generate_html_table(candidates: list, candidate_type: str) -> str:
    if not candidates:
        return f"<p>No matching {candidate_type} candidates found.</p>"

    html = f"""
    <h2>{candidate_type.upper()} Candidates ({len(candidates)} matches)</h2>
    <table border="1" cellpadding="8" cellspacing="0"
           style="border-collapse:collapse;width:100%;font-family:Arial">
        <tr style="background:#4CAF50;color:white">
            <th>S.No</th><th>Name</th><th>Email</th><th>WhatsApp</th>
            <th>College</th><th>Course</th><th>Skills</th>
    """
    if candidate_type == "pwbd":
        html += "<th>Disability</th>"
    html += "</tr>"

    for i, c in enumerate(candidates, 1):
        bg    = "#f9f9f9" if i % 2 == 0 else "white"
        name  = c.get("name") or c.get("fullName", "N/A")
        html += f"""
        <tr style="background:{bg}">
            <td>{i}</td><td>{name}</td>
            <td>{c.get('email','N/A')}</td><td>{c.get('whatsapp','N/A')}</td>
            <td>{c.get('college','N/A')}</td><td>{c.get('course','N/A')}</td>
            <td>{c.get('skills','N/A')}</td>
        """
        if candidate_type == "pwbd":
            html += f"<td>{c.get('disability','N/A')}</td>"
        html += "</tr>"

    html += "</table><br><br>"
    return html


def send_email_with_resumes(to_email: str, company_name: str,
                             fresher_candidates: list, pwbd_candidates: list) -> bool:
    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"Matched Candidates for {company_name}"
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = to_email

    body = f"""
    <html><body>
        <p>Hello {company_name},</p>
        <p>We have searched our database and found the following freshers who match
           your job profile. Please find their details below:</p>
        {generate_html_table(fresher_candidates, "fresher")}
        <p>We are also sending you the details of PwBD candidates who match your
           job profile:</p>
        {generate_html_table(pwbd_candidates, "pwbd")}
        <p><strong>Note:</strong> Resumes for all candidates are attached.</p>
        <p>Best Regards,<br>The Mumbai121 Team</p>
    </body></html>
    """
    msg_body = MIMEMultipart("alternative")
    msg_body.attach(MIMEText(body, "html"))
    msg.attach(msg_body)

    # Attach resumes
    for label, candidates in [("Fresher", fresher_candidates),
                               ("PwBD",   pwbd_candidates)]:
        for i, c in enumerate(candidates, 1):
            resume_id = c.get("resumeFileId")
            if not resume_id:
                continue
            try:
                resume_file = get_resume_from_gridfs(resume_id)
                if resume_file:
                    name     = c.get("name") or c.get("fullName", f"{label}_{i}")
                    part     = MIMEBase("application", "pdf")
                    part.set_payload(resume_file.read())
                    encoders.encode_base64(part)
                    safe_name = secure_filename(name)
                    part.add_header("Content-Disposition",
                                    f'attachment; filename="{safe_name}_{label}_Resume.pdf"')
                    msg.attach(part)
                    print(f"✅ Attached resume for {name}")
            except Exception as e:
                print(f"⚠️  Could not attach resume for {label} {i}: {e}")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"✅ Email sent to {to_email} — "
              f"{len(fresher_candidates)} freshers, {len(pwbd_candidates)} PwBDs")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        traceback.print_exc()
        return False


def send_no_candidates_email(to_email: str, company_name: str) -> bool:
    try:
        msg = MIMEMultipart()
        msg["Subject"] = f"No Additional Candidates Available — {company_name}"
        msg["From"]    = EMAIL_ADDRESS
        msg["To"]      = to_email
        body = f"""
        <html><body>
            <h2>Hello {company_name},</h2>
            <p>We have searched our database thoroughly, but unfortunately we do not
               have any additional candidates matching your requirements at this time.</p>
            <p>All available matching candidates have already been sent to you.</p>
            <p>We recommend reviewing the candidates already sent, and checking back
               in a few days as new candidates register daily.</p>
            <p>Best regards,<br>The Mumbai121 Team</p>
        </body></html>
        """
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"❌ No-candidates email error: {e}")
        return False


def process_requirement_internal(requirement_id: str, is_resend: bool = False) -> bool:
    try:
        requirement = db.Requirements.find_one({"_id": ObjectId(requirement_id)})
        if not requirement:
            print(f"❌ Requirement {requirement_id} not found")
            return False

        company  = requirement.get("company")
        email    = requirement.get("email")
        job      = requirement.get("jobPreference")
        desc     = requirement.get("workDescription")
        railways = requirement.get("railways", [])

        print(f"\n{'='*60}")
        print(f"📧 Processing: {company} | Job: {job}")
        print(f"{'='*60}")

        all_freshers = get_matching_candidates(job, railways, db.Freshers)
        all_pwbds    = get_matching_candidates(job, railways, db.PwBDs)
        print(f"📊 Found: {len(all_freshers)} freshers, {len(all_pwbds)} PwBDs")

        sent_fresher_ids = set(requirement.get("sentFresherIds", []))
        sent_pwbd_ids    = set(requirement.get("sentPwbdIds",    []))

        available_freshers = [c for c in all_freshers
                               if str(c.get("_id")) not in sent_fresher_ids]
        available_pwbds    = [c for c in all_pwbds
                               if str(c.get("_id")) not in sent_pwbd_ids]

        print(f"📊 Available: {len(available_freshers)} freshers, "
              f"{len(available_pwbds)} PwBDs (after deduplication)")

        if not available_freshers and not available_pwbds:
            print(f"⚠️  No new candidates for {company}")
            send_no_candidates_email(email, company)
            if is_resend:
                db.Requirements.update_one(
                    {"_id": ObjectId(requirement_id)},
                    {"$set": {"resendRequested": False}},
                )
            return False

        ranked_freshers = rank_candidates_with_ml(desc, available_freshers, job, "fresher")
        ranked_pwbds    = rank_candidates_with_ml(desc, available_pwbds,    job, "pwbd")

        final_freshers = ranked_freshers[:10]
        final_pwbds    = ranked_pwbds[:10]

        print(f"✅ Sending: {len(final_freshers)} freshers, {len(final_pwbds)} PwBDs")

        if send_email_with_resumes(email, company, final_freshers, final_pwbds):
            new_fresher_ids = [str(c.get("_id")) for c in final_freshers]
            new_pwbd_ids    = [str(c.get("_id")) for c in final_pwbds]
            sent_fresher_ids.update(new_fresher_ids)
            sent_pwbd_ids.update(new_pwbd_ids)

            update_data = {
                "emailSent":         True,
                "aiProcessedAt":     datetime.now(timezone.utc),
                "sentFresherIds":    list(sent_fresher_ids),
                "sentPwbdIds":       list(sent_pwbd_ids),
                "totalFreshersSent": len(sent_fresher_ids),
                "totalPwbdsSent":    len(sent_pwbd_ids),
                "lastEmailSentAt":   datetime.now(timezone.utc),
            }
            if not is_resend:
                update_data["aiProcessed"] = True
            else:
                update_data["resendRequested"] = False

            db.Requirements.update_one(
                {"_id": ObjectId(requirement_id)},
                {"$set": update_data},
            )
            print(f"{'='*60}\n")
            return True

        print(f"❌ Email failed for {company}")
        return False

    except Exception as e:
        print(f"❌ process_requirement_internal error: {e}")
        traceback.print_exc()
        return False


# ── CHANGE STREAM (runs in single daemon thread) ──────────────────────────────
def watch_requirements():
    print("👀 Change stream watcher started")
    try:
        with db.Requirements.watch(full_document="updateLookup") as stream:
            for change in stream:
                if change["operationType"] != "update":
                    continue
                req_id        = str(change["documentKey"]["_id"])
                doc           = change.get("fullDocument", {})
                updated_fields = change.get("updateDescription", {}).get("updatedFields", {})

                if updated_fields.get("processed") is True and not doc.get("aiProcessed", False):
                    print(f"\n🆕 Admin approved: {req_id}")
                    process_requirement_internal(req_id, is_resend=False)

                if updated_fields.get("resendRequested") is True:
                    print(f"\n🔄 Resend requested: {req_id}")
                    process_requirement_internal(req_id, is_resend=True)

    except Exception as e:
        print(f"❌ Change stream error: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/requirements", status_code=201)
def create_requirement(data: dict):
    # Normalise field names — CompanyForm previously sent 'jobType' instead of 'jobPreference'
    # and 'workType' instead of 'workDescription'
    if "jobType" in data and "jobPreference" not in data:
        data["jobPreference"] = data.pop("jobType")
    if "workType" in data and "workDescription" not in data:
        data["workDescription"] = data.pop("workType")

    data.update({
        "processed":         False,
        "aiProcessed":       False,
        "processedAt":       None,
        "aiProcessedAt":     None,
        "emailSent":         False,
        "sentFresherIds":    [],
        "sentPwbdIds":       [],
        "totalFreshersSent": 0,
        "totalPwbdsSent":    0,
        "resendRequested":   False,
    })
    result = db.Requirements.insert_one(data)
    print(f"New requirement: {result.inserted_id} | job: {data.get('jobPreference')} | railways: {data.get('railways')}")
    return {"message": "Requirement submitted", "id": str(result.inserted_id)}


@app.post("/register/fresher", status_code=201)
async def register_fresher(
    fullName:          str        = Form(...),
    email:             str        = Form(...),
    whatsapp:          str        = Form(...),
    railwayPreference: str        = Form("[]"),
    college:           str        = Form(...),
    course:            str        = Form(...),
    year:              str        = Form(...),
    jobPreference:     str        = Form(...),
    skills:            str        = Form(...),
    mmrResident:       str        = Form(...),
    consent:           str        = Form(...),
    resume:            UploadFile = File(...),
):
    try:
        # Validate file type
        if not allowed_file(resume.filename):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_bytes = await resume.read()

        # Validate file size
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")

        railways_list = json.loads(railwayPreference)

        data = {
            "fullName":          fullName,
            "name":              fullName,          # normalised field
            "email":             email,
            "whatsapp":          whatsapp,
            "railwayPreference": railways_list,
            "railways":          railways_list,     # normalised field
            "college":           college,
            "course":            course,
            "year":              year,
            "jobPreference":     jobPreference,
            "skills":            skills,
            "mmrResident":       mmrResident,
            "consent":           consent,
            "registrationDate":  datetime.now(timezone.utc),
        }

        resume_id = save_resume_to_gridfs(file_bytes, resume.filename, fullName, email)
        if resume_id:
            data["resumeFileId"]  = resume_id
            data["resumeFilename"] = secure_filename(resume.filename)

        result = db.Freshers.insert_one(data)
        print(f"👤 Fresher registered: {result.inserted_id} (Resume: {resume_id})")
        return {"message": "Fresher registered successfully", "id": str(result.inserted_id)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Fresher registration error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register/pwbd", status_code=201)
async def register_pwbd(
    name:          str        = Form(...),
    disability:    str        = Form(...),
    email:         str        = Form(...),
    whatsapp:      str        = Form(...),
    railways:      str        = Form("[]"),
    college:       str        = Form(...),
    course:        str        = Form(...),
    year:          str        = Form(...),
    jobPreference: str        = Form(...),
    skills:        str        = Form(...),
    mmrResident:   str        = Form(...),
    consent:                str        = Form(...),
    resume:                 UploadFile = File(...),
    disability_certificate: UploadFile = File(...),
):
    try:
        # Validate resume
        if not allowed_file(resume.filename):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed for resume")
        file_bytes = await resume.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Resume file size exceeds 5MB limit")

        # Validate disability certificate
        if not allowed_file(disability_certificate.filename):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed for disability certificate")
        cert_bytes = await disability_certificate.read()
        if len(cert_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Disability certificate file size exceeds 5MB limit")

        railways_list = json.loads(railways)

        data = {
            "name":             name,
            "disability":       disability,
            "email":            email,
            "whatsapp":         whatsapp,
            "railways":         railways_list,
            "college":          college,
            "course":           course,
            "year":             year,
            "jobPreference":    jobPreference,
            "skills":           skills,
            "mmrResident":      mmrResident,
            "consent":          consent,
            "registrationDate": datetime.now(timezone.utc),
        }

        # Save resume
        resume_id = save_resume_to_gridfs(file_bytes, resume.filename, name, email)
        if resume_id:
            data["resumeFileId"]   = resume_id
            data["resumeFilename"] = secure_filename(resume.filename)

        # Save disability certificate with name suffix
        cert_safe_filename = f"{name}_Disability_Certificate.pdf"
        cert_id = save_resume_to_gridfs(cert_bytes, cert_safe_filename, name, email)
        if cert_id:
            data["disabilityCertificateFileId"]   = cert_id
            data["disabilityCertificateFilename"] = cert_safe_filename

        result = db.PwBDs.insert_one(data)
        print(f"👤 PwBD registered: {result.inserted_id} (Resume: {resume_id} | Cert: {cert_id})")
        return {"message": "PwBD registered successfully", "id": str(result.inserted_id)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ PwBD registration error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register/volunteer", status_code=201)
def register_volunteer(data: dict):
    result = db.Volunteers.insert_one(data)
    print(f"👤 Volunteer registered: {result.inserted_id}")
    return {"message": "Volunteer registered"}


@app.post("/contact", status_code=201)
def contact(data: dict):
    result = db.ContactUs.insert_one(data)
    print(f"📧 Contact submitted: {result.inserted_id}")
    return {"message": "Contact submitted"}


@app.get("/admin/requirements")
def get_requirements():
    try:
        requirements = list(db.Requirements.find())
        for req in requirements:
            req["_id"] = str(req["_id"])
        stats = {
            "freshers":  db.Freshers.count_documents({}),
            "pwbd":      db.PwBDs.count_documents({}),
            "volunteers": db.Volunteers.count_documents({}),
        }
        return {"requirements": requirements, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/resend-candidates/{requirement_id}")
def resend_candidates(requirement_id: str):
    try:
        db.Requirements.update_one(
            {"_id": ObjectId(requirement_id)},
            {"$set": {"resendRequested": True}},
        )
        return {"message": "Resend triggered — 10 more candidates will be sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/companies")
def get_companies():
    """Fetch up to 12 approved companies for the homepage showcase."""
    try:
        companies = list(
            db.Requirements.find(
                {},
                {"company": 1, "jobPreference": 1, "employees": 1, "_id": 0}
            ).limit(12)
        )
        return companies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """Public stats endpoint for homepage counters."""
    try:
        return {
            "freshers":  db.Freshers.count_documents({}),
            "pwbd":      db.PwBDs.count_documents({}),
            "companies": db.Requirements.count_documents({}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Simple health check endpoint for load balancer / uptime monitoring."""
    return {"status": "ok"}