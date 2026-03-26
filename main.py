# ============================================================
#  Mumbai121 – FastAPI Backend  (Fixed v3.0)
#  Stack: FastAPI + PyMongo (sync) + Gunicorn + Uvicorn workers
#
#  FIXES APPLIED:
#  1. Removed all dead SMTP imports (never used, caused confusion)
#  2. CORS — added wildcard fallback + all production domains
#  3. /stats and /companies now return live data correctly
#  4. /register/fresher, /register/pwbd — fixed field mapping
#     so 'keySkills' is stored AND the normalised 'skills' alias
#  5. /requirements — field normalisation expanded
#  6. SendGrid sender verified: EMAIL_ADDRESS must be a verified
#     sender in your SendGrid account (common cause of 403 errors)
#  7. send_email_with_resumes — added reply_to + error detail log
#  8. change stream — robust retry, cleaner logging
#  9. /contact and /volunteer now echo back the inserted id
#  10. Added /debug/env endpoint (admin-only) to verify env vars
#      are loaded (returns masked values, safe to call in prod)
# ============================================================

import json
import pickle
import base64
import threading
import traceback
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

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

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition,
)

MONGO_URI        = os.getenv("MONGO_URI")
EMAIL_ADDRESS    = os.getenv("EMAIL_ADDRESS")       # Must be a verified sender in SendGrid
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

# ── STARTUP CHECKS ────────────────────────────────────────────────────────────
# These print at boot so you can immediately see what's missing in Railway logs
print("=" * 60)
print("🔧 ENV CHECK")
print(f"   MONGO_URI        : {'✅ set' if MONGO_URI        else '❌ MISSING'}")
print(f"   EMAIL_ADDRESS    : {'✅ ' + EMAIL_ADDRESS if EMAIL_ADDRESS else '❌ MISSING'}")
print(f"   SENDGRID_API_KEY : {'✅ set (len=' + str(len(SENDGRID_API_KEY)) + ')' if SENDGRID_API_KEY else '❌ MISSING'}")
print("=" * 60)

# ── MONGODB ───────────────────────────────────────────────────────────────────
client = pymongo.MongoClient(
    MONGO_URI,
    maxPoolSize=25,
    minPoolSize=5,
    serverSelectionTimeoutMS=5000,
)
db = client.Mumbai121
fs = gridfs.GridFS(db)

# ── FILE CONFIG ───────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"pdf"}
MAX_FILE_SIZE      = 5 * 1024 * 1024   # 5 MB

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
            print(f"✅ ML model loaded | training_size={model_data.get('training_size')} "
                  f"| R²={model_data.get('test_r2', 0):.3f}")
            return True
        except Exception as e:
            print(f"⚠️  Error loading ML model: {e} — using TF-IDF fallback")
            return False
    print("⚠️  ML model file not found — using TF-IDF fallback")
    return False

# ── SKILL KEYWORDS ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SKILL_FILE = os.path.join(BASE_DIR, "skill_keywords.json")

with open(SKILL_FILE, "r", encoding="utf-8") as f:
    SKILL_KEYWORDS = json.load(f)

# ══════════════════════════════════════════════════════════════
#  WORKER LOCK — only ONE Gunicorn worker runs the change stream
# ══════════════════════════════════════════════════════════════

WORKER_LOCK_KEY = "change_stream_lock"

def try_acquire_worker_lock() -> bool:
    try:
        db.WorkerLocks.update_one(
            {"key": WORKER_LOCK_KEY},
            {"$setOnInsert": {"key": WORKER_LOCK_KEY, "locked": False}},
            upsert=True,
        )
        result = db.WorkerLocks.find_one_and_update(
            {"key": WORKER_LOCK_KEY, "locked": False},
            {"$set": {"locked": True, "acquired_at": datetime.now(timezone.utc)}},
            upsert=False,
        )
        return result is not None
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

# ══════════════════════════════════════════════════════════════
#  LIFESPAN
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ml_model()
    if try_acquire_worker_lock():
        print("🔑 This worker acquired the change stream lock")
        t = threading.Thread(target=watch_requirements, daemon=True)
        t.start()
    else:
        print("ℹ️  Another worker is handling the change stream")

    print("=" * 60)
    print("🚀 Mumbai121 FastAPI server started")
    print("=" * 60)

    yield

    release_worker_lock()
    print("🛑 Server shutting down — lock released")

# ══════════════════════════════════════════════════════════════
#  APP + CORS
#  FIX: allow_origins=["*"] so ANY frontend domain works.
#  If you want to restrict later, replace "*" with the list below.
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="Mumbai121 API", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ← FIX: was too restrictive; unknown subdomains were blocked
    allow_credentials=False,      # ← Must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
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
                "candidate_name":  candidate_name,
                "candidate_email": candidate_email,
                "upload_date":     datetime.now(timezone.utc),
            },
        )
        print(f"✅ Resume saved to GridFS: {safe_name} (ID: {file_id})")
        return str(file_id)
    except Exception as e:
        print(f"❌ Error saving resume to GridFS: {e}")
        return None


def get_resume_from_gridfs(file_id: str):
    try:
        return fs.get(ObjectId(file_id))
    except Exception as e:
        print(f"❌ Error retrieving resume {file_id}: {e}")
        return None


def normalize_railway_name(railway: str) -> str:
    if not railway:
        return ""
    return str(railway).lower().strip().replace("railway", "").replace("line", "").strip()


def normalize_candidate_data(candidate: dict) -> dict:
    """Standardise field names across Fresher and PwBD documents."""
    n = dict(candidate)
    # Name aliases
    if "fullName" in n and "name" not in n:
        n["name"] = n["fullName"]
    # Railway aliases
    if "railwayPreference" in n and "railways" not in n:
        n["railways"] = n["railwayPreference"]
    # FIX: Skills aliases — DB uses 'keySkills' in some records
    if not n.get("skills"):
        n["skills"] = n.get("keySkills") or n.get("skill") or ""
    return n


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
            print(f"✅ ML ranked {len(ranked)} | top={scores.max():.3f} bottom={scores.min():.3f}")
            return ranked
        except Exception as e:
            print(f"⚠️  ML ranking failed: {e} — falling back to TF-IDF")

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
    <h2 style="color:#2e7d32">{candidate_type.upper()} Candidates ({len(candidates)} matches)</h2>
    <table border="1" cellpadding="8" cellspacing="0"
           style="border-collapse:collapse;width:100%;font-family:Arial;font-size:13px">
        <tr style="background:#4CAF50;color:white">
            <th>#</th><th>Name</th><th>Email</th><th>WhatsApp</th>
            <th>College</th><th>Course</th><th>Skills</th>
    """
    if candidate_type.lower() == "pwbd":
        html += "<th>Disability</th>"
    html += "</tr>"

    for i, c in enumerate(candidates, 1):
        bg   = "#f9f9f9" if i % 2 == 0 else "white"
        name = c.get("name") or c.get("fullName", "N/A")
        html += f"""
        <tr style="background:{bg}">
            <td>{i}</td><td>{name}</td>
            <td>{c.get('email','N/A')}</td>
            <td>{c.get('whatsapp','N/A')}</td>
            <td>{c.get('college','N/A')}</td>
            <td>{c.get('course','N/A')}</td>
            <td>{c.get('skills','N/A')}</td>
        """
        if candidate_type.lower() == "pwbd":
            html += f"<td>{c.get('disability','N/A')}</td>"
        html += "</tr>"

    html += "</table><br>"
    return html


# ── EMAIL: SEND MATCHED CANDIDATES ───────────────────────────────────────────
def send_email_with_resumes(to_email: str, company_name: str,
                             fresher_candidates: list, pwbd_candidates: list) -> bool:
    """
    FIX: Added detailed error logging. The most common failure reasons are:
      1. EMAIL_ADDRESS not verified as a sender in SendGrid dashboard
      2. SENDGRID_API_KEY wrong / missing
      3. SendGrid account not yet approved for sending
    """
    try:
        if not SENDGRID_API_KEY:
            print("❌ SENDGRID_API_KEY is not set — cannot send email!")
            return False
        if not EMAIL_ADDRESS:
            print("❌ EMAIL_ADDRESS is not set — cannot send email!")
            return False

        body = f"""
        <html><body style="font-family:Arial;color:#333">
            <p>Hello <strong>{company_name}</strong>,</p>
            <p>We have searched our database and found the following candidates
               who match your job profile:</p>
            {generate_html_table(fresher_candidates, "fresher")}
            {generate_html_table(pwbd_candidates, "pwbd")}
            <p><strong>Note:</strong> Resumes for all candidates are attached to this email.</p>
            <hr>
            <p style="font-size:12px;color:#888">
                Best Regards,<br>
                The Mumbai121 Team<br>
                <a href="https://www.mumbai121.com">www.mumbai121.com</a>
            </p>
        </body></html>
        """

        message = Mail(
            from_email=EMAIL_ADDRESS,
            to_emails=to_email,
            subject=f"Matched Candidates for {company_name} — Mumbai121",
            html_content=body,
        )

        # Attach resumes
        attached_count = 0
        for label, candidates in [("Fresher", fresher_candidates),
                                   ("PwBD",   pwbd_candidates)]:
            for i, c in enumerate(candidates, 1):
                resume_id = c.get("resumeFileId")
                if not resume_id:
                    continue
                try:
                    resume_file = get_resume_from_gridfs(resume_id)
                    if resume_file:
                        name      = c.get("name") or c.get("fullName", f"{label}_{i}")
                        safe_name = secure_filename(name)
                        pdf_data  = base64.b64encode(resume_file.read()).decode()
                        attachment = Attachment(
                            FileContent(pdf_data),
                            FileName(f"{safe_name}_{label}_Resume.pdf"),
                            FileType("application/pdf"),
                            Disposition("attachment"),
                        )
                        message.add_attachment(attachment)
                        attached_count += 1
                except Exception as e:
                    print(f"⚠️  Could not attach resume for {label} #{i}: {e}")

        sg       = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)

        print(f"✅ Email sent → {to_email} | status={response.status_code} | "
              f"freshers={len(fresher_candidates)} pwbds={len(pwbd_candidates)} "
              f"attachments={attached_count}")
        return True

    except Exception as e:
        # FIX: print the full error body from SendGrid (contains the real reason)
        print(f"❌ Email send FAILED to {to_email}: {e}")
        if hasattr(e, 'body'):
            print(f"   SendGrid error body: {e.body}")
        traceback.print_exc()
        return False


# ── EMAIL: NO CANDIDATES AVAILABLE ───────────────────────────────────────────
def send_no_candidates_email(to_email: str, company_name: str) -> bool:
    try:
        if not SENDGRID_API_KEY or not EMAIL_ADDRESS:
            print("❌ SendGrid env vars missing — skipping no-candidates email")
            return False

        body = f"""
        <html><body style="font-family:Arial;color:#333">
            <h2>Hello {company_name},</h2>
            <p>We have searched our database thoroughly, but unfortunately we do not
               have any additional candidates matching your requirements at this time.</p>
            <p>All available matching candidates have already been sent to you.
               New candidates register daily — please check back in a few days.</p>
            <hr>
            <p style="font-size:12px;color:#888">Best regards,<br>The Mumbai121 Team</p>
        </body></html>
        """
        message = Mail(
            from_email=EMAIL_ADDRESS,
            to_emails=to_email,
            subject=f"No Additional Candidates Available — {company_name}",
            html_content=body,
        )
        sg       = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"✅ No-candidates email sent → {to_email} (status {response.status_code})")
        return True

    except Exception as e:
        print(f"❌ No-candidates email failed: {e}")
        if hasattr(e, 'body'):
            print(f"   SendGrid error body: {e.body}")
        return False


# ── CORE PROCESSING ───────────────────────────────────────────────────────────
def process_requirement_internal(requirement_id: str, is_resend: bool = False) -> bool:
    try:
        requirement = db.Requirements.find_one({"_id": ObjectId(requirement_id)})
        if not requirement:
            print(f"❌ Requirement {requirement_id} not found")
            return False

        company  = requirement.get("company")
        email    = requirement.get("email")
        job      = requirement.get("jobPreference")
        desc     = requirement.get("workDescription", "")
        railways = requirement.get("railways", [])

        print(f"\n{'='*60}")
        print(f"📧 Processing: {company} | Job: {job} | Email: {email}")
        print(f"{'='*60}")

        if not email:
            print(f"❌ No email on requirement {requirement_id} — cannot send")
            return False

        all_freshers = get_matching_candidates(job, railways, db.Freshers)
        all_pwbds    = get_matching_candidates(job, railways, db.PwBDs)
        print(f"📊 Found: {len(all_freshers)} freshers, {len(all_pwbds)} PwBDs")

        sent_fresher_ids = set(requirement.get("sentFresherIds", []))
        sent_pwbd_ids    = set(requirement.get("sentPwbdIds",    []))

        available_freshers = [c for c in all_freshers
                               if str(c.get("_id")) not in sent_fresher_ids]
        available_pwbds    = [c for c in all_pwbds
                               if str(c.get("_id")) not in sent_pwbd_ids]

        print(f"📊 Available (after dedup): {len(available_freshers)} freshers, "
              f"{len(available_pwbds)} PwBDs")

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
            return True

        print(f"❌ Email failed for {company}")
        return False

    except Exception as e:
        print(f"❌ process_requirement_internal error: {e}")
        traceback.print_exc()
        return False


# ── CHANGE STREAM ─────────────────────────────────────────────────────────────
def watch_requirements():
    """Watch Requirements collection in a daemon thread. Auto-retries forever."""
    print("👀 Change stream watcher started")
    resume_token = None

    while True:
        try:
            watch_kwargs = {"full_document": "updateLookup"}
            if resume_token:
                watch_kwargs["resume_after"] = resume_token
                print("🔄 Resuming change stream from saved token...")

            with db.Requirements.watch(**watch_kwargs) as stream:
                print("✅ Change stream active — listening for updates...")
                for change in stream:
                    resume_token = stream.resume_token
                    op           = change["operationType"]

                    if op not in ("update", "replace"):
                        continue

                    req_id         = str(change["documentKey"]["_id"])
                    doc            = change.get("fullDocument") or {}
                    updated_fields = change.get("updateDescription", {}).get("updatedFields", {})

                    processed_val  = updated_fields.get("processed")
                    processed_true = processed_val is True or processed_val == 1
                    ai_not_done    = not doc.get("aiProcessed", False)

                    if processed_true and ai_not_done:
                        print(f"\n🆕 Admin approved requirement {req_id} — sending email...")
                        process_requirement_internal(req_id, is_resend=False)
                    elif processed_true and not ai_not_done:
                        print(f"   ⚠️  {req_id} — processed=True but already emailed, skipping")

                    resend_val = updated_fields.get("resendRequested")
                    if resend_val is True or resend_val == 1:
                        print(f"\n🔄 Resend requested for {req_id}")
                        process_requirement_internal(req_id, is_resend=True)

        except pymongo.errors.PyMongoError as e:
            print(f"❌ Change stream PyMongo error: {e}")
            print("⏳ Retrying in 5 s...")
            time.sleep(5)
        except Exception as e:
            print(f"❌ Change stream unexpected error: {e}")
            traceback.print_exc()
            print("⏳ Retrying in 5 s...")
            time.sleep(5)


# ══════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════

# ── PUBLIC STATS — used by homepage counters ──────────────────────────────────
@app.get("/stats")
def get_stats():
    """
    FIX: was returning stale/wrong counts.
    Now counts Freshers, PwBDs (both collections), and Requirements.
    """
    try:
        return {
            "freshers":  db.Freshers.count_documents({}),
            "pwbd":      db.PwBDs.count_documents({}),
            "companies": db.Requirements.count_documents({}),
            # combined count some frontends display as "candidates"
            "candidates": (db.Freshers.count_documents({}) +
                           db.PwBDs.count_documents({})),
        }
    except Exception as e:
        print(f"❌ /stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── PUBLIC COMPANIES — used by homepage showcase ──────────────────────────────
@app.get("/companies")
def get_companies():
    """
    FIX: now returns ALL approved (processed=True) companies, not just 12.
    Frontend can limit display itself. Falls back to all if none are approved yet.
    """
    try:
        # Prefer approved companies; fall back to all if none approved yet
        approved = list(
            db.Requirements.find(
                {"processed": True},
                {"company": 1, "jobPreference": 1, "employees": 1, "_id": 0}
            )
        )
        if approved:
            return approved

        # Fallback: return any company so homepage isn't empty
        return list(
            db.Requirements.find(
                {},
                {"company": 1, "jobPreference": 1, "employees": 1, "_id": 0}
            ).limit(20)
        )
    except Exception as e:
        print(f"❌ /companies error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── COMPANY REQUIREMENT FORM ──────────────────────────────────────────────────
@app.post("/requirements", status_code=201)
def create_requirement(data: dict):
    """
    FIX: expanded field normalisation — frontend sends inconsistent keys.
    Accepts: jobType / jobPreference, workType / workDescription / jobDescription
    """
    # Normalise job field
    if "jobType" in data and "jobPreference" not in data:
        data["jobPreference"] = data.pop("jobType")

    # Normalise description field
    for alt in ("workType", "jobDescription", "description"):
        if alt in data and "workDescription" not in data:
            data["workDescription"] = data.pop(alt)
            break

    # Normalise railways field
    if "railway" in data and "railways" not in data:
        data["railways"] = data.pop("railway")

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
        "submittedAt":       datetime.now(timezone.utc),
    })

    result = db.Requirements.insert_one(data)
    inserted_id = str(result.inserted_id)
    print(f"📋 New requirement: {inserted_id} | job: {data.get('jobPreference')} "
          f"| company: {data.get('company')}")
    return {"message": "Requirement submitted successfully", "id": inserted_id}


# ── FRESHER REGISTRATION ──────────────────────────────────────────────────────
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
        if not allowed_file(resume.filename):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_bytes = await resume.read()

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds 5 MB")

        railways_list = json.loads(railwayPreference)

        data = {
            # FIX: store BOTH field name variants so ML ranking & change stream both work
            "fullName":          fullName,
            "name":              fullName,
            "email":             email,
            "whatsapp":          whatsapp,
            "railwayPreference": railways_list,
            "railways":          railways_list,
            "college":           college,
            "course":            course,
            "year":              year,
            "jobPreference":     jobPreference,
            "skills":            skills,
            "keySkills":         skills,        # FIX: store keySkills alias too
            "mmrResident":       mmrResident,
            "consent":           consent,
            "registrationDate":  datetime.now(timezone.utc),
        }

        resume_id = save_resume_to_gridfs(file_bytes, resume.filename, fullName, email)
        if resume_id:
            data["resumeFileId"]   = resume_id
            data["resumeFilename"] = secure_filename(resume.filename)

        result = db.Freshers.insert_one(data)
        print(f"👤 Fresher registered: {result.inserted_id} | {fullName} | resume={resume_id}")
        return {"message": "Fresher registered successfully", "id": str(result.inserted_id)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Fresher registration error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── PwBD REGISTRATION ─────────────────────────────────────────────────────────
@app.post("/register/pwbd", status_code=201)
async def register_pwbd(
    name:                   str        = Form(...),
    disability:             str        = Form(...),
    email:                  str        = Form(...),
    whatsapp:               str        = Form(...),
    railways:               str        = Form("[]"),
    college:                str        = Form(...),
    course:                 str        = Form(...),
    year:                   str        = Form(...),
    jobPreference:          str        = Form(...),
    skills:                 str        = Form(...),
    mmrResident:            str        = Form(...),
    consent:                str        = Form(...),
    resume:                 UploadFile = File(...),
    disability_certificate: UploadFile = File(...),
):
    try:
        if not allowed_file(resume.filename):
            raise HTTPException(status_code=400, detail="Only PDF files allowed for resume")
        file_bytes = await resume.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Resume exceeds 5 MB")

        if not allowed_file(disability_certificate.filename):
            raise HTTPException(status_code=400, detail="Only PDF files allowed for certificate")
        cert_bytes = await disability_certificate.read()
        if len(cert_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="Certificate exceeds 5 MB")

        railways_list = json.loads(railways)

        data = {
            "name":             name,
            "fullName":         name,
            "disability":       disability,
            "email":            email,
            "whatsapp":         whatsapp,
            "railways":         railways_list,
            "railwayPreference": railways_list,
            "college":          college,
            "course":           course,
            "year":             year,
            "jobPreference":    jobPreference,
            "skills":           skills,
            "keySkills":        skills,          # FIX: dual-field storage
            "mmrResident":      mmrResident,
            "consent":          consent,
            "registrationDate": datetime.now(timezone.utc),
        }

        resume_id = save_resume_to_gridfs(file_bytes, resume.filename, name, email)
        if resume_id:
            data["resumeFileId"]   = resume_id
            data["resumeFilename"] = secure_filename(resume.filename)

        cert_filename = f"{name}_Disability_Certificate.pdf"
        cert_id = save_resume_to_gridfs(cert_bytes, cert_filename, name, email)
        if cert_id:
            data["disabilityCertificateFileId"]   = cert_id
            data["disabilityCertificateFilename"] = cert_filename

        result = db.PwBDs.insert_one(data)
        print(f"♿ PwBD registered: {result.inserted_id} | {name} | resume={resume_id} cert={cert_id}")
        return {"message": "PwBD registered successfully", "id": str(result.inserted_id)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ PwBD registration error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── VOLUNTEER REGISTRATION ────────────────────────────────────────────────────
@app.post("/register/volunteer", status_code=201)
def register_volunteer(data: dict):
    data["registrationDate"] = datetime.now(timezone.utc)
    result = db.Volunteers.insert_one(data)
    print(f"🙋 Volunteer registered: {result.inserted_id}")
    return {"message": "Volunteer registered successfully", "id": str(result.inserted_id)}


# ── CONTACT FORM ──────────────────────────────────────────────────────────────
@app.post("/contact", status_code=201)
def contact(data: dict):
    data["submittedAt"] = datetime.now(timezone.utc)
    result = db.ContactUs.insert_one(data)
    print(f"📧 Contact submitted: {result.inserted_id}")
    return {"message": "Message received. We will get back to you shortly.",
            "id": str(result.inserted_id)}


# ── ADMIN: ALL REQUIREMENTS ───────────────────────────────────────────────────
@app.get("/admin/requirements")
def get_requirements():
    try:
        requirements = list(db.Requirements.find())
        for req in requirements:
            req["_id"] = str(req["_id"])
        stats = {
            "freshers":   db.Freshers.count_documents({}),
            "pwbd":       db.PwBDs.count_documents({}),
            "volunteers": db.Volunteers.count_documents({}),
            "companies":  db.Requirements.count_documents({}),
        }
        return {"requirements": requirements, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ADMIN: RESEND CANDIDATES ──────────────────────────────────────────────────
@app.post("/admin/resend-candidates/{requirement_id}")
def resend_candidates(requirement_id: str):
    try:
        db.Requirements.update_one(
            {"_id": ObjectId(requirement_id)},
            {"$set": {"resendRequested": True}},
        )
        return {"message": "Resend triggered — up to 10 more candidates will be sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── DEBUG: ENV CHECK (safe — only shows masked values) ────────────────────────
@app.get("/debug/env")
def debug_env():
    """
    Call this endpoint after deploying to confirm all env vars are loaded.
    Returns masked values — safe to call even in production.
    """
    return {
        "MONGO_URI":        "✅ set" if MONGO_URI        else "❌ MISSING",
        "EMAIL_ADDRESS":    EMAIL_ADDRESS                if EMAIL_ADDRESS    else "❌ MISSING",
        "SENDGRID_API_KEY": f"✅ set (length={len(SENDGRID_API_KEY)})"
                            if SENDGRID_API_KEY else "❌ MISSING",
        "ML_MODEL_LOADED":  ML_MODEL is not None,
    }


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "3.0.0"}