import os
import csv
import sqlite3
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    Response,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from utils import analyze_image
from email.mime.application import MIMEApplication
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from PIL import Image
import numpy as np

app = Flask(__name__)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "cheie_secreta_generica")

app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["GRADCAM_FOLDER"] = "static/gradcam"
app.config["DATABASE_PATH"] = "database/pneumoai.db"

# Ascunde credențialele reale folosind variabile de mediu
app.config["SMTP_EMAIL"] = os.environ.get("SMTP_EMAIL", "adresa_ta@gmail.com")
app.config["SMTP_PASSWORD"] = os.environ.get("SMTP_PASSWORD", "parola_aplicatie_aici")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def create_folders():
    os.makedirs("database", exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["GRADCAM_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
def basic_xray_validation(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        height, width, _ = img_np.shape

        if width < 100 or height < 100:
            return False, "Imaginea este prea mică pentru o radiografie toracică validă."

        r = img_np[:, :, 0].astype(float)
        g = img_np[:, :, 1].astype(float)
        b = img_np[:, :, 2].astype(float)

        color_difference = np.mean(
            np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
        )

        if color_difference > 25:
            return False, "Imagine respinsă. Sistemul acceptă doar radiografii toracice, nu imagini color."

        gray = np.mean(img_np, axis=2)

        dark_ratio = np.mean(gray < 30)
        bright_ratio = np.mean(gray > 245)

        if dark_ratio > 0.75:
            return False, "Imaginea este prea întunecată pentru o radiografie toracică validă."

        if bright_ratio > 0.75:
            return False, "Imaginea este prea luminoasă pentru o radiografie toracică validă."

        return True, ""

    except Exception:
        return False, "Fișierul încărcat nu este o imagine validă."


def get_db():
    conn = sqlite3.connect(app.config["DATABASE_PATH"])
    conn.row_factory = sqlite3.Row
    return conn


def add_column_if_missing(cursor, table, column, column_type):
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    if column not in columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def log_action(action, details=""):
    conn = get_db()
    conn.execute(
        "INSERT INTO logs (timestamp, action, details) VALUES (?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, details),
    )
    conn.commit()
    conn.close()


def init_db():
    create_folders()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            patient_email TEXT,
            patient_age INTEGER,
            patient_sex TEXT,
            prediction TEXT,
            confidence REAL,
            original_path TEXT,
            gradcam_path TEXT,
            created_at TEXT
        )
    """)

    add_column_if_missing(cursor, "history", "prob_normal", "REAL DEFAULT 0")
    add_column_if_missing(cursor, "history", "prob_viral", "REAL DEFAULT 0")
    add_column_if_missing(cursor, "history", "prob_bacterial", "REAL DEFAULT 0")
    add_column_if_missing(cursor, "history", "clinical_notes", "TEXT DEFAULT ''")
    add_column_if_missing(cursor, "history", "guardian_name", "TEXT DEFAULT ''")
    add_column_if_missing(cursor, "history", "guardian_email", "TEXT DEFAULT ''")
    add_column_if_missing(cursor, "history", "relationship", "TEXT DEFAULT ''")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            details TEXT
        )
    """)

    cursor.execute(
        "INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
        ("admin", generate_password_hash("admin123"), "Administrator"),
    )

    conn.commit()
    conn.close()


init_db()


def generate_pdf_report(record):
    os.makedirs("static/reports", exist_ok=True)

    pdf_path = f"static/reports/PulmoScan_Report_{record['id']}.pdf"

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    y = height - 25 * mm

    c.setFont("Helvetica-Bold", 20)
    c.drawString(20 * mm, y, "PulmoScan AI - Raport medical")

    y -= 15 * mm
    c.setFont("Helvetica", 11)
    c.drawString(20 * mm, y, f"Pacient: {record['patient_name']}")
    y -= 7 * mm
    c.drawString(
    20 * mm, y, f"Vârsta: {record['patient_age']} | Sex: {record['patient_sex']}")
    
    y -= 7 * mm
    c.drawString(20 * mm, y, f"Data: {record['created_at']}")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Informatii parinte / tutore")

    y -= 7 * mm
    c.setFont("Helvetica", 11)
    c.drawString(20 * mm, y, f"Tutore: {record['guardian_name']}")

    y -= 7 * mm
    c.drawString(20 * mm, y, f"Relatie: {record['relationship']}")

    y -= 7 * mm
    c.drawString(20 * mm, y, f"E-mail tutore: {record['guardian_email']}")

    y -= 10 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Note clinice")

    y -= 7 * mm
    c.setFont("Helvetica", 11)
    c.drawString(20 * mm, y, f"{record['clinical_notes']}")

    y -= 15 * mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, y, f"Diagnostic AI: {record['prediction']}")

    y -= 8 * mm
    c.setFont("Helvetica", 11)
    c.drawString(20 * mm, y, f"Nivel de încredere: {record['confidence']}%")
    y -= 7 * mm
    c.drawString(20 * mm, y, f"Probabilitate normal: {record['prob_normal']}%")
    y -= 7 * mm
    c.drawString(20 * mm, y, f"Probabilitate viral: {record['prob_viral']}%")
    y -= 7 * mm
    c.drawString(20 * mm, y, f"Probabilitate bacterian: {record['prob_bacterial']}%")

    y -= 15 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Radiografie originala:")

    original_path = record["original_path"]

    if original_path and os.path.exists(original_path):
        y -= 80 * mm
        c.drawImage(
            original_path,
            20 * mm,
            y,
            width=75 * mm,
            height=75 * mm,
            preserveAspectRatio=True,
            mask="auto",
        )

    c.setFont("Helvetica-Bold", 12)
    c.drawString(110 * mm, y + 80 * mm, "Rezultat Grad-CAM:")

    gradcam_path = record["gradcam_path"]

    if gradcam_path and os.path.exists(gradcam_path):
        c.drawImage(
            gradcam_path,
            110 * mm,
            y,
            width=75 * mm,
            height=75 * mm,
            preserveAspectRatio=True,
            mask="auto",
        )

    y -= 15 * mm
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(
        20 * mm,
        y,
        "Acest raport a fost generat automat cu ajutorul unui model de inteligenta artificiala si trebuie interpretat în context clinic de catre un medic ",
    )

    c.save()
    return pdf_path


def template_context():
    return {
        "current_user": session.get("user", "admin"),
        "current_role": session.get("role", "Administrator"),
    }


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = user["username"]
            session["role"] = user["role"]
            log_action("AUTENTIFICARE", f"Utilizatorul {username} s-a autentificat.")
            return redirect(url_for("home"))

        return render_template("login.html", error="Username sau parolă incorectă.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    user = session.get("user", "unknown")
    session.clear()
    log_action("DECONECTARE", f"Utilizatorul {user} s-a deconectat.")
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    alerts = conn.execute("""
        SELECT * FROM history
        WHERE prediction IN ('BACTERIAL', 'VIRAL', 'INCONCLUSIVE')
        ORDER BY id DESC
        LIMIT 3
    """).fetchall()

    stats = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN prediction='NORMAL' THEN 1 ELSE 0 END) as normal,
            SUM(CASE WHEN prediction='VIRAL' THEN 1 ELSE 0 END) as viral,
            SUM(CASE WHEN prediction='BACTERIAL' THEN 1 ELSE 0 END) as bacterial,
            SUM(CASE WHEN prediction='INCONCLUSIVE' THEN 1 ELSE 0 END) as inconclusive
        FROM history
    """).fetchone()

    conn.close()

    if request.method == "POST":
        try:
            name = request.form.get("patient_name")
            guardian_name = request.form.get("guardian_name", "")
            guardian_email = request.form.get("guardian_email", "")
            relationship = request.form.get("relationship", "")

            email = guardian_email
            age = request.form.get("patient_age")
            sex = request.form.get("patient_sex")
            clinical_notes = request.form.get("clinical_notes", "")
            file = request.files.get("file")

            if not file or file.filename == "":
                return render_template(
                    "index.html",
                    result={"success": False, "error": "Nu ai încărcat nicio imagine."},
                    alerts=alerts,
                    stats=stats,
                    **template_context(),
                )

            if not allowed_file(file.filename):
                return render_template(
                    "index.html",
                    result={
                        "success": False,
                        "error": "Format invalid. Folosește PNG, JPG sau JPEG.",
                    },
                    alerts=alerts,
                    stats=stats,
                    **template_context(),
                )

            try:
                age_int = int(age)

                if age_int < 0 or age_int > 18:
                    return render_template(
                        "index.html",
                        result={
                            "success": False,
                            "error": "Sistemul PulmoScan AI acceptă doar pacienți pediatrici, cu vârsta între 0 și 18 ani."
                        },
                        alerts=alerts,
                        stats=stats,
                        **template_context(),
                    )

            except:
                return render_template(
                    "index.html",
                    result={
                        "success": False,
                        "error": "Vârsta pacientului trebuie să fie un număr valid."
                    },
                    alerts=alerts,
                    stats=stats,
                    **template_context(),
                )

            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{filename}"

            original_path = os.path.join(
                app.config["UPLOAD_FOLDER"], filename
            ).replace("\\", "/")

            gradcam_path = os.path.join(
                app.config["GRADCAM_FOLDER"], "gradcam_" + filename
            ).replace("\\", "/")

            
            file.save(original_path)

            is_valid_xray, validation_error = basic_xray_validation(original_path)

            if not is_valid_xray:
                os.remove(original_path)
                return render_template(
                    "index.html",
                    result={
                        "success": False,
                        "error": validation_error
                    },
                    alerts=alerts,
                    stats=stats,
                    **template_context(),
                )

            ai_data = analyze_image(original_path, gradcam_path)
            probs = ai_data["probabilities"]

            prob_normal = round(probs["NORMAL"] * 100, 2)
            prob_viral = round(probs["VIRAL"] * 100, 2)
            prob_bacterial = round(probs["BACTERIAL"] * 100, 2)

            prob_values = [prob_normal, prob_viral, prob_bacterial]
            sorted_probs = sorted(prob_values, reverse=True)

            max_confidence = sorted_probs[0]
            difference_between_top_two = sorted_probs[0] - sorted_probs[1]

            if max_confidence < 70 or difference_between_top_two < 20:
                prediction = "INCONCLUSIVE"
                confidence = max_confidence
            else:
                prediction = ai_data["pred_label"]
                confidence = round(ai_data["confidence"] * 100, 2)

            conn = get_db()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO history (
                    patient_name, guardian_name, guardian_email, relationship,
                    patient_email, patient_age, patient_sex, prediction,
                    confidence, original_path, gradcam_path, created_at,
                    prob_normal, prob_viral, prob_bacterial, clinical_notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name, guardian_name, guardian_email, relationship,
                    email, age, sex, prediction, confidence,
                    original_path, gradcam_path,
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                    prob_normal, prob_viral, prob_bacterial, clinical_notes,
                ),
            )

            record_id = cursor.lastrowid
            conn.commit()
            conn.close()

            log_action(
                "ANALIZĂ_RADIOGRAFIE",
                f"Pacientul {name} a fost analizat. Diagnostic: {prediction}, nivel de încredere: {confidence}%."
            )
            if prediction == "NORMAL":
                recommendation = (
                      "Nu au fost identificate semne radiologice sugestive pentru pneumonie. "
                       "Se recomandă corelarea cu tabloul clinic dacă simptomele persistă."
                )

            elif prediction == "VIRAL":
                recommendation = (
                    "Constatările imagistice sunt sugestive pentru pneumonie virală. "
                    "Se recomandă corelarea cu simptomele pacientului și evaluarea medicală de specialitate."
                )

            elif prediction == "BACTERIAL":
                recommendation = (
                     "Constatările imagistice sunt sugestive pentru pneumonie bacteriană. "
    "Se recomandă evaluare medicală promptă și investigații suplimentare."
                )

            else:
                recommendation = (
                     "Modelul de inteligență artificială nu a putut stabili un diagnostic cu un nivel suficient de încredere. "
    "Se recomandă evaluare clinică suplimentară."
                )

            result = {
                "success": True,
                "record_id": record_id,
                "prediction": prediction,
                "confidence": confidence,
                "recommendation": recommendation,
                "patient_email": email,
                "original_url": original_path,
                "gradcam_url": gradcam_path,
                "probs": {
                    "NORMAL": prob_normal,
                    "VIRAL": prob_viral,
                    "BACTERIAL": prob_bacterial,
                },
            }
            result = {
                "success": True,
                "record_id": record_id,
                "prediction": prediction,
                "confidence": confidence,
                "patient_email": email,
                "original_url": original_path,
                "gradcam_url": gradcam_path,
                "probs": {
                    "NORMAL": prob_normal,
                    "VIRAL": prob_viral,
                    "BACTERIAL": prob_bacterial,
                },
            }

            return render_template(
                "index.html",
                result=result,
                alerts=alerts,
                stats=stats,
                **template_context(),
            )

        except Exception as e:
            return render_template(
                "index.html",
                result={"success": False, "error": str(e)},
                alerts=alerts,
                stats=stats,
                **template_context(),
            )

    return render_template(
        "index.html",
        alerts=alerts,
        stats=stats,
        **template_context(),
    )
@app.route("/history")
def history():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    records = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    conn.close()

    return render_template("history.html", records=records, **template_context())


@app.route("/report/<int:record_id>")
def report(record_id):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    record = conn.execute("SELECT * FROM history WHERE id = ?", (record_id,)).fetchone()
    conn.close()

    if not record:
        return "Raportul nu a fost găsit.", 404

    return render_template("report.html", record=record, **template_context())


@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    if "user" not in session:
        return redirect(url_for("login"))

    result = None

    if request.method == "POST":
        symptoms = request.form.getlist("symptoms")

        bacterial_rules = {
            "febra_mare": 2,
            "frisoane": 2,
            "sputa_galben_verzuie": 3,
            "durere_toracica": 2,
            "debut_brusc": 2,
            "tahipnee": 2,
            "dispnee": 3,
            "alimentatie_redusa": 1,
        }

        viral_rules = {
            "febra_moderata": 2,
            "tuse_seaca": 2,
            "nas_infundat": 2,
            "oboseala": 1,
            "debut_lent": 2,
            "wheezing": 2,
            "iritabilitate": 1,
            "alimentatie_redusa": 1,
        }

        bacterial_score = sum(bacterial_rules.get(s, 0) for s in symptoms)
        viral_score = sum(viral_rules.get(s, 0) for s in symptoms)

        if bacterial_score > viral_score:
            suggestion = "Simptomele selectate sunt mai sugestive pentru o posibilă pneumonie bacteriană. Rezultatul trebuie corelat cu radiografia și validat de medic."
        elif viral_score > bacterial_score:
            suggestion = "Simptomele selectate sunt mai sugestive pentru o posibilă pneumonie virală sau atipică. Rezultatul trebuie corelat cu radiografia și validat de medic."
        else:
            suggestion = "Scorurile sunt apropiate. Este necesară corelarea cu radiografia, datele clinice și evaluarea medicului."

        result = {
            "symptoms": symptoms,
            "bacterial_score": bacterial_score,
            "viral_score": viral_score,
            "suggestion": suggestion,
        }

    return render_template("quiz.html", result=result, **template_context())


@app.route("/logs")
def logs():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    logs_data = conn.execute("SELECT * FROM logs ORDER BY id DESC").fetchall()
    conn.close()

    return render_template("logs.html", logs=logs_data, **template_context())


@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    conn.execute("DELETE FROM logs")
    conn.commit()
    conn.close()

    log_action("JURNAL_ȘTERS", "Jurnalul de activitate a fost șters de administrator.")
    return redirect(url_for("logs"))

@app.route("/export_csv")
def export_csv():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()
    records = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    conn.close()

    output = StringIO()
    output.write("\ufeff")  # ajută Excel să citească UTF-8 corect

    writer = csv.writer(output, delimiter=";")

    writer.writerow([
        "Record ID",
        "Patient Name",
        "Patient Age",
        "Patient Sex",
        "Parent / Guardian Name",
        "Relationship",
        "Guardian Email",
        "Prediction",
        "Confidence",
        "Prob Normal",
        "Prob Viral",
        "Prob Bacterial",
        "Clinical Notes",
        "Created At"
    ])

    for r in records:
        writer.writerow([
            r["id"],
            r["patient_name"],
            r["patient_age"],
            r["patient_sex"],
            r["guardian_name"],
            r["relationship"],
            r["guardian_email"],
            r["prediction"],
            f"{r['confidence']:.2f} %",
            f"{r['prob_normal']:.2f} %",
            f"{r['prob_viral']:.2f} %",
            f"{r['prob_bacterial']:.2f} %",
            r["clinical_notes"],
            r["created_at"]
        ])

    log_action("EXPORT_CSV", "Istoricul pacienților a fost exportat în format CSV.")

    return Response(
        output.getvalue(),
        mimetype="text/csv; charset=utf-8-sig",
        headers={
            "Content-Disposition": "attachment; filename=pulmoscan_history.csv"
        }
    )
@app.route("/send_email/<int:record_id>", methods=["POST"])
def send_email(record_id):
    if "user" not in session:
        return jsonify(success=False, error="Unauthorized"), 401
    conn = get_db()
    record = conn.execute("SELECT * FROM history WHERE id = ?", (record_id,)).fetchone()
    conn.close()

    if not record or not record["patient_email"]:
        return jsonify(success=False, error="Pacientul nu are email."), 400

    try:
        msg = MIMEMultipart()
        msg["Subject"] = "Rezultat analiză medicală - PulmoScan AI"
        msg["From"] = app.config["SMTP_EMAIL"]
        msg["To"] = record["patient_email"]

        guardian_name = record["guardian_name"] if record["guardian_name"] else "părinte/tutore"

        body = f"""
                    Bună ziua, {guardian_name},

    Analiza radiografiei toracice pentru pacientul {record["patient_name"]} a fost finalizată.

    Rezultatul preliminar AI este: {record["prediction"]}
    Nivel de încredere: {record["confidence"]}%

    Raportul complet este atașat acestui email în format PDF.

    Vă rugăm să interpretați rezultatul împreună cu un medic specialist. Acest raport este generat de un sistem AI și nu înlocuiește diagnosticul medical.

    Cu respect,
    PulmoScan AI
    """

        msg.attach(MIMEText(body, "plain"))

        pdf_path = generate_pdf_report(record)

        with open(pdf_path, "rb") as f:
            pdf_attachment = MIMEApplication(f.read(), _subtype="pdf")
            pdf_attachment.add_header(
                "Content-Disposition",
                "attachment",
                filename=os.path.basename(pdf_path)
            )
            msg.attach(pdf_attachment)

        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(app.config["SMTP_EMAIL"], app.config["SMTP_PASSWORD"])
        server.sendmail(
            app.config["SMTP_EMAIL"],
            record["patient_email"],
            msg.as_string()
        )
        server.quit()

        log_action(
            "EMAIL_TRIMIS",
              f"Raportul a fost trimis către {record['patient_email']} pentru fișa #{record_id}."
        )

        return jsonify(success=True)

    except Exception as e:
        log_action("EROARE_EMAIL", str(e))
        return jsonify(success=False, error=str(e)), 500


@app.route("/delete_record/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db()

    record = conn.execute("SELECT * FROM history WHERE id = ?", (record_id,)).fetchone()

    if record:
        for path_key in ["original_path", "gradcam_path"]:
            file_path = record[path_key]
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

        conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
        conn.commit()
        log_action("ȘTERGERE_FIȘĂ",f"A fost ștearsă fișa pacientului #{record_id}."
        )

    conn.close()
    return redirect(url_for("history"))


@app.route("/delete_selected_records", methods=["POST"])
def delete_selected_records():
    if "user" not in session:
        return redirect(url_for("login"))

    selected_ids = request.form.getlist("selected_records")
    conn = get_db()

    for record_id in selected_ids:
        record = conn.execute(
            "SELECT * FROM history WHERE id = ?", (record_id,)
        ).fetchone()

        if record:
            for path_key in ["original_path", "gradcam_path"]:
                file_path = record[path_key]
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)

            conn.execute("DELETE FROM history WHERE id = ?", (record_id,))

    conn.commit()
    conn.close()

    log_action("ȘTERGERE_FIȘE", f"Au fost șterse fișele selectate: {selected_ids}.")
    
    return redirect(url_for("history"))


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )