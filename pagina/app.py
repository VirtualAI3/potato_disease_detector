from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from datetime import datetime
from functools import wraps
from cargar_modelos import cargar_modelos_desde_directorio
from PIL import Image
import io
import torch
import torchvision.transforms as transforms

CLASES = ["Agusanada", "Podredumbre seca", "Sano", "Sarna común", "Sarna negra"]

modelos_cargados = cargar_modelos_desde_directorio("modelos", num_clases=len(CLASES))

app = Flask(__name__)
app.secret_key = "secreto_super_seguro"

DB_PATH = "db.sqlite3"

# ---------------------- BASE DE DATOS ----------------------

def crear_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            rol TEXT NOT NULL DEFAULT 'usuario'
        )''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS actividad (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            modelo TEXT,
            inicio TEXT,
            fin TEXT,
            duracion REAL,
            FOREIGN KEY(user_id) REFERENCES usuarios(id)
        )''')
        c.execute("SELECT * FROM usuarios WHERE rol='admin'")
        if c.fetchone() is None:
            password_admin = generate_password_hash("admin123")  # Cambia esta contraseña!
            c.execute("INSERT INTO usuarios (nombre, email, password, rol) VALUES (?, ?, ?, 'admin')",
                      ("Admin", "diegoapazaphocco@gmail.com", password_admin))
crear_db()

# ---------------------- DECORADORES ----------------------

def login_requerido(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "usuario_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap

def solo_admin(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if session.get("rol") != "admin":
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrap

# ---------------------- RUTAS PRINCIPALES ----------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["POST"])
def register():
    nombre = request.form["name"]
    email = request.form["email"]
    password = generate_password_hash(request.form["password"])
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO usuarios (nombre, email, password) VALUES (?, ?, ?)", (nombre, email, password))
        return redirect(url_for("index"))
    except sqlite3.IntegrityError:
        return "Correo ya registrado", 400

@app.route("/login", methods=["POST"])
def login():
    email = request.form["email"]
    password = request.form["password"]
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, password, rol FROM usuarios WHERE email = ?", (email,))
        usuario = c.fetchone()
        if usuario and check_password_hash(usuario[1], password):
            session["usuario_id"] = usuario[0]
            session["rol"] = usuario[2]
            if usuario[2] == "admin":
                return redirect(url_for("admin"))
            else:
                return redirect(url_for("dashboard"))
    return "Credenciales incorrectas", 401

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_requerido
def dashboard():
    return render_template("dashboard.html")

@app.route("/admin")
@login_requerido
@solo_admin
def admin():
    return render_template("admin.html")

@app.route("/iniciar_sesion_deteccion", methods=["POST"])
@login_requerido
def iniciar_sesion_deteccion():
    modelo = request.form["modelo"]
    session["deteccion_viva"] = {
        "modelo": modelo,
        "inicio": datetime.now().isoformat()
    }
    return jsonify({"mensaje": "Sesión de detección iniciada"})

@app.route("/finalizar_sesion_deteccion", methods=["POST"])
@login_requerido
def finalizar_sesion_deteccion():
    data = session.pop("deteccion_viva", None)
    if not data:
        return jsonify({"mensaje": "No hay sesión activa"}), 400

    inicio = datetime.fromisoformat(data["inicio"])
    fin = datetime.now()
    duracion = (fin - inicio).total_seconds()

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO actividad (user_id, modelo, inicio, fin, duracion) VALUES (?, ?, ?, ?, ?)",
            (session["usuario_id"], data["modelo"], inicio.isoformat(), fin.isoformat(), duracion)
        )

    return jsonify({"mensaje": "Sesión de detección finalizada"})


# ---------------------- DETECCIÓN ----------------------

@app.route("/detectar", methods=["POST"])
@login_requerido
def detectar():
    modelo_nombre = request.form.get("modelo")
    imagen = request.files.get("imagen")

    if modelo_nombre is None or imagen is None:
        return jsonify({"resultado": "Faltan parámetros"}), 400

    # Obtener el modelo cargado según el nombre
    model = modelos_cargados.get(modelo_nombre)
    if model is None:
        return jsonify({"resultado": "Modelo no encontrado"}), 400

    try:
        # Leer imagen en formato PIL
        img_bytes = imagen.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Transformaciones estándar para modelos preentrenados
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # o el tamaño que tu modelo espere
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(pil_image).unsqueeze(0)  # Batch dimension

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            clase_predicha = CLASES[pred.item()]

        # Guardar actividad en DB
        inicio = datetime.now()
        fin = datetime.now()
        duracion = (fin - inicio).total_seconds()
        if "deteccion_viva" not in session:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO actividad (user_id, modelo, inicio, fin, duracion) VALUES (?, ?, ?, ?, ?)",
                    (session["usuario_id"], modelo_nombre, inicio.isoformat(), fin.isoformat(), duracion)
                )

        return jsonify({"resultado": f"Enfermedad detectada: {clase_predicha}"})

    except Exception as e:
        print("Error en detección:", e)
        return jsonify({"resultado": "Error al procesar la imagen"}), 500

# ---------------------- API ADMIN ----------------------

@app.route("/api/usuarios")
@login_requerido
@solo_admin
def api_usuarios():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT id, nombre, email, rol FROM usuarios")
        usuarios = [{"id": u[0], "nombre": u[1], "email": u[2], "rol": u[3]} for u in c.fetchall()]
    return jsonify(usuarios)

@app.route("/api/actividad")
@login_requerido
@solo_admin
def api_actividad():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            SELECT u.nombre, a.modelo, a.inicio, a.fin, a.duracion
            FROM actividad a JOIN usuarios u ON a.user_id = u.id
            ORDER BY a.id DESC
        ''')
        logs = [{"usuario": l[0], "modelo": l[1], "inicio": l[2], "fin": l[3], "duracion": round(l[4], 2)} for l in c.fetchall()]
    return jsonify(logs)

@app.route("/api/eliminar_usuario/<int:usuario_id>", methods=["DELETE"])
@login_requerido
@solo_admin
def eliminar_usuario(usuario_id):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("DELETE FROM usuarios WHERE id = ?", (usuario_id,))
        conn.commit()
    return jsonify({"mensaje": "Usuario eliminado correctamente"})


@app.route("/api/modelos")
@login_requerido
def api_modelos():
    lista_modelos = [
        {
            "id": nombre,
            "nombre": nombre.capitalize(),
            "descripcion": f"Modelo {nombre.capitalize()} para detección de enfermedades."
        }
        for nombre in modelos_cargados.keys()
    ]
    return jsonify(lista_modelos)

# ---------------------- INICIO ----------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
