from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import requests
from flask_cors import CORS
from collections import Counter

app = Flask(__name__)
CORS(app)

# 🔹 Cargar modelo
model = load_model("sunlit_model_fixed.h5")
# 🔹 Clases (AJUSTA ESTO A TU MODELO REAL)
classes = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]
# 🔹 Preprocesamiento
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# 🔹 NASA POWER (clima)
def get_climate(lat, lon):
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,PRECTOT&community=AG&longitude={lon}&latitude={lat}&format=JSON"
        response = requests.get(url, timeout=10)
        data = response.json()

        params = data["properties"]["parameter"]

        temp = list(params["T2M"].values())[-1]
        humidity = list(params["RH2M"].values())[-1]
        rain = list(params["PRECTOT"].values())[-1]

        return {
            "temperature": temp,
            "humidity": humidity,
            "rain": rain
        }

    except:
        # 🔥 fallback si NASA falla
        return {
            "temperature": None,
            "humidity": None,
            "rain": None
        }

# 🔹 Análisis inteligente
def generate_analysis(pred_class, confidence, climate):

    # 🔥 Caso: no es planta o baja confianza
    if pred_class == "unknown":
        return {
            "analysis": "No se pudo identificar una planta en la imagen.",
            "climate": climate,
            "suggestions": [
                "Asegúrate de enfocar claramente una planta.",
                "Evita fondos confusos o poca iluminación."
            ]
        }

    # 🔹 Caso normal
    if pred_class == "healthy":
        analysis = "La planta se encuentra en buen estado general."
    else:
        analysis = f"Se detecta posible {pred_class} con una confianza de {confidence:.2f}."

    suggestions = []

    # 🔹 Reglas basadas en clima
    if climate["humidity"] and climate["humidity"] > 80:
        suggestions.append("Alta humedad: riesgo de hongos. Mejora ventilación.")

    if climate["temperature"] and climate["temperature"] > 30:
        suggestions.append("Temperatura elevada: aumenta frecuencia de riego.")

    # 🔹 Reglas por enfermedad
    if pred_class == "leaf_spot":
        suggestions.append("Eliminar hojas afectadas y evitar exceso de riego.")

    if pred_class == "rust":
        suggestions.append("Aplicar fungicida y evitar mojar hojas.")

    if pred_class == "powdery_mildew":
        suggestions.append("Mejorar circulación de aire y reducir sombra.")

    if not suggestions:
        suggestions.append("Mantener monitoreo regular.")

    return {
        "analysis": analysis,
        "climate": climate,
        "suggestions": suggestions
    }

# 🔥 ENDPOINT PRINCIPAL
@app.route("/predict", methods=["POST"])
def predict():
    try:
        files = request.files.getlist("images")
        lat = request.form.get("lat")
        lon = request.form.get("lon")

        if not files:
            return jsonify({"error": "No se enviaron imágenes"})

        results = []
        predictions_list = []

        # 🔹 Obtener clima una vez
        climate = get_climate(lat, lon)

        for file in files:
            try:
                image = Image.open(file).convert("RGB")
                processed = preprocess_image(image)

                prediction = model.predict(processed)
                class_index = np.argmax(prediction)
                confidence = float(np.max(prediction))
                pred_class = classes[class_index]

                # 🔥 FILTRO DE SEGURIDAD
                if confidence < 0.6:
                    pred_class = "unknown"

                predictions_list.append(pred_class)

                analysis = generate_analysis(pred_class, confidence, climate)

                results.append({
                    "prediction": pred_class,
                    "confidence": confidence,
                    "details": analysis
                })

            except Exception:
                # 🔥 No rompe todo si una imagen falla
                results.append({
                    "error": "No se pudo procesar esta imagen"
                })

        # 🔹 Resumen global
        summary = dict(Counter(predictions_list))

        return jsonify({
            "total_images": len(results),
            "summary": summary,
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# 🔹 Ruta base
@app.route("/", methods=["GET"])
def home():
    return "API funcionando correctamente"

# 🔹 Run local

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))