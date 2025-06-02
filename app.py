import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from motor_inferencia import motor_inferencia_ponderado

# Configuración de la app
st.set_page_config(page_title="🍑 Sistema Experto Duraznero", layout="centered")
st.title("🍑 Sistema Experto para Enfermedades del Duraznero")

# Mapeo de equivalencias entre nombres del modelo y reglas
EQUIVALENCIAS = {
    "Mochedumbre": "Monilia",
    "Pulgones": "Áfidos",
    "Taladro": "Cancro bacteriano",
    "Oidio": "Oídio"
}

# Lista de enfermedades relevantes (de las reglas)
ENFERMEDADES_RELEVANTES = ["Oídio", "Áfidos", "Cancro bacteriano", "Monilia", "Deficiencia nutricional", "Sano"]

# Cargar modelo de IA solo una vez por sesión
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model('modelo_durazno.h5')

model = cargar_modelo()

# Clases del modelo original
class_names_original = [
    'Agalla de corona', 'Arañuela roja', 'Mochedumbre', 
    'Mosca de la fruta', 'Oidio', 'Pulgones', 
    'Sano', 'Taladro', 'Viruela'
]

# Función para filtrar y adaptar las predicciones
def filtrar_predicciones(prediccion, clases_originales):
    # Convertir a nombres de reglas
    clases_mapeadas = []
    for clase in clases_originales:
        if clase in EQUIVALENCIAS:
            clases_mapeadas.append(EQUIVALENCIAS[clase])
        else:
            clases_mapeadas.append(clase)
    
    # Filtrar solo enfermedades relevantes
    resultados = []
    for i, prob in enumerate(prediccion[0]):
        nombre_clase = clases_mapeadas[i]
        if nombre_clase in ENFERMEDADES_RELEVANTES:
            # Si es "Sano", le damos un tratamiento especial
            if nombre_clase == "Sano":
                # Solo mostramos "Sano" si tiene alta probabilidad
                if prob > 0.7:  # Umbral de 70% para considerar sano
                    resultados.append({
                        "enfermedad": nombre_clase,
                        "probabilidad": float(prob),
                        "clase_original": clases_originales[i]
                    })
            else:
                resultados.append({
                    "enfermedad": nombre_clase,
                    "probabilidad": float(prob),
                    "clase_original": clases_originales[i]
                })
    
    # Ordenar por probabilidad descendente
    resultados.sort(key=lambda x: x["probabilidad"], reverse=True)
    # Si "Sano" tiene la mayor probabilidad y es >50%, mostramos solo ese
    if resultados and resultados[0]["enfermedad"] == "Sano" and resultados[0]["probabilidad"] > 0.5:
        return [resultados[0]]
    
    return resultados
# Sidebar para elegir método de diagnóstico
opcion = st.sidebar.radio(
    "Selecciona el método de diagnóstico:",
    ("Diagnóstico por Imagen", "Diagnóstico por Formulario", "Comparar ambos")
)

# ----------------------- Diagnóstico por Imagen -----------------------
if opcion == "Diagnóstico por Imagen":
    st.header("🔍 Diagnóstico por Imagen")
    uploaded_file = st.file_uploader("Sube una imagen del duraznero", type=["jpg", "jpeg", "png", "jfif"])

    if uploaded_file is not None:
        original_img = Image.open(uploaded_file)
        # Convertir imagen a RGB si tiene canal alpha (RGBA)
        if original_img.mode == 'RGBA':
            original_img = original_img.convert('RGB')
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🌱 Imagen Original")
            st.image(original_img, use_container_width=True)
        
        # Preprocesar la imagen
        img_resized = original_img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        with col2:
            st.subheader("🔎 Imagen Preprocesada")
            st.image(img_resized, use_container_width=True)
        
        # Predicción
        prediction = model.predict(img_array)
        resultados_filtrados = filtrar_predicciones(prediction, class_names_original)
        
        if not resultados_filtrados:
            st.warning("No se detectaron enfermedades relevantes en la imagen.")
            st.stop()
        
        st.markdown("---")
        
        # Mensaje especial si está sano
        if resultados_filtrados[0]['enfermedad'] == "Sano":
            st.markdown(f"""
            <div style="background-color:#e8f5e9;padding:15px;border-radius:10px;margin:10px 0">
                <h3 style="color:#2e7d32;text-align:center;">🌿 DIAGNÓSTICO PRINCIPAL 🌿</h3>
                <h2 style="color:#1b5e20;text-align:center;">¡Planta Sana!</h2>
                <h3 style="color:#2e7d32;text-align:center;">📊 CONFIANZA: {resultados_filtrados[0]['probabilidad']*100:.1f}% 📊</h3>
                <p style="text-align:center;">No se detectaron enfermedades significativas en la planta.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color:#f0f2f6;padding:15px;border-radius:10px;margin:10px 0">
                <h3 style="color:#2e7d32;text-align:center;">🌿 DIAGNÓSTICO PRINCIPAL 🌿</h3>
                <h2 style="color:#1b5e20;text-align:center;">{resultados_filtrados[0]['enfermedad']}</h2>
                <h3 style="color:#2e7d32;text-align:center;">📊 CONFIANZA: {resultados_filtrados[0]['probabilidad']*100:.1f}% 📊</h3>
            </div>
            """, unsafe_allow_html=True)

        # Diagnóstico probabilístico
        st.subheader("🔥 Diagnóstico Probabilístico")
        for resultado in resultados_filtrados:
            st.write(f"- {resultado['enfermedad']}: {resultado['probabilidad']*100:.2f}%")

        # Visualización de gráficas
        st.subheader("📈 Gráfico de Confianza")
        fig, ax = plt.subplots(figsize=(8,4))
        enfermedades = [r['enfermedad'] for r in resultados_filtrados]
        probabilidades = [r['probabilidad']*100 for r in resultados_filtrados]
        
        # Color diferente para "Sano"
        colores = ['lightgreen' if enf == 'Sano' else 'lightcoral' for enf in enfermedades]
        
        ax.bar(enfermedades, probabilidades, color=colores)
        ax.set_ylabel('Confianza (%)')
        ax.set_xticklabels(enfermedades, rotation=45, ha='right')
        st.pyplot(fig)
# ----------------------- Diagnóstico por Formulario -----------------------

# ----------------------- Diagnóstico por Formulario -----------------------
elif opcion == "Diagnóstico por Formulario":
    st.header("📝 Diagnóstico por Formulario")
    st.markdown("""
    <style>
    .symptom-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    .symptom-title {
        font-weight: bold;
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### Por favor, marca los síntomas que observas en tu planta:")

    # Síntomas con imágenes y descripciones
    sintomas_info = [
        {
            "key": "manchas_hojas",
            "label": "Manchas en las hojas",
            "description": "Presencia de manchas circulares o irregulares en las hojas",
            "image": "images/484120_1_En_14_Fig16_HTML.jpg",
            "recomendacion": "Fungicidas con: Clorotalonil, Mancozeb, o Fungicidas a base de Cobre",
            "recomendacion_img": "images/clorotalonil.jpg"
        },
        {
            "key": "polvo_blanco",
            "label": "Polvo blanco en hojas/tallos",
            "description": "Aspecto de polvo o ceniza blanca sobre la superficie de la planta",
            "image": "images/360_F_1379881865_mwP0pB77fUCIDRtxUfpIXdm7R2blLKBm.jpg",
            "recomendacion": "Fungicidas con: Azufre (polvo mojable), Azoxystrobin o Miclobutanil",
            "recomendacion_img": "images/miclobutanil.jpg"
        },
        {
            "key": "hojas_amarillas",
            "label": "Amarillamiento de hojas",
            "description": "Hojas que pierden su color verde y se vuelven amarillas",
            "image": "images/Amarrillas.jpg",
            "recomendacion": "Fertilizantes: Aplicación de un fertilizante balanceado (NPK)",
            "recomendacion_img": "images/fertilizantebalanceado(npk).jpg"
        },
        {
            "key": "hojas_enrolladas",
            "label": "Hojas enrolladas o deformadas",
            "description": "Hojas que se curvan, enrollan o presentan deformaciones",
            "image": "images/Enrrolladas.jpg",
            "recomendacion": "Fungicidas a base de Cobre o Clorotalonil o Captan durante el ciclo",
            "recomendacion_img": "images/Fungicida-AzufrePanteraMojable.jpg"
        },
        {
            "key": "plagas",
            "label": "Presencia de insectos visibles",
            "description": "Observación de pequeños insectos en hojas o tallos",
            "image": "images/909_0.jpeg",  # Usando imagen de agujeros como referencia visual
            "recomendacion": "Insecticidas de amplio espectro: Piretrinas/Piretroides",
            "recomendacion_img": "images/piretrina.jpg"
        },
        {
            "key": "hojas_agujeros",
            "label": "Agujeros en las hojas",
            "description": "Hojas con perforaciones o mordeduras visibles",
            "image": "images/Agujero1.jpg",
            "recomendacion": "Piretrinas/Piretroides, o Bacillus thuringiensis",
            "recomendacion_img": "images/BACILLUSTHURINGIENSIS.jpg"
        },
        {
            "key": "ramas_secas",
            "label": "Ramas secas o marchitas",
            "description": "Ramas que pierden vitalidad y se secan prematuramente",
            "image": "images/Secas.jpg",
            "recomendacion": "Captan",
            "recomendacion_img": "images/captan.jpg"
        },
        {
            "key": "corteza_rajada",
            "label": "Corteza agrietada o exudados",
            "description": "Grietas en la corteza o secreción de goma/resina",
            "image": "images/Gomoso.jpg",
            "recomendacion": "Oxicloruro de Cobre",
            "recomendacion_img": "images/axiclorurodecobre.jpg"
        },
        {
            "key": "muerte_planta",
            "label": "Muerte de partes de la planta",
            "description": "Partes de la planta que mueren repentinamente",
            "image": "images/images (3).jfif",  # Usando imagen de ramas secas como referencia
            "recomendacion": "Tebuconazol",
            "recomendacion_img": "images/Tebuconazol.jpg"
        },
        {
            "key": "frutos_podridos",
            "label": "Podredumbre en frutos",
            "description": "Frutos con manchas, moho o descomposición",
            "image": "images/Prodrido.jpg",
            "recomendacion": "Boscalid + Pyraclostrobin, Tebuconazol, o Captan",
            "recomendacion_img": "images/Boscalid.jpg"
        },
        {
            "key": "olor_raro",
            "label": "Olor desagradable",
            "description": "Olores anormales provenientes de la planta o frutos",
            "image": "images/images (7).jfif",
            "recomendacion": "Tebuconazol, Captan",
            "recomendacion_img": "images/Tebuconazol.jpg"
        },
        {
            "key": "presencia_hongos",
            "label": "Hongos visibles",
            "description": "Presencia de estructuras fúngicas en la planta",
            "image": "images/hongos1.jpg",
            "recomendacion": "Clorotalonil, Mancozeb",
            "recomendacion_img": "images/clorotalonil.jpg"
        },
        {
            "key": "crecimiento_lento",
            "label": "Crecimiento atrofiado",
            "description": "Desarrollo más lento de lo normal en la planta",
            "image": "images/FIg-3.jpg",  # Usando imagen de hojas amarillas como referencia
            "recomendacion": "Fertilizante balanceado",
            "recomendacion_img": "images/fertilizantebalanceado(npk).jpg"
        },
        {
            "key": "caida_frutos",
            "label": "Caída prematura de frutos",
            "description": "Frutos que caen antes de madurar completamente",
            "image": "images/Prodrido.jpg",  # Usando imagen de frutos podridos como referencia
            "recomendacion": "Fertilizante foliar o al suelo",
            "recomendacion_img": "images/fertilizantefoliaR.jpg"
        }
    ]


    hechos_usuario = {}
    for sintoma in sintomas_info:
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    st.image(sintoma["image"], width=150)
                except:
                    st.warning("Imagen no encontrada")
            with col2:
                st.markdown(f"""
                <div class="symptom-item">
                    <div class="symptom-title">{sintoma['label']}</div>
                    <div style="font-size: 0.9em; color: #555;">{sintoma['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                hechos_usuario[sintoma["key"]] = st.checkbox(
                    f"Confirmar presencia de: {sintoma['label'].lower()}",
                    key=sintoma["key"],
                )

    if st.button("🩺 Realizar Diagnóstico", type="primary"):
        # Ponderaciones para cada síntoma
        sintomas_ponderados = {
            "manchas_hojas": 0.7,
            "polvo_blanco": 0.8,
            "hojas_amarillas": 0.5,
            "hojas_enrolladas": 0.4,
            "plagas": 0.6,
            "hojas_agujeros": 0.3,
            "ramas_secas": 0.4,
            "corteza_rajada": 0.5,
            "muerte_planta": 1.0,
            "frutos_podridos": 0.6,
            "olor_raro": 0.3,
            "hongos_visibles": 0.7,
            "crecimiento_lento": 0.4,
            "caida_frutos": 0.5
        }
        
        # Llama al motor de inferencia ponderado
        diagnostico, log = motor_inferencia_ponderado(hechos_usuario, sintomas_ponderados)
        
        st.subheader("📋 Resultados del Diagnóstico")
        
        if not diagnostico or all(d["porcentaje"] == 0 for d in diagnostico):
            st.info("ℹ️ No se detectaron enfermedades con los síntomas proporcionados.")
            st.stop()
            
        # Filtrar diagnósticos con porcentaje > 0
        diagnostico_filtrado = [d for d in diagnostico if d["porcentaje"] > 0]
        
        # Ordenar por porcentaje descendente
        diagnostico_filtrado.sort(key=lambda x: x["porcentaje"], reverse=True)
        
        # Mostrar resultados con imágenes de recomendaciones
        for resultado in diagnostico_filtrado:
            porcentaje = resultado["porcentaje"] * 100
            
            # Definir colores según el nivel
            if porcentaje >= 70:
                color = "#4caf50"  # Verde
                emoji = "✅"
                nivel = "Alta probabilidad"
            elif porcentaje >= 40:
                color = "#ff9800"  # Naranja
                emoji = "⚠️"
                nivel = "Probabilidad media"
            else:
                color = "#f44336"  # Rojo
                emoji = "🔍"
                nivel = "Baja probabilidad"
            
            # Mostrar diagnóstico
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {color}20; 
                        border-left: 4px solid {color}; margin: 10px 0;">
                <div style="font-size: 1.1em; font-weight: bold;">
                    {emoji} {resultado['enfermedad']}: {porcentaje:.1f}% ({nivel})
                </div>
                <div style="margin-top: 5px;">
                    <strong>Síntomas detectados:</strong> {', '.join(resultado['sintomas_presentes']).replace('_', ' ')}
                </div>
                <div style="margin-top: 5px;">
                    <strong>Diagnóstico:</strong> {resultado['diagnostico'].capitalize()}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar recomendación con imagen
            st.subheader(f"💡 Recomendación para {resultado['enfermedad']}")
            
            # Buscar si hay una recomendación específica para esta enfermedad
            recomendacion_especifica = None
            recomendacion_img_especifica = None
            
            # Mapeo de enfermedades a claves de recomendación
            recomendaciones_especiales = {
                "Oídio": ("Fungicidas con: Azufre, Miclobutanil o Trifloxistrobin", "miclobutanil.jpg"),
                "Áfidos": ("Insecticidas sistémicos como Imidacloprid o aceites hortícolas", "piretrina.jpg"),
                "Cancro bacteriano": ("Eliminación de ramas afectadas, bactericidas como cobre", "oxicloruro_cobre.jpg"),
                "Monilia": ("Fungicidas como Tebuconazol, Boscalid + Pyraclostrobin", "tebuconazol.jpg"),
                "Deficiencia nutricional": ("Aplicar fertilizantes NPK y análisis de suelo", "fertilizantefoliaR.jpg")
            }
            
            if resultado['enfermedad'] in recomendaciones_especiales:
                recomendacion_especifica, img = recomendaciones_especiales[resultado['enfermedad']]
                try:
                    recomendacion_img_especifica = f"images/{img}"
                except:
                    pass
            
            # Mostrar recomendación en columnas (imagen + texto)
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if recomendacion_img_especifica:
                    try:
                        st.image(recomendacion_img_especifica, width=200)
                    except:
                        st.warning(f"No se pudo cargar la imagen: {recomendacion_img_especifica}")
                else:
                    # Mostrar imagen genérica de recomendación
                    try:
                        st.image("images/generico.jpg", width=200)
                    except:
                        pass
            
            with col2:
                if recomendacion_especifica:
                    st.write(recomendacion_especifica)
                else:
                    st.write(resultado['diagnostico'])
                
                # Mostrar síntomas específicos que llevaron a este diagnóstico
                st.markdown("**Síntomas relacionados:**")
                for sintoma in resultado['sintomas_presentes']:
                    # Buscar la descripción del síntoma
                    desc_sintoma = next((s["description"] for s in sintomas_info if s["key"] == sintoma), sintoma)
                    st.write(f"- {desc_sintoma}")

        # Visualización de gráficas
        st.subheader("📊 Resumen Gráfico")
        fig, ax = plt.subplots(figsize=(10, 5))

        enfermedades = [d["enfermedad"] for d in diagnostico_filtrado]
        porcentajes = [d["porcentaje"] * 100 for d in diagnostico_filtrado]
        colores = []

        for p in porcentajes:
            if p >= 70:
                colores.append("#4caf50")  # Verde
            elif p >= 40:
                colores.append("#ff9800")  # Naranja
            else:
                colores.append("#f44336")  # Rojo

        bars = ax.bar(enfermedades, porcentajes, color=colores)
        ax.set_ylabel('Porcentaje de coincidencia (%)')
        ax.set_xlabel('Enfermedades')
        ax.set_title('Correlación con enfermedades conocidas')

        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

        st.pyplot(fig)

        # ☁️ Lógica difusa para riesgo ambiental
        st.subheader("🌡 Nivel de riesgo ambiental (difuso)")
        
        # Calcular riesgo basado en síntomas críticos
        riesgo = 0
        if hechos_usuario.get("hongos_visibles", False):
            riesgo += 0.4
        if hechos_usuario.get("olor_raro", False):
            riesgo += 0.3
        if hechos_usuario.get("corteza_rajada", False):
            riesgo += 0.2
        if hechos_usuario.get("frutos_podridos", False):
            riesgo += 0.3
        if hechos_usuario.get("muerte_planta", False):
            riesgo += 0.5  # Síntoma muy grave
            
        # Normalizar el riesgo a máximo 1.0
        riesgo = min(riesgo, 1.0)
        
        # Determinar nivel de riesgo
        if riesgo >= 0.7:
            nivel_riesgo = "Alto"
            color_riesgo = "#f44336"
            icono = "🔥"
        elif riesgo >= 0.4:
            nivel_riesgo = "Medio"
            color_riesgo = "#ff9800"
            icono = "⚠️"
        else:
            nivel_riesgo = "Bajo"
            color_riesgo = "#4caf50"
            icono = "🌱"
        
        # Mostrar resultado con barra de progreso
        st.markdown(f"""
        <div style="margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span>Nivel: <strong style="color: {color_riesgo};">{icono} {nivel_riesgo}</strong></span>
                <span>{riesgo*100:.0f}%</span>
            </div>
            <div style="height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden;">
                <div style="height: 100%; width: {riesgo*100}%; background: {color_riesgo}; 
                    border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explicación del riesgo
        with st.expander("ℹ️ ¿Qué significa este nivel de riesgo?"):
            st.markdown("""
            - **Alto (70-100%)**: Condiciones ambientales muy favorables para la propagación de enfermedades. 
              Se recomienda acción inmediata y posible aislamiento de la planta afectada.
              
            - **Medio (40-69%)**: Riesgo moderado. Las condiciones podrían favorecer el desarrollo de 
              enfermedades si no se toman medidas preventivas.
              
            - **Bajo (0-39%)**: Riesgo mínimo. Las condiciones ambientales no son propicias para la 
              rápida propagación de enfermedades.
            """)
# ----------------------- Comparar ambos -----------------------
elif opcion == "Comparar ambos":
    st.header("🔬 Comparar Diagnóstico por Imagen y Formulario")

    # Imagen
    uploaded_file = st.file_uploader("Sube una imagen del duraznero", type=["jpg", "jpeg", "png", "jfif"], key="img_compare")

    # Síntomas
    st.markdown("### 📝 Marca los síntomas que observes:")
    
    sintomas_info = [
        {"key": "manchas_hojas", "label": "Manchas en las hojas"},
        {"key": "polvo_blanco", "label": "Polvo blanco en hojas/tallos"},
        {"key": "hojas_amarillas", "label": "Amarillamiento de hojas"},
        {"key": "hojas_enrolladas", "label": "Hojas enrolladas o deformadas"},
        {"key": "plagas", "label": "Presencia de insectos visibles"},
        {"key": "hojas_agujeros", "label": "Agujeros en las hojas"},
        {"key": "ramas_secas", "label": "Ramas secas o marchitas"},
        {"key": "corteza_rajada", "label": "Corteza agrietada o exudados"},
        {"key": "muerte_planta", "label": "Muerte de partes de la planta"},
        {"key": "frutos_podridos", "label": "Podredumbre en frutos"},
        {"key": "olor_raro", "label": "Olor desagradable"},
        {"key": "hongos_visibles", "label": "Hongos visibles"},
        {"key": "crecimiento_lento", "label": "Crecimiento atrofiado"},
        {"key": "caida_frutos", "label": "Caída prematura de frutos"}
    ]

    hechos_usuario = {}
    for sintoma in sintomas_info:
        hechos_usuario[sintoma["key"]] = st.checkbox(
            sintoma["label"],
            key=f"compare_{sintoma['key']}"
        )

    if st.button("⚡ Comparar Diagnósticos", type="primary"):
        if uploaded_file is None:
            st.error("🚨 Por favor sube una imagen para diagnóstico por imagen.")
            st.stop()

        # Diagnóstico por imagen
        original_img = Image.open(uploaded_file)
        
        # Convertir imagen a RGB si tiene canal alpha (RGBA)
        if original_img.mode == 'RGBA':
            original_img = original_img.convert('RGB')
            
        img_resized = original_img.resize((128, 128))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        try:
            prediction = model.predict(img_array)
            resultados_img = filtrar_predicciones(prediction, class_names_original)
            
            if not resultados_img:
                st.error("No se detectaron enfermedades relevantes en la imagen.")
                st.stop()
                
            pred_enfermedad = resultados_img[0]["enfermedad"]
            pred_confianza = resultados_img[0]["probabilidad"]
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")
            st.stop()

        # Diagnóstico por formulario
        sintomas_ponderados = {
            "manchas_hojas": 0.7,
            "polvo_blanco": 0.8,
            "hojas_amarillas": 0.5,
            "hojas_enrolladas": 0.4,
            "plagas": 0.6,
            "hojas_agujeros": 0.3,
            "ramas_secas": 0.4,
            "corteza_rajada": 0.5,
            "muerte_planta": 1.0,
            "frutos_podridos": 0.6,
            "olor_raro": 0.3,
            "hongos_visibles": 0.7,
            "crecimiento_lento": 0.4,
            "caida_frutos": 0.5
        }
        
        diagnostico_formulario, log = motor_inferencia_ponderado(hechos_usuario, sintomas_ponderados)

        if diagnostico_formulario:
            top_formulario = max(diagnostico_formulario, key=lambda x: x["porcentaje"])
            top_nombre = top_formulario["enfermedad"]
            top_prob = top_formulario["porcentaje"]
        else:
            top_nombre = "No detectado"
            top_prob = 0

        # Mostrar resultados
        st.subheader("📊 Resultados Comparativos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖼️ Diagnóstico por Imagen")
            st.write(f"**Enfermedad detectada:** {pred_enfermedad}")
            st.write(f"**Nivel de confianza:** {pred_confianza*100:.1f}%")
            
            # Gráfico para imagen
            fig_img, ax_img = plt.subplots(figsize=(6, 4))  # Aumenté un poco el alto para mejor visualización
            enfermedades_img = [r["enfermedad"] for r in resultados_img]
            probabilidades_img = [r["probabilidad"]*100 for r in resultados_img]
            bars_img = ax_img.bar(enfermedades_img, probabilidades_img, color='#2196f3')
            ax_img.set_ylabel('Confianza (%)')
            ax_img.set_xlabel('Enfermedades')
            ax_img.set_title('Resultados por imagen')

            # Rotar etiquetas del eje X para mejor legibilidad
            plt.xticks(rotation=45, ha='right')

            # Añadir etiquetas de porcentaje encima de cada barra
            for bar in bars_img:
                height = bar.get_height()
                ax_img.text(bar.get_x() + bar.get_width()/2, height + 1,
                            f'{height:.1f}%',
                            ha='center', va='bottom')

            st.pyplot(fig_img)
        
        with col2:
            st.markdown("### 📝 Diagnóstico por Formulario")
            st.write(f"**Enfermedad detectada:** {top_nombre}")
            st.write(f"**Porcentaje de coincidencia:** {top_prob*100:.1f}%")
            
            # Gráfico para formulario (solo si hay resultados)
            if diagnostico_formulario:
                fig_form, ax_form = plt.subplots(figsize=(6, 4))
                enfermedades_form = [d["enfermedad"] for d in diagnostico_formulario if d["porcentaje"] > 0]
                porcentajes_form = [d["porcentaje"]*100 for d in diagnostico_formulario if d["porcentaje"] > 0]
                
                if enfermedades_form:  # Solo si hay datos
                    bars_form = ax_form.bar(enfermedades_form, porcentajes_form, color='#4caf50')
                    ax_form.set_ylabel('Coincidencia (%)')
                    ax_form.set_xlabel('Enfermedades')
                    ax_form.set_title('Resultados por formulario')
                    
                    # Rotar etiquetas del eje X
                    plt.xticks(rotation=45, ha='right')
                    
                    # Añadir etiquetas de porcentaje
                    for bar in bars_form:
                        height = bar.get_height()
                        ax_form.text(bar.get_x() + bar.get_width()/2, height + 1,
                                    f'{height:.1f}%',
                                    ha='center', va='bottom')
                    
                    st.pyplot(fig_form)

        # Comparación
        st.subheader("🔍 Comparación de Resultados")
        
        if pred_enfermedad == top_nombre and top_prob > 0:
            st.success(f"✅ Ambos métodos coinciden en el diagnóstico: {pred_enfermedad}")
        elif top_prob == 0:
            st.warning("⚠️ El diagnóstico por formulario no detectó enfermedades relevantes")
        else:
            st.warning(f"⚠️ Los diagnósticos no coinciden: Imagen → {pred_enfermedad} | Formulario → {top_nombre}")