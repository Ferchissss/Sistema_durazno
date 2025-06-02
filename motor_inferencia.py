from base_reglas import reglas

def motor_inferencia_ponderado(hechos_usuario, sintomas_ponderados):
    resultados = []
    log = []

    for regla in reglas:
        sintomas = regla["sintomas"]  # dict con sintoma: peso
        suma_pesos = sum(sintomas.values())
        suma_pesos_detectados = sum(peso for sintoma, peso in sintomas.items() if hechos_usuario.get(sintoma, False))
        
        porcentaje = suma_pesos_detectados / suma_pesos if suma_pesos > 0 else 0
        
        # Definís umbrales para diagnostico
        if porcentaje >= 0.7:
            diagnostico = "confirmado"
        elif porcentaje >= 0.4:
            diagnostico = "sospecha"
        else:
            diagnostico = "no detectado"

        resultados.append({
            "regla": regla["regla"],
            "enfermedad": regla["enfermedad"],
            "icono": regla["icono"],
            "diagnostico": diagnostico,
            "porcentaje": porcentaje,
            "sintomas_presentes": [s for s in sintomas if hechos_usuario.get(s, False)]
        })

        log.append(f"Regla {regla['regla']} ({regla['enfermedad']}): {porcentaje*100:.1f}% síntomas presentes → Diagnóstico: {diagnostico}")

    return resultados, log
