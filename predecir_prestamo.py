import pandas as pd
from xgboost import XGBClassifier


COLUMNAS = ["Ingresos", "Score_Credito", "Edad"]


def entrenar_modelo():
    datos_banco = {
        "Ingresos": [45000, 32000, 85000, 120000, 25000, 55000, 95000, 30000, 110000, 40000],
        "Score_Credito": [650, 580, 710, 790, 500, 680, 740, 600, 810, 620],
        "Edad": [34, 28, 45, 50, 22, 39, 41, 26, 48, 31],
        "Pago_Prestamo": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    }
    datos = pd.DataFrame(datos_banco)
    modelo = XGBClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=2,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    modelo.fit(datos[COLUMNAS], datos["Pago_Prestamo"])
    return modelo


def solicitar_valor(mensaje):
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("Entrada inválida. Escribe un valor numérico, por ejemplo 45000 o 650.5.")


def desea_otra_prediccion():
    while True:
        respuesta = input("¿Quieres analizar otro cliente? [S/n]: ").strip().lower()
        if respuesta in {"s", "si", "sí"}:
            return True
        if respuesta in {"", "n", "no"}:
            return False
        print("Responde sí (S) o no (N).")


def main():
    print("Preparando el modelo...")
    modelo = entrenar_modelo()
    print("Modelo listo.\n")

    while True:
        print("=== Predicción de aprobación de préstamo ===")
        ingresos = solicitar_valor("Ingresos anuales en USD: ")
        score = solicitar_valor("Score de crédito: ")
        edad = solicitar_valor("Edad del cliente: ")

        cliente = pd.DataFrame([[ingresos, score, edad]], columns=COLUMNAS)
        probabilidades = modelo.predict_proba(cliente)[0]
        indice_pago = list(modelo.classes_).index(1)
        indice_no_pago = list(modelo.classes_).index(0)
        probabilidad_pago = probabilidades[indice_pago] * 100
        probabilidad_no_pago = probabilidades[indice_no_pago] * 100
        resultado = int(modelo.predict(cliente)[0])

        if resultado == 1:
            print("\n✅ Préstamo APROBADO según el modelo")
        else:
            print("\n❌ Préstamo RECHAZADO según el modelo")

        print(f"Probabilidad estimada de pago: {probabilidad_pago:.1f}%")
        print(f"Probabilidad estimada de no pago: {probabilidad_no_pago:.1f}%")
        print("Estimación educativa; no representa la decisión ni la confianza de un banco.\n")

        if not desea_otra_prediccion():
            print("Programa finalizado.")
            break
        print()


if __name__ == "__main__":
    main()
