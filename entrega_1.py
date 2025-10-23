# ==============================================================================
# PROYECTO: ANALISIS Y REGRESION LINEAL - ENTREGA ESS11
# Código para obtener resultados de Regresión (Puntos b, c, d, e, f)
# ==============================================================================

# -- Importamos las bibliotecas necesarias --
import pandas as pd           # Para manejar los datos en tablas
import numpy as np            # Para operaciones matemáticas avanzadas
import statsmodels.api as sm  # Herramienta para OLS y reportes estadísticos
from sklearn.preprocessing import StandardScaler # Necesario para el Descenso de Gradiente


# --- 1. CARGA_DE_DATOS, DEFINICION DE VARIABLES Y LIMPIEZA ---

PATH_ARCHIVO = "ESS11_parsed2.xlsx" # Ruta del archivo
df_DATOS = pd.read_excel(PATH_ARCHIVO)

# Nombres de variables según el proyecto
VAR_RESPUESTA_Y = 'happy'  # Felicidad (la variable a predecir!)
VARS_PREDICTORAS_X = ['fltlnl', 'hinctnta', 'dosprt', 'rlgdgr']  # Las X

# Definimos el dataframe de USO
df_USO = df_DATOS[[VAR_RESPUESTA_Y] + VARS_PREDICTORAS_X].copy()

# Eliminar nulos.
df_USO = df_USO.dropna()

NUM_MUESTRA = len(df_USO) # Tamaño final de la muestra


# ==============================================================================
#                     REGRESION_SIMPLE (Puntos b y c)
# ==============================================================================

# Calculamos las medias para usarlas como los X0 para IC e IP
medias_X0 = df_USO[VARS_PREDICTORAS_X].mean().to_dict()
resultados_tabla_b = []
NIVEL_ALPHA = 0.05 # Para 95% de confianza

print("--- PUNTO B) RESULTADOS DE REGRESION LINEAL SIMPLE EN TABLA ---")
# Usamos 'statsmodels' para el ajuste, es lo más robusto
for var_pred in VARS_PREDICTORAS_X:
    X_var = df_USO[var_pred]
    Y_var = df_USO[VAR_RESPUESTA_Y]

    # 1. Ajuste OLS
    X_simple_const = sm.add_constant(X_var)
    modelo_simple = sm.OLS(Y_var, X_simple_const).fit()

    # -- Obtención de métricas --

    # Coeficiente R
    R_2 = modelo_simple.rsquared
    r_COEF = np.sqrt(R_2) * np.sign(modelo_simple.params[var_pred])

    # Varianza Estimada (MSE)
    sigma2_EST = modelo_simple.scale

    # IC para Beta0 y Beta1 (95%)
    IC_b1 = modelo_simple.conf_int().loc[var_pred].tolist()
    IC_b0 = modelo_simple.conf_int().loc['const'].tolist()

    # -- Cálculo de ICM(Y) e IP(Y) en X0 --
    x0_USADO = medias_X0[var_pred]

    # Preparamos el punto X0
    X_predic = pd.DataFrame([[1, x0_USADO]], columns=['const', var_pred])
    resumen_pred = modelo_simple.get_prediction(X_predic).summary_frame(alpha=NIVEL_ALPHA)

    # Extraemos rangos
    ICM_RANGO = resumen_pred['mean_ci_lower'][0], resumen_pred['mean_ci_upper'][0]
    IP_RANGO = resumen_pred['obs_ci_lower'][0], resumen_pred['obs_ci_upper'][0]

    # Guardamos los resultados en el formato de la tabla
    resultados_tabla_b.append({
        'Predictor': var_pred,
        'Media(X0)': f"{x0_USADO:.4f}",
        'Beta0_hat': f"{modelo_simple.params['const']:.4f}",
        'Beta1_hat': f"{modelo_simple.params[var_pred]:.4f}",
        'Sigma2_hat': f"{sigma2_EST:.4f}",
        'R_CUADRADO': f"{R_2:.4f}",
        'r': f"{r_COEF:.4f}",
        'IC_B1': f"({IC_b1[0]:.4f}, {IC_b1[1]:.4f})",
        'IC_B0': f"({IC_b0[0]:.4f}, {IC_b0[1]:.4f})",
        'ICM(Y)': f"({ICM_RANGO[0]:.4f}, {ICM_RANGO[1]:.4f})",
        'IP(Y)': f"({IP_RANGO[0]:.4f}, {IP_RANGO[1]:.4f})",
    })

# Imprimir el Cuadro Final (Punto b)
df_tabla_final = pd.DataFrame(resultados_tabla_b)
print(df_tabla_final.to_string(index=False))

# Variables para el Punto f
MEJOR_R2_SIMPLE = float(df_tabla_final.loc[df_tabla_final['R_CUADRADO'].astype(float).idxmax()]['R_CUADRADO'])


# ==============================================================================
#                 REGRESION_MULTIPLE (Puntos d, e y f)
# ==============================================================================

# --- PUNTO E) OLS (Mínimos Cuadrados) ---
print("\n\n--- PUNTO E) REGRESION MULTIPLE POR OLS (SOLUCION EXACTA) ---\n")
print("Reporte estadístico completo de la ecuación por Mínimos Cuadrados:")

X_MULT_CONST = sm.add_constant(df_USO[VARS_PREDICTORAS_X])
modelo_OLS_MULTIPLE = sm.OLS(df_USO[VAR_RESPUESTA_Y], X_MULT_CONST).fit()
print(modelo_OLS_MULTIPLE.summary().as_text()) # Imprimimos el summary, es lo que pide el punto e


# --- PUNTO D) DESCENSO DEL GRADIENTE (SOLUCION ITERATIVA) ---
print("\n\n--- PUNTO D) REGRESIÓN MÚLTIPLE POR DESCENSO DE GRADIENTE ---\n")
print("Coeficientes finales después de la iteración (Datos Escalados):")

# 1. ESCALADO DE DATOS: Necesario para que GD evite oscilaciones y converja rápido
scaler_X = StandardScaler()
X_escalado = scaler_X.fit_transform(df_USO[VARS_PREDICTORAS_X])

# Añadir la columna de la intersección (el '1')
X_gradiente = np.insert(X_escalado, 0, 1, axis=1)
Y_gradiente = df_USO[VAR_RESPUESTA_Y].values.reshape(-1, 1)

# Parámetros y arranque
TASA_APRENDIZAJE = 0.01  # ALFA
NUM_EPOCAS = 1000
coeficientes_theta = np.zeros(X_gradiente.shape[1]).reshape(-1, 1) # Inicializamos en cero

# Función de Costo (J(theta) - MSE)
def funcion_costo(X, Y, theta):
    m = len(Y)
    predicciones = X.dot(theta)
    costo = (1/(2*m)) * np.sum(np.square(predicciones - Y))
    return costo

# Algoritmo de Descenso
def calcular_gradiente(X, Y, theta, tasa, epocas):
    m = len(Y)
    for i in range(epocas):
        # 1. Calculamos error
        errores = (X.dot(theta)) - Y
        # 2. Calculamos gradiente
        gradiente = (1/m) * X.T.dot(errores)
        # 3. Actualizamos coeificientes (el paso)
        theta = theta - tasa * gradiente
    return theta

# Ejecutamos
COEFICIENTES_FINALES_GD = calcular_gradiente(X_gradiente, Y_gradiente, coeficientes_theta, TASA_APRENDIZAJE, NUM_EPOCAS)

# Imprimir Resultados de GD
tabla_GD = pd.DataFrame({
    'Coeficiente': ['Interseccion'] + VARS_PREDICTORAS_X,
    'Valor_FINAL_GD': COEFICIENTES_FINALES_GD.flatten()
})
print(tabla_GD.to_string(index=False))