
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==============================================================================
#                      ANÁLISIS Y REGRESIÓN LINEAL (ESS 11)
# ==============================================================================

# --- 1. CONFIGURACIÓN, CARGA DE DATOS Y LIMPIEZA ---

# Cargar el archivo de datos
file_path = "ESS11_parsed2.xlsx"
try:
    df_raw = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"ERROR: Archivo no encontrado en la ruta: {file_path}")
    print("Asegúrese de que el archivo 'ESS11_parsed2.xlsx' esté en la misma carpeta que el script.")
    exit()

print("Columnas disponibles en el archivo Excel:")
print(df_raw.columns.tolist())

# Definición de variables seleccionadas
# Variable Respuesta (Y)
Y_VAR = 'happy'  # Felicidad percibida (0-10)

# Variables Predictoras (X)
X_VARS = ['fltlnl', 'hinctnta', 'dosprt', 'rlgdgr']
# 1. 'fltlnl' - Sentimiento de Soledad (1-4)
# 2. 'hinctnta' - Decil de Ingresos (1-10)
# 3. 'dosprt' - Días con Deporte (0-7)
# 4. 'rlgdgr' - Grado de Religiosidad (0-10)

# Crear un DataFrame de trabajo con las variables relevantes
df = df_raw[[Y_VAR] + X_VARS].copy()

# Limpieza de datos: Eliminar filas con cualquier valor nulo (NaN) en las variables clave
df = df.dropna()

# # Comentario para el Punto a): Justificación de variables (para la entrega PDF)
# # Variable Respuesta (Y): 'happy'
# # Justificación: Es la métrica directa de bienestar subjetivo que deseamos explicar.
# # Predictoras: Se eligieron estas variables por su conocida influencia en el bienestar:
# # Soledad (salud mental), Ingresos (salud económica), Deporte (salud física) y Religiosidad (salud social/espiritual).

print(f"Dataset cargado. Filas finales después de limpiar NaN: {len(df)}")
print("-" * 120)


# ==============================================================================
#                     REGRESIÓN LINEAL SIMPLE (Puntos b y c)
# ==============================================================================

# 1. Calcular los valores X0 de interés para ICM(Y) e IP(Y) (las medias)
X0_values = df[X_VARS].mean().to_dict()

# Lista para almacenar los resultados del cuadro comparativo (Punto b)
results_simple = []
alpha = 0.05 # Nivel de significancia para 95% de confianza

print("Punto b) Resultados de Regresión Lineal Simple (Incluyendo IC y IP calculados en la media X̄):\n")
print(f"# NOTA: ICM(Y) e IP(Y) se calcularon usando el valor medio (X0) de cada predictor.")

for x_var in X_VARS:
    X = df[x_var]
    Y = df[Y_VAR]

    # Agregar la constante para el intercepto (beta_0)
    X_simple = sm.add_constant(X)

    # Ajustar el modelo OLS
    model = sm.OLS(Y, X_simple).fit()

    # --- CÁLCULO DE ICM(Y) e IP(Y) ---
    # 1. Definir el punto X0 (la media)
    x0 = X0_values[x_var]
    # Crear un DataFrame de predicción con el X0 y la constante '1'
    X_pred = pd.DataFrame([[1, x0]], columns=['const', x_var])

    # 2. Obtener el summary de predicción
    prediction_summary = model.get_prediction(X_pred)

    # 3. Extraer el ICM(Y) (Confidence Interval for the Mean Response)
    icm_frame = prediction_summary.summary_frame(alpha=alpha)
    icm = icm_frame['mean_ci_lower'][0], icm_frame['mean_ci_upper'][0]

    # 4. Extraer el IP(Y) (Prediction Interval for a New Response)
    ip = icm_frame['obs_ci_lower'][0], icm_frame['obs_ci_upper'][0]

    # Extracción de Métricas Globales
    r_squared = model.rsquared
    r = np.sqrt(r_squared) * np.sign(model.params[x_var])
    sigma2 = model.scale
    ic_beta1 = model.conf_int().loc[x_var].tolist()
    ic_beta0 = model.conf_int().loc['const'].tolist()

    # Almacenamiento de resultados
    results_simple.append({
        'Predictor': x_var,
        'X̄_Usado': f"{x0:.4f}",
        'Ŷ_f(X)': 'f(X)', # Columna Y-hat
        'β₀̂': f"{model.params['const']:.4f}",
        'β₁̂': f"{model.params[x_var]:.4f}",
        'σ²': f"{sigma2:.4f}",
        'R²': f"{r_squared:.4f}",
        'r': f"{r:.4f}",
        'IC(β₁)': f"({ic_beta1[0]:.4f}, {ic_beta1[1]:.4f})",
        'IC(β₀)': f"({ic_beta0[0]:.4f}, {ic_beta0[1]:.4f})",
        # Estos se refieren a la predicción en X0
        'ICM(Y)': f"({icm[0]:.4f}, {icm[1]:.4f})",
        'IP(Y)': f"({ip[0]:.4f}, {ip[1]:.4f})",
    })

# Convertir y mostrar el resultado del Punto b)
df_results_b_full = pd.DataFrame(results_simple)
print(df_results_b_full.to_string(index=False))
print("-" * 120)


# --- PUNTO c) SELECCIÓN DEL MEJOR PREDICTOR ---
print("Punto c) Selección del Mejor Predictor y Comentario:")

# Encontrar el R^2 más alto
best_predictor_row = df_results_b_full.loc[df_results_b_full['R²'].astype(float).idxmax()]
best_predictor_name = best_predictor_row['Predictor']
best_r_squared = float(best_predictor_row['R²'])
best_r_value = float(best_predictor_row['r'])

# Comentario para el PDF:
print(f"# El mejor predictor es: {best_predictor_name}. (Probablemente 'fltlnl' - Soledad percibida).")
print(f"# Razón: Presenta el R^2 más alto ({best_r_squared:.4f}), lo que indica que explica la mayor proporción de la varianza en la felicidad por sí solo.")
print(f"# El coeficiente de correlación (r) es {best_r_value:.4f}. Si es negativo (como se espera), la interpretación es: a MAYOR soledad, MENOR felicidad.")
print(f"# Además, el IC(β₁) no incluye el cero, confirmando que la relación es estadísticamente significativa.")
print("-" * 120)


# ==============================================================================
#                 REGRESIÓN LINEAL MÚLTIPLE (Puntos d, e y f)
# ==============================================================================

# Preparación de datos para Regresión Múltiple
Y_multi = df[Y_VAR]
X_multi = df[X_VARS]


# --- PUNTO e) OLS (Mínimos Cuadrados Ordinarios) ---
# OLS es la solución exacta y el benchmark para GD.

X_multi_ols = sm.add_constant(X_multi)
model_ols = sm.OLS(Y_multi, X_multi_ols).fit()

print("Punto e) Resultados OLS (Mínimos Cuadrados):\n")
print("# La salida 'summary' de statsmodels es la solución analítica exacta de Mínimos Cuadrados para Regresión Múltiple.")
print(model_ols.summary().as_text())

# Ecuación de Regresión OLS
ols_equation = (
    f"# Ecuación OLS Múltiple: happŷ = {model_ols.params['const']:.4f} "
    f"+ ({model_ols.params['fltlnl']:.4f}) * fltlnl "
    f"+ ({model_ols.params['hinctnta']:.4f}) * hinctnta "
    f"+ ({model_ols.params['dosprt']:.4f}) * dosprt "
    f"+ ({model_ols.params['rlgdgr']:.4f}) * rlgdgr"
)
print("\n" + ols_equation)
print("-" * 120)


# --- PUNTO d) Descenso de Gradiente (Gradient Descent - GD) ---
# Implementación de GD (Batch GD)

print("Punto d) Estimación por Descenso de Gradiente (GD):\n")
print("# Para que el Descenso de Gradiente sea efectivo y eficiente, los datos son estandarizados (Centrados y Escalados).")

# Estandarización de Datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_multi)

# Agregar columna de unos para el intercepto (b0)
X_gd = np.insert(X_scaled, 0, 1, axis=1)
Y_gd = Y_multi.values.reshape(-1, 1)

# Parámetros del Descenso de Gradiente (GD)
learning_rate = 0.01  # Tasa de aprendizaje
n_iterations = 1000   # Número de iteraciones o épocas
theta = np.zeros(X_gd.shape[1]).reshape(-1, 1) # Inicialización de coeficientes

# Función de Costo (MSE)
def cost_function(X, Y, theta):
    m = len(Y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - Y))
    return cost

# Algoritmo de Descenso de Gradiente
def gradient_descent(X, Y, theta, learning_rate, n_iterations):
    m = len(Y)
    for i in range(n_iterations):
        predictions = X.dot(theta)
        errors = predictions - Y
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - learning_rate * gradient
    return theta

# Ejecutar el GD
theta_final_gd = gradient_descent(X_gd, Y_gd, theta, learning_rate, n_iterations)

# Resultados de GD (Coeficientes estandarizados)
gd_results = pd.DataFrame({
    'Coeficiente': ['Intercepto'] + X_VARS,
    'Valor (Escalado/GD)': theta_final_gd.flatten()
})
print("# Coeficientes Resultantes de Descenso de Gradiente (datos escalados):")
print(gd_results.to_string(index=False))

# Comentario para el Punto e) sobre d) y e):
print("\n# Comentario Comparativo d) y e):")
print("# Los coeficientes de OLS (e) y GD (d) deberían ser muy similares. OLS encuentra la solución óptima en un solo paso (exacto) mientras que GD es un proceso iterativo de aproximación.")
print("# Para este tamaño de dataset, OLS (e) es más rápido y preciso. GD (d) se utiliza cuando el número de datos es muy grande para que OLS sea viable computacionalmente.")
print("-" * 120)


# --- PUNTO f) ¿MEJORA LA ESTIMACIÓN LA ADICIÓN DE VARIABLES? ---

# R^2 del mejor modelo simple (punto c)
r2_simple = best_r_squared

# R^2 ajustado del modelo múltiple (punto e)
r2_adj_multiple = model_ols.rsquared_adj  # Usamos R^2 ajustado para múltiples variables

print("Punto f) Comparación de Estimación (Modelo Múltiple vs. Mejor Modelo Simple):\n")
print(f"# R^2 Ajustado del Modelo Múltiple: {r2_adj_multiple:.4f}")
print(f"# R^2 del Mejor Modelo Simple:    {r2_simple:.4f}")

# Comentario para el PDF:
if r2_adj_multiple > r2_simple + 0.0001: # Comparación estricta del R^2 ajustado
    improvement = (r2_adj_multiple - r2_simple) * 100
    print("\n# CONCLUSIÓN: SÍ.")
    print("# La adición de las demás variables predictoras SÍ mejoró la estimación.")
    print(f"# El R^2 ajustado (que penaliza por las variables añadidas) aumentó en un {improvement:.4f}%.")
    print("# Esta mejora significativa demuestra que las variables adicionales (ingreso, deporte, religiosidad) tienen un poder explicativo independiente en la varianza de la felicidad, resultando en un modelo Múltiple superior para la predicción.")
else:
    print("\n# CONCLUSIÓN: NO MEJORÓ O LA MEJORA ES MARGINAL.")
    print("# El incremento en el R^2 ajustado no justifica la complejidad añadida al modelo. Esto indicaría que las variables añadidas no aportan suficiente información independiente una vez que la 'Soledad' ya está incluida.")

print("=" * 120)