import pandas as pd
import matplotlib.pyplot as plt
# Lectura de archivo
df = pd.read_excel("ESS11_parsed.xlsx")
print(df.columns.tolist())
x_pts = df["pplfair"]
y_pts = df["ppltrst"] #VAMOS CAMBIANDO VARIABLE INDEPENDIENTE
# Resumen estadístico
y_mean = y_pts.mean()
x_mean = x_pts.mean()
x_xm_diff = x_pts - x_mean
y_ym_diff = y_pts-y_mean
sxy = (x_xm_diff * y_ym_diff).sum()
sxx = ((x_xm_diff)**2).sum()
syy = (y_ym_diff**2).sum()
b1 = sxy/sxx
b0 = y_mean - b1*x_mean
sce = syy - ((sxy)**2)/sxx
varianza = sce / (x_pts.count() - 2)
coef_determinacion = 1 - (sce/syy)
#Construcción de puntos para recta
x_min = x_pts.min()
x_max = x_pts.max()
y1 = b0 + b1*x_min
y2 = b0 + b1*x_max
# Grafico
fig, ax = plt.subplots()
plt.xlabel("La mayoría de la gente intenta sacarte ventaja")
plt.ylabel("No se puede confiar en la mayoría de la gente")
ax.scatter(x_pts, y_pts)
ax.plot([x_min,x_max],[y1,y2],"-r")# -- coding: utf-8 --