# Ajuste por minimos cuadrados en Vercel (Python)

Aplicacion web en Python para ajustar modelos lineales en coeficientes:

\[
y \approx c_1 f_1(x_1,\dots,x_n) + c_2 f_2(x_1,\dots,x_n) + \cdots + c_m f_m(x_1,\dots,x_n)
\]

La app:

- Lee un archivo CSV o Excel numerico y detecta encabezados automaticamente:
  si el primer valor no es numerico, toma la primera fila como encabezados;
  en caso contrario, usa todas las filas como datos y nombra columnas `col1`, `col2`, etc.
- Toma como variables `x1..xn` todas las columnas excepto la ultima.
  Si solo existe una variable independiente, tambien puedes referirla como `x`.
- Toma como `y` la ultima columna.
- Ajusta coeficientes con `numpy.linalg.lstsq`.
- Muestra la tabla leida y la ecuacion ajustada en LaTeX.
- Grafica datos observados vs modelo ajustado bajo demanda cuando existe una sola variable independiente.
- Permite cargar datos desde archivo CSV/Excel y, opcionalmente, activar una caja para pegar CSV manualmente.
- Conserva el ultimo CSV cargado para recalcular sin tener que volver a seleccionarlo.
- Indica en pantalla cuando existe un CSV en memoria y la fuente usada en el ultimo calculo.
- Incluye modelos predefinidos (recta, polinomios, plano, curva logística y respuestas al escalón de 1er/2do orden) y permite escribir uno personalizado.

## Estructura

- `api/index.py`: backend Flask + logica del ajuste.
- `api/templates/index.html`: interfaz web.
- `requirements.txt`: dependencias Python.
- `vercel.json`: configuracion de despliegue.

## Uso local

1. Crear entorno e instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Ejecutar:

```bash
flask --app api/index.py run --debug
```

3. Abrir:

`http://127.0.0.1:5000`

## Despliegue en Vercel

1. Instala Vercel CLI y autentica:

```bash
npm i -g vercel
vercel login
```

2. Desde la raiz del proyecto:

```bash
vercel
```

3. Produccion:

```bash
vercel --prod
```

## Sintaxis del modelo

En la caja de texto puedes usar funciones base separadas por coma, salto de linea o `;`.

Ejemplos:

- `1, x1`
- `1, x`
- `1, x1, x1^2`
- `1, x1, x2, x1*x2`
- `1, sin(x1), exp(x2), x1^2`
- `1, e^(x1), x1^3`

Funciones disponibles: todas las trigonométricas e hiperbólicas (directas e inversas),
además de `exp`, `log`, `ln` y `sqrt`.
Tambien se acepta `^` para potencia y `e^(...)` como equivalente de `exp(...)`.
