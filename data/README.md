# Dataset — Catastro Bogotá (TPREDIO)

## Descripción

Conjunto de datos **abiertos del catastro multipropósito de Bogotá D.C.**, con
información tabular y espacial de los predios urbanos y rurales de la ciudad.
Incluye identificadores catastrales, áreas de terreno y construcción, destinos
económicos, avalúos y puntajes de calificación constructiva.

| Atributo | Valor |
|----------|-------|
| **Nombre** | TPREDIO (Predios de Bogotá) |
| **Fuente** | IDECA / Datos Abiertos Bogotá — Catastro Multipropósito |
| **Filas** | ~3.8 millones (versión completa) |
| **Columnas** | 51 |
| **Tamaño** | ~1.4 GB (CSV) |
| **Licencia** | Datos abiertos (libre para uso y reutilización) |
| **Formato** | CSV separado por coma (`,`) |

## Por qué el archivo completo no está en el repositorio

El archivo `TPREDIO.csv` pesa ~1.4 GB y supera el límite de 100 MB por archivo
de GitHub. Por ello:

- El **dataset completo** se excluye mediante `.gitignore`.
- Se incluye una **muestra** de 6.000 filas en [`sample/TPREDIO_sample.csv`](sample/TPREDIO_sample.csv)
  para poder ejecutar y probar el código sin descargar el archivo completo.

## Cómo obtener el dataset completo

1. Ingresa al portal de **Datos Abiertos de Bogotá** (IDECA) y busca el
   conjunto de datos **"Predios / TPREDIO — Catastro"**.
2. Descarga el archivo `TPREDIO.csv`.
3. Colócalo en esta carpeta con el nombre exacto `data/TPREDIO.csv`.

La ruta esperada es:

```
data/TPREDIO.csv
```

## Uso en Google Colab

El notebook `notebooks/proyecto_catastro_bogota.ipynb` detecta automáticamente
tres orígenes, en este orden:

1. Google Drive montado (`/content/drive/MyDrive/.../TPREDIO.csv`).
2. Archivo local del repositorio (`data/TPREDIO.csv`).
3. La muestra incluida (`data/sample/TPREDIO_sample.csv`).

## Diccionario de datos (variables principales)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `PRECHIP` | texto | Código Homologado de Identificación Predial (llave primaria) |
| `PRECBARRIO` / `PRECMANZ` / `PRECPREDIO` | texto | Códigos jerárquicos de barrio, manzana y predio |
| `PREDIRECC` | texto | Dirección alfanumérica estandarizada del lote |
| `PREATERRE` | numérica | Área total del terreno (m²) — predictor |
| `PREACONST` | numérica | Área construida (m²) — **objetivo de regresión** |
| `PRECDESTIN` | categórica | Destino económico del suelo — **objetivo de clasificación** |
| `PREVACTUAL` | numérica | Avalúo catastral actual |
| `PREPUNTAJE` | numérica | Calificación global de calidad estructural |
| `PREEARMAZ` / `PREEMUROS` / `PREECUBIER` | numérica | Puntajes de armazón, muros y cubierta |
| `PRECINDUS` / `PREACERCHA` | numérica | Calificaciones de uso industrial (nulos = no aplica) |
| `PREUCALIF` | categórica | Calificación constructiva (A, B, C, …) |
| `PRECLASE` | categórica | Clase del predio (N = normal, P = propiedad horizontal) |

> Para el diccionario exhaustivo de las 51 columnas, consulta el reporte de
> calidad generado por el notebook (`notebooks/proyecto_catastro_bogota.ipynb`).
