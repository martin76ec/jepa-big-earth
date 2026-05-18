# Clasificación de cobertura terrestre en BigEarthNet-S2 con JEPA + clasificador clásico

## 1. Introducción

La clasificación automática de cobertura terrestre a partir de imágenes
satelitales Sentinel-2 es clave para el monitoreo ambiental, la
planificación territorial y la detección de cambios de uso de suelo.
Etiquetar manualmente parches satelitales es costoso, por lo que interesa
aprovechar representaciones aprendidas de forma auto-supervisada.

Este proyecto usa **JEPA** (*Joint-Embedding Predictive Architecture*,
variante I-JEPA preentrenada por Meta AI) como extractor de
características congelado sobre el dataset
[BigEarthNet-S2-v1.0](https://huggingface.co/datasets/danielz01/BigEarthNet-S2-v1.0)
(configuración `s2-rgb`), y sobre esos *embeddings* monta clasificadores
clásicos para distinguir las **5 clases más frecuentes**. Se compara
estadísticamente una **Regresión Logística Regularizada** contra una
**SVM con kernel RBF**.

## 2. Metodología

1. **Escaneo de etiquetas** (`scan_labels`): pasada en *streaming* leyendo
   solo la columna `labels` para el EDA y para elegir las 5 clases más
   frecuentes (dataset multi-etiqueta de 43 clases CLC).
2. **Subconjunto multiclase**: se conservan parches con **etiqueta única**
   dentro del top-5, formando un problema de clasificación de 5 clases.
3. **Feature Extraction (JEPA)**: I-JEPA ViT-H/14 congelado; *embedding*
   por imagen = *mean pooling* de los tokens de parche. Es *transfer
   learning* sobre imágenes, por lo que —según el rubric— el paso de
   **Feature Selection** puede omitirse; queda disponible como filtro
   opcional (`SelectKBest`, `config.select_k_best`).
4. **Pipelines** (`sklearn.Pipeline`): `StandardScaler` → [filtro opc.] →
   clasificador.
5. **Optimización de hiperparámetros**: `GridSearchCV` sobre `C` (=1/λ)
   con `RepeatedStratifiedKFold`; se generan curva de validación y curva
   de aprendizaje.
6. **Comparación estadística**: *repeated k-fold CV* (10×10 por defecto,
   mismos *splits* para ambos modelos) + test pareado de **Wilcoxon**
   (α = 0.05).
7. **t-SNE**: proyección 2D de los *embeddings* coloreada por clase.

## 3. Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

El dataset es `gated: auto`. Crear un token en
<https://huggingface.co/settings/tokens>, visitar una vez la página del
dataset para aceptar las condiciones, y exportar el token:

```bash
cp .env.example .env        # editar HF_TOKEN
export HF_TOKEN=hf_xxx       # o: huggingface-cli login
```

## 4. Uso

```bash
python main.py all        # pipeline completo (EDA → features → tune → compare → tsne)
python main.py eda        # solo EDA + construcción del subconjunto
python main.py features   # extrae y cachea embeddings JEPA
python main.py tune       # optimización de hiperparámetros
python main.py compare    # comparación estadística de 2 técnicas
python main.py tsne       # visualización t-SNE
```

Resultados en `outputs/`: `eda/`, `figures/`, `metrics/` y la caché en
`cache/` (subconjunto y *embeddings*, para no re-descargar).

Parámetros (tamaño de muestra, modelo JEPA, *folds*, etc.) en
`src/ben_jepa/config.py`. En Mac sin GPU CUDA se usa Apple MPS si está
disponible; el costo de JEPA se paga una sola vez gracias a la caché.

## 5. EDA

Generado por `python main.py eda` en `outputs/eda/`:

- `label_frequency.png` — frecuencia de etiquetas (top-5 resaltado).
- `label_cardinality.png` — etiquetas por parche (naturaleza multi-etiqueta).
- `subset_balance.png` — tamaño del subconjunto por clase.
- `sample_grid.png` — ejemplos RGB por clase.
- `eda_summary.json` — estadísticas numéricas.

## 6. Resultados

Tras ejecutar el pipeline, los valores quedan en:

- `outputs/metrics/tuning.json` — mejor λ, accuracy de CV.
- `outputs/metrics/comparison.json` — medias, p-valor de Wilcoxon, veredicto.
- `outputs/figures/` — `validation_curve.png`, `learning_curve.png`,
  `model_comparison.png`, `tsne.png`.

## 7. Conclusiones

Se completa tras ejecutar el pipeline, discutiendo: (a) separabilidad de
las clases en el espacio JEPA según t-SNE, (b) λ óptimo y diagnóstico
sesgo/varianza de las curvas, y (c) si la diferencia LogReg vs SVM es
estadísticamente significativa según Wilcoxon.

## 8. Referencias (IEEE)

[1] M. Assran *et al.*, "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture (I-JEPA)," *CVPR*, 2023.

[2] G. Sumbul, M. Charfuelan, B. Demir, and V. Markl, "BigEarthNet: A
Large-Scale Benchmark Archive for Remote Sensing Image Understanding,"
*IEEE IGARSS*, 2019.

[3] F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python,"
*JMLR*, vol. 12, pp. 2825–2830, 2011.

[4] L. van der Maaten and G. Hinton, "Visualizing Data using t-SNE,"
*JMLR*, vol. 9, pp. 2579–2605, 2008.

[5] F. Wilcoxon, "Individual Comparisons by Ranking Methods,"
*Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945.

[6] Anthropic, "Claude Code," herramienta de asistencia para la
generación del código base (ver nota de transparencia en `main.py`).
