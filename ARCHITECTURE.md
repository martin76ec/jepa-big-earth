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

## 2. Arquitectura y proceso

Pipeline de un solo comando, con caché por etapa:

```
BigEarthNet-S2 (streaming) ─▶ EDA + subconjunto balanceado (caché)
        │
        ▼
  I-JEPA ViT-H/14 (CONGELADO) ─▶ embeddings 1280-D (caché)
        │
        ▼
  Pipeline sklearn:  StandardScaler ─▶ [SelectKBest opc.] ─▶ clasificador
        │
        ├─▶ Optimización de hiperparámetros (GridSearchCV)
        └─▶ Evaluación: CV repetida + Wilcoxon (LogReg vs SVM) + t-SNE
```

### 2.1 Representación

- **Datos** (`scan_labels`): pasada en *streaming* leyendo solo `labels`
  para el EDA y elegir el **top-5** de clases (dataset multi-etiqueta,
  43 clases CLC). El escaneo se cachea.
- **Subconjunto balanceado** (`build_subset`): BigEarthNet es
  multi-etiqueta (~2.7 etiquetas/parche); exigir parches con una *única*
  etiqueta del top-5 degeneraba el subconjunto. En su lugar se toma
  cualquier parche con ≥1 etiqueta del top-5 y se le asigna la clase
  **menos representada** hasta el momento (balanceo *greedy*,
  determinista). Tope `per_class = 2000` → ≈10 000 imágenes, 5 clases
  balanceadas; caché invalidada por `per_class` y versión de
  construcción.
- **Extracción de características (JEPA)**: I-JEPA ViT-H/14 **congelado**;
  *embedding* por imagen = *mean pooling* de los tokens de parche
  (vector **1280-D**). Es *transfer learning*, por lo que **Feature
  Selection** puede omitirse (`SelectKBest` queda opcional). El encoder
  se aplica **una sola vez**; *embeddings* cacheados (fp16 en CUDA,
  *checkpoint* reanudable).
- **Pipeline** (`sklearn.Pipeline`): `StandardScaler` → [`SelectKBest`
  opc.] → clasificador. JEPA queda **fuera del bucle de CV** (congelado,
  sin fuga de información).

### 2.2 Optimización

- **Hiperparámetro**: `C` (= 1/λ) de la Regresión Logística **L2**, con
  `GridSearchCV` + `RepeatedStratifiedKFold` **10×3** y métrica
  **balanced accuracy** (clases balanceadas; evita el sesgo de accuracy).
- **Malla**: `C ∈ logspace(-3, 1, 9)`, acotada a la región que converge
  (`lbfgs`, `tol=1e-3`); los `C` grandes (poca regularización) no
  convergían y daban peor resultado.
- **Diagnóstico**: la **curva de validación** se deriva de
  `grid.cv_results_` (sin reejecutar *fits*) y se genera la **curva de
  aprendizaje** (sesgo/varianza).

### 2.3 Evaluación

- **Dos técnicas**: Regresión Logística **L2** vs **SVM RBF**, con
  *repeated k-fold CV* **10×3** sobre **los mismos *splits***.
- **Test estadístico**: **Wilcoxon** pareado (*signed-rank*) sobre las
  *balanced accuracy* por *fold*, α = 0.05 (H₀: sin diferencia).
- **Visualización**: proyección **t-SNE** 2D de los *embeddings*
  coloreada por clase (separabilidad cualitativa).

Resultados analizados en [`README.md`](README.md).

## 3. Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Servidor con GPU CUDA:** instalar primero el wheel CUDA de PyTorch
para no quedarse con el build `+cpu` (silenciosamente lento y agota la
RAM del host). El driver es retrocompatible: la versión que reporta
`nvidia-smi` (p. ej. CUDA 12.8) es el máximo soportado, no un requisito
exacto; `cu124` funciona en drivers ≥ 12.4.

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt   # después del torch CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# OK = versión SIN sufijo '+cpu' y True. En tiempo de encode, la línea
# '[jepa] ... -> device=cuda' lo confirma.
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

- `outputs/metrics/tuning.json` — mejor λ, balanced accuracy de CV.
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
