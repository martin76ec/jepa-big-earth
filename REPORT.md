# Reporte de resultados — BigEarthNet-S2 + JEPA + clasificador clásico

Clasificación de cobertura terrestre sobre las **5 clases más frecuentes**
de BigEarthNet-S2 (config `s2-rgb`), usando **I-JEPA ViT-H/14 congelado**
como extractor de características y clasificadores clásicos por encima.

## 1. Tabla de cumplimiento (score table)

| Ítem del rubric | Pts | Cómo se cumple | Evidencia |
|---|:--:|---|---|
| EDA | — | `scan_labels` + `run_eda`: frecuencias, cardinalidad, balance, muestras | `outputs/eda/*.png`, `eda_summary.json` |
| Feature Extraction | — | I-JEPA ViT-H/14 congelado, *mean pooling* de tokens (transfer learning) | `features.py`, `embeddings.npz` |
| Feature Selection | — | Omitida con justificación (transfer learning sobre imágenes); `SelectKBest` queda opcional | `models.py`, `config.select_k_best` |
| Pipelines | — | `sklearn.Pipeline`: `StandardScaler → [filtro opc.] → clasificador` | `models.py` |
| Técnica 1 (obligatoria) | — | Regresión Logística Regularizada **L2** (`l1_ratio=0`) | `models.make_logreg` |
| Optimización de hiperparámetros | **3.5** | `GridSearchCV` sobre `C`(=1/λ), `RepeatedStratifiedKFold`; curvas de validación y aprendizaje | `tuning.py`, `figures/validation_curve.png`, `learning_curve.png` |
| Comparación estadística (2 técnicas) | **3.5** | LogReg L2 vs **SVM RBF**, *repeated k-fold* (10×3) + **Wilcoxon** (α=0.05) | `compare.py`, `figures/model_comparison.png` |
| Visualización | **1** | t-SNE 2D de los *embeddings* coloreado por clase | `figures/tsne.png` |
| Referencias (IEEE) | — | Sección 4 + nota de transparencia (asistencia LLM) | este documento |

## 2. Datos y EDA

- Parches escaneados: **20 000**; etiquetas CLC únicas observadas: **35**;
  promedio **2.69** etiquetas/parche (dataset multi-etiqueta).
- Top-5 (orden de frecuencia): **Pastures** (14 236), **Non-irrigated
  arable land** (8 737), **Coniferous forest** (4 635), **Sea and ocean**
  (3 302), **Mixed forest** (3 276).
- Subconjunto de modelado: **10 000** imágenes, **balanceado a 2 000/clase**
  (asignación *greedy* sobre parches con ≥1 etiqueta del top-5).
- Métrica: **balanced accuracy** (las clases del problema se equilibran;
  evita inflar el resultado por la clase mayoritaria).

| Frecuencia de etiquetas (top-5 resaltado) | Etiquetas por parche |
|---|---|
| ![Frecuencia de etiquetas](report_assets/label_frequency.png) | ![Cardinalidad](report_assets/label_cardinality.png) |

| Balance del subconjunto (2 000/clase) | Ejemplos RGB por clase |
|---|---|
| ![Balance del subconjunto](report_assets/subset_balance.png) | ![Muestras](report_assets/sample_grid.png) |

## 3. Optimización de hiperparámetros (Regresión Logística L2)

- Mejor `C = 3.16e-3` → **λ ≈ 316.2** (regularización fuerte).
- **Balanced accuracy CV (mejor): 0.791**.
- Curva de validación: el rendimiento es estable (~0.78–0.79) para
  `C` pequeño/medio y **cae a ~0.76 con `C` grande** (poca
  regularización) → el óptimo en λ alto indica que los *embeddings*
  JEPA necesitan *shrinkage* (mitiga varianza, sin sesgo apreciable).

| Curva de validación (λ vs balanced acc.) | Curva de aprendizaje |
|---|---|
| ![Curva de validación](report_assets/validation_curve.png) | ![Curva de aprendizaje](report_assets/learning_curve.png) |

## 4. Comparación estadística (LogReg L2 vs SVM RBF)

| Modelo | Balanced accuracy (media ± std, 10×3 CV) |
|---|:--:|
| **Regresión Logística L2** | **0.791 ± 0.091** |
| SVM RBF | 0.746 ± 0.098 |

- Wilcoxon pareado (mismos *splits*): estadístico = 12.5,
  **p = 0.0024 < 0.05**.
- **Veredicto: se rechaza H₀** — diferencia estadísticamente
  significativa; **la Regresión Logística L2 es mejor** que la SVM RBF
  sobre estos *embeddings*.

![Comparación de modelos (boxplot por fold)](report_assets/model_comparison.png)

## 5. Conclusiones

- Los *embeddings* I-JEPA congelados son **linealmente separables** de
  forma razonable: un clasificador lineal regularizado alcanza
  **0.79 balanced accuracy** en 5 clases.
- La **regularización fuerte (λ≈316)** es clave; sin ella el modelo
  pierde ~3–4 puntos → varianza alta en el espacio de 1280 dimensiones.
- El modelo lineal **supera de forma significativa** a la SVM RBF
  (Wilcoxon p≈0.002), sugiriendo que el *kernel* no aporta sobre
  representaciones ya ricas; coherente con el t-SNE (abajo).

![t-SNE de los embeddings JEPA coloreado por clase](report_assets/tsne.png)

## 6. Referencias (IEEE)

[1] M. Assran *et al.*, "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture (I-JEPA)," *Proc. IEEE/CVF CVPR*,
2023.

[2] G. Sumbul, M. Charfuelan, B. Demir, and V. Markl, "BigEarthNet: A
Large-Scale Benchmark Archive for Remote Sensing Image Understanding,"
*Proc. IEEE IGARSS*, 2019.

[3] F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python,"
*J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[4] L. van der Maaten and G. Hinton, "Visualizing Data using t-SNE,"
*J. Mach. Learn. Res.*, vol. 9, pp. 2579–2605, 2008.

[5] F. Wilcoxon, "Individual Comparisons by Ranking Methods,"
*Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945.

[6] Anthropic, "Claude Code," herramienta de asistencia usada para la
generación del código base (ver nota de transparencia en `main.py`).

---
*Pts*: pesos del rubric documentados en el código (`tuning.py`,
`compare.py`, `viz.py`); los ítems con "—" forman parte del rubric sin
peso numérico explícito en el código. Resultados generados por
`python main.py all` (`outputs/metrics/{tuning,comparison}.json`,
`outputs/eda/eda_summary.json`).
