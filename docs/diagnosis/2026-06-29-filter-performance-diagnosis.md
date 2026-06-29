# Diagnóstico: por qué los filtros se estancan en ~33 %

**Fecha:** 2026-06-29
**Pregunta:** los resultados son flojos (máximo `test_mean ≈ 0.33`). ¿El cuello de botella es
falta de imágenes, falta de capacidad de los filtros, o una mala estrategia de
caracterización de activaciones? ¿Conviene centrarse en ciertas capas? ¿Es RF-DETR el
modelo adecuado? El filtro de red neuronal aún está sin probar.

Este documento es un diagnóstico basado en el código y en los 283 runs válidos de
`results/experiments/experiment_results.csv`. Las pruebas decisivas que aún faltan se
implementan en `scripts/diagnose_ceiling.py` (este entorno no tiene torch/datos, así que
hay que ejecutarlas en una máquina con GPU y dataset).

---

## TL;DR — dónde tirar

1. **NO es falta de imágenes.** En las 283 filas `test_mean ≥ train_reduction` (el mejor:
   test 0.334 vs train 0.314). Si los datos fueran el límite, los filtros de alta
   capacidad (LUT size 17, STC P16) sobreajustarían y test quedaría por debajo de train.
   Ocurre lo contrario: **cero sobreajuste**. Más datos no es la palanca principal.

2. **El fallo nº1 es de estrategia/medición: nunca habéis medido lo que de verdad
   importa.** Todo el proyecto optimiza y reporta *distancia de activaciones del
   projector*, pero `src/benchmark.py` es un stub con TODOs. **No existe una sola medida de
   recuperación de detección.** Un 33 % de reducción de activaciones podría equivaler a
   recuperar el 90 % de la detección… o el 5 %. No lo sabéis, y toda decisión ("filtros
   débiles", "modelo equivocado") es prematura hasta cerrar ese bucle.

3. **La capacidad ayuda pero satura.** Escalera de capacidad (mejor test por familia):
   brightness 0.12 → affine 0.20 → matrix 0.21 → ccm 0.24 → tone_curve 0.24 → lut_3d 0.30 →
   spatial_tone_curve 0.33. Rendimientos decrecientes claros. Si existe un *techo* en ~0.33
   que ni un optimizador de píxeles libres supera, un filtro neuronal tampoco lo hará: el
   límite sería el modelo/representación, no el filtro.

4. **RF-DETR/DINOv2 es un sospechoso fuerte.** El backbone es DINOv2 ViT-S/14,
   auto-supervisado con augmentación fotométrica agresiva → sus features son invariantes a
   iluminación *por diseño*. Dos consecuencias: (a) puede que la detección apenas caiga bajo
   el cambio de luz (poco que recuperar), y (b) el gap que queda es justo la parte **no
   invertible con píxeles**. Un detector CNN clásico mostraría un gap mayor y más
   corregible.

**Orden de ataque:** Prioridad 0 = medir recuperación de detección (cerrar el bucle).
Prioridad 1 = medir el techo "oracle" (píxeles libres) para separar "filtro débil" de
"gap irreducible". Solo entonces decidir sobre filtro neuronal / cambio de modelo.

---

## Evidencia

### A. No hay sobreajuste → no es un problema de datos

Mejor configuración por familia de filtro y su `train_reduction` correspondiente:

| filtro | test_mean | train_reduction | grupo |
|---|---|---|---|
| chromatic_adaptation | 0.106 | 0.047 | projector |
| brightness_2param | 0.123 | 0.097 | projector |
| gamma_3param | 0.136 | 0.074 | projector |
| local_tonemap | 0.148 | 0.141 | projector |
| affine_6param | 0.204 | 0.158 | projector |
| matrix_12param | 0.206 | 0.143 | projector |
| ccm_high_order | 0.237 | 0.144 | projector |
| tone_curve | 0.241 | 0.187 | projector |
| lut_3d | 0.301 | 0.253 | phase2_group |
| spatial_tone_curve | 0.334 | 0.314 | phase2_group |

`test ≥ train` en todos los casos. El split es **por escena** (escenas enteras fuera del
train), así que esto es generalización real a escenas no vistas. La conclusión es robusta:
el filtro aprende el *desplazamiento de iluminación*, no memoriza escenas. **Añadir
imágenes no va a mover la aguja** con los filtros actuales (sí podría importar para un
filtro neuronal de mucha capacidad — ver Prioridad 4).

### B. La capacidad ayuda con rendimientos decrecientes

0.12 (brightness) → 0.33 (spatial_tone_curve) al subir capacidad, pero la curva se aplana
entre LUT3D (0.30) y STC (0.33). Esto es compatible con **dos** hipótesis que hay que
desempatar con el probe oracle:

- *Filtros aún cortos*: un mapeo de píxeles más expresivo (red neuronal) seguiría subiendo.
- *Techo de representación*: la diferencia A↔B en el projector tiene una componente que
  **ningún** filtro de píxeles puede cerrar (contenido + no-linealidad del bottleneck +
  ruido de la métrica). En ese caso 0.33 ya está cerca del máximo recuperable.

### C. La elección de capa ya está bien orientada (pero el target no está validado)

Mejor `test_mean` por grupo de capas:

| grupo | best | media |
|---|---|---|
| projector | 0.31–0.33 | 0.21 |
| backbone.late (+proj) | 0.20–0.21 | 0.12 |
| backbone.early | 0.14 | 0.06 |
| backbone.mid | 0.08 | 0.03 |
| decoder / proj+decoder | 0.06 | 0.03 |

El projector domina y el decoder es inútil como target — coherente con la intuición del
proyecto. **Pero**: que el projector sea donde más se reduce la distancia **no implica** que
reducir esa distancia recupere detección. El projector es un bottleneck con cabezas de
clasificación/caja aguas abajo; `l2_rel` sobre el tensor aplanado pondera por igual canales
y posiciones que la detección no usa por igual. Esto solo se valida midiendo detección
(Prioridad 0).

### D. Magnitud del shift

`level_1_vs_level_2` (shift menor) llega a 0.33; `level_1_vs_level_3` (shift mayor) se queda
en 0.25. Cuanto mayor el cambio de luz, más difícil — esperado, y otra señal de que parte
del gap es estructural.

### E. El bucle nunca se cerró

- `src/benchmark.py`: 44 líneas, todo TODOs. **Fase 3 jamás se ejecutó.**
- Los 4 runs en modo `detection`/`combined` (`det_stc_*`) solo se compararon en la métrica
  de *activaciones* (0.17–0.19), que penaliza al objetivo de detección por construcción.
  No se comparó nada en la métrica que importa.

---

## Plan de diagnóstico (priorizado)

### Prioridad 0 — Cerrar el bucle: medir recuperación de detección  *(decisivo, barato)*

Implementado en `scripts/diagnose_ceiling.py --mode detection`. Para cada par de test:
mide el "desacuerdo" de detección A↔B (KL de logits + L1 de cajas, vía
`detection_output_loss`, ya existente) **sin filtro** y **con el filtro entrenado**, y reporta
`detection_recovery = (gap_sin_filtro − gap_con_filtro) / gap_sin_filtro`.

Qué decide:
- Reporta primero el **gap base A↔B sin filtro**. Si RF-DETR apenas se degrada con el
  cambio de luz, la premisa del proyecto es débil *para este modelo* → ir a Prioridad 3.
- Si el gap es grande pero `detection_recovery ≫ 0.33`, entonces **33 % de activaciones ya
  recupera buena detección** y el "problema" es en parte cosmético (la métrica proxy
  subestima). Dejar de optimizar el número de activaciones y optimizar/medir detección.
- Si `detection_recovery ≈ 0` pese a reducir activaciones, el target (projector L2-rel) está
  desalineado con la detección → cambiar de estrategia de caracterización (ver Prioridad 2).

### Prioridad 1 — Techo "oracle": optimización de píxeles libres  *(decisivo)*

Implementado en `scripts/diagnose_ceiling.py --mode oracle`. Sustituye el filtro paramétrico
por un **residuo por-píxel libre** (`delta` aprendible del tamaño de la imagen, salida =
`clamp(B + delta, 0, 1)`) y corre la misma calibración por par. Es la cota superior que
*cualquier* filtro de píxeles podría alcanzar (no generaliza — es un oráculo, no un filtro
desplegable). Compara, **en el mismo par y mismo nº de pasos**:

- reducción del filtro paramétrico (overfit a un par)
- reducción del oráculo de píxeles libres

Qué decide:
- **oráculo ≫ paramétrico** (p.ej. 0.85 vs 0.33) → el cuello de botella es la **capacidad del
  filtro**. Un filtro neuronal (Prioridad 4) tiene recorrido real.
- **oráculo ≈ paramétrico** (p.ej. 0.38 vs 0.33) → el cuello de botella es la
  **representación**: el gap restante no es invertible en píxeles. Filtro neuronal NO
  ayudará; hay que cambiar capa/métrica (Prioridad 2) o modelo (Prioridad 3).

### Prioridad 2 — Suelo de ruido y descomposición de la métrica

Mide la distancia de activaciones entre A y **otra augmentación geométrica del mismo A**
(misma luz, distinto encuadre): es el **suelo irreducible** de la métrica. Resta ese suelo
del denominador para obtener la *fracción realmente recuperable*; 0.33 sobre el total puede
ser 0.6+ sobre lo recuperable. Probar además:
- `l2_rel` **por canal** del projector (no aplanado global), o ponderado por importancia
  para detección.
- Targets alternativos: salida de detección directa (ya hay modo `detection`), o capas
  intermedias del propio projector.

### Prioridad 3 — Probe de modelo  *(prueba la hipótesis "RF-DETR no es el adecuado")*

Repetir Prioridad 0+1 con un detector CNN (p.ej. YOLO/ResNet-FPN) o un RF-DETR de mayor
tamaño. Hipótesis: un backbone menos invariante a fotometría mostrará un gap base **mayor** y
**más corregible** por un filtro de píxeles (oráculo más alto). Si en CNN el oráculo sube a
0.8 y en DINOv2 se queda en 0.4, queda demostrado que el problema es el modelo, no el filtro.

### Prioridad 4 — Filtro neuronal  *(solo si Prioridad 1 muestra recorrido)*

Construir el filtro neuronal **únicamente si el oráculo demuestra headroom** sobre el
paramétrico. Aun así, evaluarlo contra **detección** (Prioridad 0), no contra distancia de
activaciones, y vigilar generalización por escena (aquí sí la capacidad puede empezar a
sobreajustar, a diferencia de los filtros actuales).

---

## Resumen de hipótesis del usuario

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| Falta de imágenes | **Descartada como límite principal** | `test ≥ train` en las 283 filas; sin sobreajuste |
| Filtros demasiado simples | **Posible, sin confirmar** | Capacidad ayuda pero satura; falta el probe oracle (Prioridad 1) |
| Mala caracterización de activaciones | **Confirmada en parte** | Nunca se validó el target contra detección (benchmark = stub); `l2_rel` global mezcla señal y ruido |
| Centrarse en ciertas capas | **Ya hecho y correcto** | projector domina; decoder inútil. El margen no está en cambiar de capa sino en validar el target |
| RF-DETR no es el mejor modelo | **Sospechoso fuerte, sin confirmar** | DINOv2 invariante a fotometría por diseño; medir gap base y oráculo cross-model (Prioridad 3) |
| Filtro neuronal | **Aplazar hasta Prioridad 1** | Solo merece la pena si el oráculo supera claramente al paramétrico |
