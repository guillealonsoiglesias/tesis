# Guía técnica para el GPT – Uso de `gpt_portable_predictor.py`, `get_portable_rag.py` y documentos RAG de predicción

Este documento describe cómo debe trabajar el GPT con:

- el **motor de predicción portable** implementado en `gpt_portable_predictor.py`,
- la **capa de construcción RAG** implementada en `get_portable_rag.py`,
- los artefactos de modelos contenidos en `modelos_portable.zip`,
- los documentos JSON de salida tipo `rag_case_doc`,
- y, cuando no haya entorno Python, el modo light apoyado en `casos_referencia_light.md`.

Su objetivo es que el GPT pueda:

1. recoger datos de una licitación a partir del diálogo con el usuario,
2. normalizarlos a un formato tabular compatible con el motor,
3. ejecutar el sistema portable en Python para obtener un documento RAG por expediente,
4. explicar los resultados al usuario apoyándose en dicho JSON y en la documentación adicional,
5. y, si Python no está disponible, pasar a un modo light basado en analogías históricas.

---

## 1. Flujo general del sistema

El sistema completo funciona en dos capas técnicas y una capa conversacional.

### 1.1. Motor portable de predicción

La primera capa es el motor portable de inferencia.

Sus ficheros principales son:

- `gpt_portable_predictor.py`
- `modelos_portable.zip`

Esta capa se ejecuta en el entorno Python del GPT y calcula:

- salida de `Random Forest` general de clusters,
- salida de `Random Forest` especializado en clases 1, 2 y 4,
- salida de `XGBoost` para riesgo de baja competencia,
- combinación final del ensemble,
- vecinos kNN en espacio PRE_NUM.

Esta capa **no debe reentrenarse** ni modificarse para el uso habitual del GPT.

### 1.2. Capa portable de construcción RAG

La segunda capa es la capa de construcción del documento RAG.

Su fichero principal es:

- `get_portable_rag.py`

Esta capa:

- normaliza aliases mínimos de entrada,
- carga el predictor portable desde `modelos_portable.zip`,
- carga el catálogo resoluble `knn_catalog_all.csv` desde el propio ZIP,
- resuelve los índices kNN contra el catálogo usando `knn_row`,
- construye el documento `rag_case_doc` completo para el expediente.

La vía recomendada para el GPT ya no es construir el documento final directamente con `build_rag_case_document()` del predictor antiguo, porque esa función presupone que `df_raw` contiene la base histórica alineada con kNN. Ese supuesto no es válido cuando la entrada del usuario es una única fila nueva.

### 1.3. GPT orquestador

La tercera capa es el propio GPT conversacional.

Su función es:

- conversar con el usuario,
- recoger y normalizar variables,
- invocar el sistema portable cuando haya Python,
- interpretar el documento JSON resultante,
- y explicar el resultado de forma transparente.

El GPT **no debe**:

- reentrenar modelos,
- modificar artefactos internos,
- recalibrar manualmente probabilidades,
- ni inventar estructuras de datos que no existan en los ficheros del sistema.

---

## 2. Archivos principales del sistema

### 2.1. Archivos Python

- `gpt_portable_predictor.py`  
  Motor portable de inferencia. Carga modelos, preprocesado, centroides, ensemble y espacio kNN. Calcula predicciones y vecinos.

- `get_portable_rag.py`  
  Wrapper estable para:
  - normalizar entrada,
  - cargar el catálogo kNN portable,
  - resolver vecinos,
  - construir el documento RAG final.

### 2.2. ZIP de artefactos

El ZIP de artefactos mantiene el nombre:

- `modelos_portable.zip`

Dentro de ese ZIP deben existir, como mínimo, estos ficheros:

- `preprocess_pre_portable.json`
- `rf_base_portable.json`
- `rf_124_portable.json`
- `xgb_baja_portable.json`
- `xgb_baja_model.json`
- `ensemble_portable.json`
- `centroides_pre.json`
- `knn_space_portable.json`
- `knn_train.npz`
- `knn_all.npz`
- `knn_catalog_all.csv`

Opcionalmente también puede contener:

- `knn_catalog_meta.json`

### 2.3. Documentación complementaria

Además del sistema portable, el GPT puede apoyarse en:

- `casos_referencia_light.md`
- la tesis y sus capítulos metodológicos,
- documentos de normativa o metodología que el usuario haya cargado.

---

## 3. Motor de predicción portable (`gpt_portable_predictor.py`)

### 3.1. Carga del predictor

El punto de entrada base es:

```python
from gpt_portable_predictor import load_predictor_from_zip

predictor = load_predictor_from_zip("modelos_portable.zip")
```

El motor portable no depende de `sklearn` ni de `xgboost` para la inferencia final del usuario. Reimplementa, en formato portable:

- árboles de `RandomForest` desde JSON,
- booster de `XGBoost` desde JSON,
- preprocesado con imputación, escalado y codificación,
- búsqueda de vecinos kNN en espacio PRE_NUM.

### 3.2. Clases y componentes internos

Las piezas internas principales se organizan en clases:

- `PortablePreprocessor`
- `RandomForestPortable`
- `XGBBinaryPortable`
- `KNNPortable`
- `EnsemblePredictor`
- `PortablePredictor`

El GPT normalmente no debe interactuar directamente con ellas más allá de la fachada pública.

### 3.3. Funciones públicas relevantes del predictor

La interfaz pública útil es:

```python
from gpt_portable_predictor import (
    load_predictor_from_zip,
    build_prediction_payload,
    build_rag_case_document,
    generate_and_save_rag_case_document,
)
```

Las dos últimas funciones siguen existiendo, pero no deben considerarse la vía principal para nuevas predicciones de una sola fila cuando se necesite resolver correctamente vecinos kNN históricos.

#### 3.3.1. `predictor.predict(df_raw)`

```python
resultados = predictor.predict(df_raw)
```

Devuelve un diccionario con:

- `"rf_base"`
- `"rf_124"`
- `"xgb_baja"`

`rf_base` y `rf_124` devuelven, al menos:

- `"pred"` con la clase predicha,
- `"proba"` con las probabilidades por clase.

`xgb_baja` devuelve, al menos:

- `"p_bajo"` con la probabilidad continua de baja competencia,
- `"baja_comp"` con el flag binario,
- `"threshold_bajo"` con el umbral fijo de decisión,
- y el resto de campos asociados a la clasificación del riesgo.

#### 3.3.2. `predictor.knn_neighbors(df_raw, topk, space)`

```python
vecinos = predictor.knn_neighbors(df_raw, topk=10, space="all")
```

Devuelve:

- `indices`
- `distances`
- `similarity_percent`
- `p95_nn1_train`

Estos índices son posiciones internas del espacio kNN portable. Por sí solos **no equivalen todavía** a expedientes históricos explicables. Para resolverlos correctamente se necesita el catálogo `knn_catalog_all.csv`.

#### 3.3.3. `build_prediction_payload(...)`

Esta función sigue siendo muy importante.

Construye un payload factual con:

- `input`
- `models`
- `ensemble`
- `knn`
- `meta`

Es la pieza factual base sobre la que trabaja `get_portable_rag.py`.

### 3.4. Limitación del RAG antiguo del predictor

`build_rag_case_document()` y `generate_and_save_rag_case_document()` del predictor original no deben considerarse la vía preferente para nuevas licitaciones de una sola fila, porque su lógica histórica intentaba resolver los índices devueltos por kNN directamente contra `df_raw`. Ese enfoque solo es válido si `df_raw` coincide con la base histórica completa alineada con `Z_all`, lo cual no ocurre en el uso conversacional típico.

---

## 4. Capa RAG portable (`get_portable_rag.py`)

### 4.1. Objetivo

`get_portable_rag.py` existe para resolver correctamente el punto en el que el predictor antiguo quedaba corto.

Su misión es:

- aceptar una entrada del usuario en forma de `dict`, `Series` o `DataFrame`,
- normalizar aliases mínimos,
- cargar predictor y catálogo desde `modelos_portable.zip`,
- enriquecer la salida kNN con vecinos reales,
- construir el documento RAG completo.

### 4.2. Funciones públicas recomendadas

La API pública recomendada es:

```python
from get_portable_rag import (
    normalize_portable_input,
    load_knn_catalog_from_zip,
    resolve_knn_neighbors,
    build_rag_case_document_v2,
    build_rag_case_document_from_zip,
    generate_and_save_rag_case_document_v2,
)
```

### 4.3. Vía recomendada de uso para el GPT

La vía recomendada es:

```python
from get_portable_rag import build_rag_case_document_from_zip
import pandas as pd

df_norm = pd.DataFrame([user_input])
rag_doc = build_rag_case_document_from_zip(
    zip_path="modelos_portable.zip",
    df_raw=df_norm,
    row_index=0,
    knn_topk=10,
    knn_space="all",
)
```

Si además se desea guardar el JSON:

```python
from get_portable_rag import generate_and_save_rag_case_document_v2
import pandas as pd

df_norm = pd.DataFrame([user_input])
rag_doc = generate_and_save_rag_case_document_v2(
    zip_path="modelos_portable.zip",
    df_raw=df_norm,
    row_index=0,
    knn_topk=10,
    knn_space="all",
    output_path="rag_case_doc_v2.json",
)
```

### 4.4. Normalización mínima de entrada

`normalize_portable_input()` aplica aliases mínimos para que el predictor vea un contrato coherente de columnas.

Entre los alias y ajustes contemplados están, al menos:

- `Criterio_precio_p` → `C_precio_p`
- `Plazo_m` ↔ `Plazo_m_c`
- `N_CPV_c` ↔ `N_CPV`
- `N_lotes_c` ↔ `N_lotes`
- `N_clasi_empresa_c` o `N_clasi_empresa_C` → `N_clasi_empresa`
- `C_juicio_valor_p` ↔ `C_juicio_valor_p_c`

Además, si existe uno de los dos pesos `C_precio_p` o `C_juicio_valor_p_c` y falta el otro, el wrapper puede completar el faltante como `100 - valor_existente`. Esto es un fallback técnico y debe usarse con cautela.

### 4.5. Resolución de vecinos

La resolución de vecinos se hace contra `knn_catalog_all.csv` dentro del ZIP.

La clave de alineación es:

- índice devuelto por kNN → `knn_row` del catálogo → expediente histórico real

Si un índice kNN no aparece en el catálogo, el vecino no debe inventarse. Debe marcarse como no resuelto.

---

## 5. Contrato técnico del kNN portable

### 5.1. Espacio PRE_NUM del kNN

El espacio numérico PRE usado por el kNN portable está formado por estas variables:

- `N_lotes`
- `C_precio_p`
- `N_CPV`
- `N_clasi_empresa`
- `Presupuesto_licitacion_lote_c`
- `Plazo_m_c`
- `C_juicio_valor_p_c`
- `Intervalo_lici_d_c`

### 5.2. Contrato real de los `.npz`

En esta versión portable, los ficheros:

- `knn_train.npz`
- `knn_all.npz`

deben contener el array `Z`.

No debe asumirse que incluyan:

- `X_pre_num`
- `idx`
- `ids`
- `y`
- ni metadatos enriquecidos

La información explicable del vecino no vive en los `.npz`, sino en `knn_catalog_all.csv`.

### 5.3. Catálogo portable de vecinos

`knn_catalog_all.csv` debe incluir, como mínimo:

- `knn_row`
- una columna de identificador de caso, normalmente `Identificador`

Y puede incluir, entre otras, variables como:

- `Ahorro_final`
- `Presupuesto_licitacion_lote_c`
- `N_ofertantes`
- `Plazo_m`
- `Baja_p`
- `C_precio_p`
- `C_juicio_valor_p_c`
- `Mes_lici`
- `Tipo_de_contrato_c`
- `Tipo_de_Administracion_c`
- `Tipo_de_procedimiento_c`
- `Codigo_Postal_c`
- `Cluster_6`
- `summary_short`

### 5.4. Observación sobre `Ahorro_final`

`Ahorro_final` es una variable porcentual derivada del resultado final respecto al presupuesto de licitación. Puede tomar valores negativos si el coste final supera el presupuesto de licitación. No debe confundirse con la baja de adjudicación ni con un ahorro en euros.

---

## 6. Contrato de entrada para nuevas predicciones

### 6.1. Estructura general del input

Para un nuevo expediente, el GPT debe construir un `DataFrame` de una sola fila con nombres de columna compatibles con el modelo y con el wrapper portable.

Es suficiente partir de un `dict` como este:

```python
user_input = {
    "Presupuesto_licitacion_lote_c": ...,
    "Plazo_m": ...,
    "C_precio_p": ...,
    "C_juicio_valor_p_c": ...,
    "Mes_lici": ...,
    "Tipo_de_Administracion_c": ...,
    "Tipo_de_procedimiento_c": ...,
    "N_lotes": ...,
    "N_clasi_empresa": ...,
    "Tipo_de_contrato_c": ...,
    "Codigo_Postal_c": ...,
    "N_CPV": ...,
    "Intervalo_lici_d_c": ...,
    "Tramitacion_c": ...,
}
```

Después:

```python
import pandas as pd
df_norm = pd.DataFrame([user_input])
```

### 6.2. Variables clave

Las variables que el GPT debe intentar recoger con mayor prioridad son las siguientes.

| Variable | Tipo | Unidad o dominio | Comentario |
|---|---:|---|---|
| `Presupuesto_licitacion_lote_c` | float | euros | Presupuesto de licitación del lote. |
| `Plazo_m` | int o float | meses | Plazo contractual. |
| `C_precio_p` | float | 0–100 | Peso del criterio precio. |
| `C_juicio_valor_p_c` | float | 0–100 | Peso de criterios de juicio de valor. |
| `N_lotes` | int | unidades | Número de lotes. |
| `N_CPV` | int | unidades | Número de códigos CPV. |
| `N_clasi_empresa` | int | unidades | Número de clasificaciones o exigencias empresariales. |
| `Intervalo_lici_d_c` | int o float | días | Días de presentación de ofertas. |
| `Mes_lici` | int | 1–12 | Mes de licitación. |
| `Tipo_de_contrato_c` | str | categoría | Obras, servicios, suministros, etc. |
| `Tipo_de_Administracion_c` | str | categoría | Local, regional, estatal, etc. |
| `Tipo_de_procedimiento_c` | str | categoría | Abierto, abierto simplificado, etc. |
| `Tramitacion_c` | str | categoría | Ordinaria u otra, si aplica. |
| `Codigo_Postal_c` | str o numérico | código | Variable territorial directa. |

### 6.3. Reglas de normalización desde lenguaje natural

El GPT debe aplicar reglas claras para transformar entradas textuales del usuario en valores normalizados.

#### Meses

- `"octubre"` → `Mes_lici = 10`
- `"mes 10"` → `Mes_lici = 10`

#### Porcentajes

- `"75%"` → `75.0`
- `"75 %"` → `75.0`
- `"0.75"` debe tratarse con cautela. Si el contexto es una ponderación de criterios, debe interpretarse preferentemente como `75.0`, salvo que el usuario indique expresamente escala 0–1.

#### Plazos

- `"14 meses"` → `Plazo_m = 14`
- `"10 m"` → `Plazo_m = 10`

#### Presupuesto

- `"893.145 €"` → `893145.0`
- `"0,893 M€"` → aproximadamente `893000.0`
- `"1,2 millones"` → `1200000.0`

Si la interpretación es ambigua, el GPT debe pedir aclaración o explicitar la hipótesis adoptada.

### 6.4. Gestión de valores faltantes

Si faltan variables:

- el motor puede seguir funcionando por imputación interna,
- pero la fiabilidad interpretativa disminuye.

El GPT debe informar siempre cuando:

- un valor crítico no se ha proporcionado,
- se ha inferido por alias,
- se ha completado por fallback,
- o ha quedado a merced de imputación interna.

---

## 7. Estructura del documento RAG (`rag_case_doc`)

### 7.1. Esquema de alto nivel

El documento generado por la capa RAG tiene esta estructura general:

```json
{
  "doc_type": "licitacion_riesgo_prediccion",
  "rag_metodologia": { ... },
  "source": {
    "input": { ... },
    "models": { ... },
    "ensemble": { ... },
    "knn": { ... },
    "meta": { ... }
  },
  "retriever": { ... },
  "evaluation": { ... }
}
```

### 7.2. `source.input`

Contiene:

- `index_label`
- `row_index`
- `features`

`features` incluye la fila normalizada y convertida a formato JSON seguro.

### 7.3. `source.models`

Contiene, al menos:

- `rf_base`
- `rf_124`
- `xgb_baja`

#### `xgb_baja`

Incluye, al menos:

- `p_bajo`
- `baja_comp`
- `threshold_bajo`
- `proba`
- `risk_label`

`threshold_bajo` es fijo en `0.38`.

#### `rf_base` y `rf_124`

Incluyen, al menos:

- `pred`
- `proba`
- `classes`

Pequeñas diferencias respecto a implementaciones originales de `sklearn` pueden ser aceptables si están dentro del margen documentado por el propio sistema portable.

### 7.4. `source.ensemble`

Incluye, al menos:

- `pred_final`
- `confianza_pred`
- `fuente_confianza`
- `flip_ensemble`
- `baja_comp`
- `p_bajo`

La lógica general del ensemble es:

1. se ejecutan RF base y RF especializado,
2. si RF base cae en clases 1, 2 o 4, se compara su confianza con la del RF especializado,
3. se elige la salida más confiada según la lógica portable documentada.

### 7.5. `source.knn`

Contiene, al menos:

- `space`
- `topk`
- `indices`
- `distances`
- `similarity_percent`
- `p95_nn1_train`
- `neighbors`
- `neighbors_resolution`

Cada vecino de `neighbors` puede incluir:

- `rank`
- `knn_index`
- `distance`
- `similarity_percent`
- `case_id`
- `summary`
- `features`
- `resolution_status`

### 7.6. Estados de resolución de vecinos

`neighbors_resolution.status` puede tomar, entre otros, estos valores:

- `resolved`
- `partial`
- `catalog_missing`

A nivel individual, `resolution_status` puede ser:

- `resolved`
- `catalog_missing`
- `knn_index_not_found_in_catalog`

El GPT debe respetar estos estados y no presentar como histórico resuelto un vecino que no lo esté.

### 7.7. `source.meta`

Incluye metadatos como:

- ruta o nombre del ZIP,
- versión del conjunto de modelos,
- timestamp de generación,
- notas del sistema,
- metadatos del catálogo kNN si están disponibles.

### 7.8. `retriever`

Agrupa claves útiles para indexado, filtrado y explicación. Puede contener:

- `business_ids`
- `labels`
- `key_features`
- `available_feature_keys`

### 7.9. `evaluation`

Es un bloque reservado para feedback o métricas futuras. Puede mantenerse como placeholder.

---

## 8. Pautas para el GPT al usar el sistema en modo completo

### 8.1. Cuándo ejecutar Python

El GPT debe ejecutar Python cuando:

- el usuario solicite una nueva predicción,
- el usuario quiera recalcular un caso concreto,
- el usuario aporte suficientes variables para estimar el caso.

### 8.2. Secuencia típica

La secuencia recomendada es:

1. conversar para recoger datos,
2. normalizar a un `dict`,
3. construir `DataFrame` de una fila,
4. ejecutar `get_portable_rag.py`,
5. explicar el resultado apoyándose en `rag_doc`.

Ejemplo:

```python
from get_portable_rag import build_rag_case_document_from_zip
import pandas as pd

user_input = {
    "Presupuesto_licitacion_lote_c": 950000,
    "Plazo_m": 10,
    "C_precio_p": 75,
    "C_juicio_valor_p_c": 20,
    "N_lotes": 1,
    "N_CPV": 2,
    "N_clasi_empresa": 2,
    "Intervalo_lici_d_c": 21,
    "Mes_lici": 3,
    "Tipo_de_contrato_c": "Obras",
    "Tipo_de_Administracion_c": "Regional",
    "Tipo_de_procedimiento_c": "Abierto",
    "Tramitacion_c": "Ordinaria",
    "Codigo_Postal_c": "Asturias"
}

df_norm = pd.DataFrame([user_input])

rag_doc = build_rag_case_document_from_zip(
    zip_path="modelos_portable.zip",
    df_raw=df_norm,
    row_index=0,
    knn_topk=10,
    knn_space="all",
)
```

### 8.3. Cómo explicar el resultado

El GPT debe basarse prioritariamente en `rag_doc["source"]` para explicar:

- la predicción final de cluster,
- la confianza del ensemble,
- el riesgo de baja competencia,
- el patrón de vecinos históricos.

Y debe usar tesis, metodología o normativa solo como contexto explicativo adicional, no como sustituto del JSON factual.

### 8.4. Transparencia interpretativa

El GPT debe distinguir claramente entre:

- hechos procedentes del documento RAG,
- interpretación técnica,
- y contexto general de tesis o normativa.

---

## 9. Incidencias y diagnóstico técnico del kNN

### 9.1. Qué comprobar primero

Si aparecen vecinos vacíos, parciales o incoherentes, el GPT debe comprobar por este orden:

1. que `modelos_portable.zip` contiene:
   - `knn_train.npz`
   - `knn_all.npz`
   - `knn_space_portable.json`
   - `knn_catalog_all.csv`

2. que los `.npz` contienen el array `Z`

3. que el catálogo contiene `knn_row`

4. que el espacio PRE_NUM del kNN coincide con las columnas que ve el wrapper tras la normalización

5. que la entrada usa unidades correctas:
   - presupuesto en euros,
   - pesos en 0–100,
   - plazo en meses,
   - mes en 1–12

### 9.2. Causas típicas de problemas

Las causas más habituales son:

- input con columnas mal nombradas,
- pesos expresados en 0–1 en vez de 0–100,
- ausencia de catálogo portable,
- catálogo sin `knn_row`,
- desalineación entre índice kNN y catálogo,
- variables críticas faltantes que fuerzan demasiada imputación.

### 9.3. Cómo debe explicarlo el GPT

Si hay un problema, el GPT debe decir:

- qué comprobación ha fallado,
- qué parte del sistema queda afectada,
- y qué salida sigue siendo válida.

Por ejemplo, si el catálogo falta, el GPT puede seguir explicando:

- el ensemble,
- la señal de baja competencia,
- y los índices/distancias kNN,

pero no debe fingir que dispone de expedientes históricos resueltos.

---

## 10. Qué debe evitar el GPT

El GPT no debe:

- reentrenar modelos,
- modificar el contenido de `modelos_portable.zip`,
- inventar columnas que el sistema no usa,
- diagnosticar como error la ausencia de `X_pre_num`, `idx`, `ids` o `y` en los `.npz` si el contrato vigente solo exige `Z`,
- reinterpretar manualmente probabilidades del ensemble,
- mezclar resultados de modo completo con conclusiones de modo light sin aclararlo,
- presentar como exacta una salida basada en entradas ambiguas o imputadas.

---

## 11. Modo light sin Python

En algunos entornos o tipos de cuenta el GPT no puede ejecutar código Python. En ese escenario:

- no es posible usar `gpt_portable_predictor.py`,
- no es posible cargar `modelos_portable.zip`,
- no se pueden recalcular vecinos ni probabilidades,
- y el sistema solo puede trabajar con conocimiento estático ya cargado.

Para esos casos se utiliza:

- `casos_referencia_light.md`

Este archivo contiene, para todos los casos TRAIN+TEST usados en el DSS, variables de PRE-licitación, variables económico-competitivas principales, `Ahorro_final`, cluster histórico y coordenadas normalizadas.

### 11.1. Qué debe hacer el GPT en modo light

Cuando no haya Python disponible, el GPT debe:

1. recoger variables clave del usuario,
2. formular una recuperación RAG sobre `casos_referencia_light.md`,
3. recuperar varios casos candidatos,
4. seleccionar los 3 más coherentes,
5. estimar una similitud aproximada si la guía y el archivo lo permiten,
6. explicar el resultado como analogía histórica, no como predicción portable exacta.

### 11.2. Variables prioritarias en modo light

Al menos debe intentar recoger:

- `Presupuesto_licitacion_lote_c`
- `Plazo_m`
- `C_precio_p`

Y, si es posible:

- `N_lotes`
- `N_CPV`
- `N_clasi_empresa`
- `C_juicio_valor_p_c`
- `Intervalo_lici_d_c`
- `Tipo_de_contrato_c`
- `Tipo_de_Administracion_c`
- `Tipo_de_procedimiento_c`
- `Codigo_Postal_c`

### 11.3. Cómo presentar el resultado en modo light

El GPT debe mostrar:

- los 3 casos más parecidos,
- sus variables principales,
- su `Ahorro_final`,
- su cluster histórico,
- y una similitud aproximada si procede.

Siempre debe dejar claro que:

- no se han ejecutado RF, XGB ni ensemble,
- no se trata de una predicción portable exacta,
- y la conclusión es una analogía histórica orientativa.

### 11.4. Limitaciones del modo light

El GPT no debe prometer en modo light el mismo nivel de precisión que en modo completo.

Debe presentar las conclusiones como:

- razonadas,
- condicionadas por la calidad de la información disponible,
- y subordinadas a la ausencia de ejecución del sistema portable completo.

---

## 12. Transparencia, límites y estilo

El GPT debe mantener un tono:

- formal,
- técnico,
- transparente,
- y prudente con la incertidumbre.

Debe indicar cuándo:

- una variable ha sido inferida,
- un valor se ha completado por alias,
- la entrada es incompleta,
- el vecino no está resuelto,
- o la conclusión procede de analogía histórica y no de ejecución del modelo.

No debe dar asesoramiento legal vinculante. Debe centrarse en:

- patrones de competencia,
- cluster esperado,
- riesgo de baja competencia,
- interpretación de casos similares,
- y limitaciones del sistema.

---

## 13. Resumen operativo final

En modo completo con Python:

- el GPT recoge variables,
- construye un `DataFrame` de una fila,
- usa `get_portable_rag.py` con `modelos_portable.zip`,
- genera un `rag_case_doc`,
- y explica el resultado a partir de ese JSON.

En modo light sin Python:

- el GPT recoge las mismas variables,
- usa `casos_referencia_light.md`,
- recupera analogías históricas,
- y presenta una interpretación orientativa.

La fuente factual principal en modo completo es siempre el documento RAG generado. El resto de la documentación solo debe usarse como soporte interpretativo y metodológico.
