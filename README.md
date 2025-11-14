# 🧬 Genetic Algorithms - Complete Tutorial Series

**Tutoriales profesionales sobre Algoritmos Genéticos desde nivel básico hasta avanzado**

Este repositorio contiene una serie completa y profesional de tutoriales sobre **Algoritmos Genéticos (GAs)**, diseñados con el mismo estándar de calidad que cursos universitarios y profesionales de machine learning.

---

## 📚 Estructura del Repositorio

### 🎓 Tutoriales Disponibles

| Tutorial | Nivel | Descripción | Estado |
|----------|-------|-------------|--------|
| **GA_Tutorial_1_Basics** | Básico | Introducción y fundamentos de GAs | ✅ Completo |
| **GA_Tutorial_2_Intermediate** | Intermedio | Operadores avanzados y estrategias de selección | ✅ Completo |
| **GA_Tutorial_3_Advanced** | Avanzado | TSP, híbridos, multi-objetivo | ✅ Completo |
| **GA_Tutorial_4_Applications** | Aplicado | Aplicaciones reales y casos de uso | ✅ Completo |
| **GA_Tutorial_5_Specialized** | Especializado | CMA-ES, Differential Evolution, PSO | ✅ Completo |
| **W2A1** | Referencia | Métodos de optimización (material original) | ✅ Completo |

### 🔬 Casos de Estudio de Data Science

| Caso de Estudio | Descripción | Estado |
|----------------|-------------|--------|
| **AutoML** | Optimización automática de modelos ML con GA | ✅ Completo |
| **Feature Engineering** | Ingeniería de características automática | ✅ Completo |
| **Ensemble Optimization** | Optimización de ensembles con GA | ✅ Completo |
| **Time Series** | Predicción de series temporales | ✅ Completo |

### 💼 Caso de Estudio Industrial

| Caso de Estudio | Descripción | Estado |
|----------------|-------------|--------|
| **Tutorial_5_Industrial_Case_Study** | **NUEVO:** Vehicle Routing Problem (VRPTW) - Optimización de rutas de delivery | ✅ Completo |

### 🛠️ Herramientas y Utilidades

| Componente | Descripción | Estado |
|-----------|-------------|--------|
| **Interactive Dashboard** | Visualización en tiempo real con Plotly Dash | ✅ Completo |
| **sklearn Integration** | Integración con scikit-learn (GAClassifier, GAFeatureSelector) | ✅ Completo |
| **Benchmarking Suite** | Suite automatizada de benchmarking | ✅ Completo |

---

## 🎯 Tutorial 1: Introducción a Algoritmos Genéticos

**Nivel:** Básico | **Duración estimada:** 4-6 horas

### Contenido

- ✅ ¿Qué son los algoritmos genéticos?
- ✅ Inspiración biológica y evolución
- ✅ Codificación binaria y real
- ✅ Función de fitness
- ✅ Operadores básicos: selección, cruce, mutación
- ✅ Implementación completa de GA
- ✅ Optimización de funciones benchmark

### Archivos Incluidos

```
GA_Tutorial_1_Basics/
├── GA_Introduction.ipynb        # Notebook principal con ejercicios
├── ga_utils_basics.py           # Funciones de utilidad
├── public_tests.py              # Tests automatizados
├── testCases.py                 # Generadores de casos de prueba
├── README.md                    # Documentación detallada
├── datasets/                    # Datos de ejemplo
└── images/                      # Visualizaciones
```

### Conceptos Clave

- **Población:** Conjunto de soluciones candidatas
- **Cromosoma:** Representación codificada de una solución
- **Fitness:** Medida de calidad de una solución
- **Selección:** Ruleta, Torneo
- **Cruce:** 1-punto, 2-puntos, Uniforme
- **Mutación:** Bit-flip, Gaussiana

### Funciones Benchmark

| Función | Fórmula | Óptimo | Dificultad |
|---------|---------|---------|------------|
| Sphere | Σ x_i² | f(0,...,0) = 0 | Fácil |
| Rastrigin | 10n + Σ[x_i² - 10cos(2πx_i)] | f(0,...,0) = 0 | Difícil |
| Rosenbrock | Σ[100(x_{i+1}-x_i²)² + (1-x_i)²] | f(1,...,1) = 0 | Media |

### Ejercicios Prácticos

1. ✏️ Inicialización de población (binaria y real)
2. ✏️ Evaluación de fitness
3. ✏️ Implementación de selección
4. ✏️ Implementación de cruce
5. ✏️ Implementación de mutación
6. ✏️ GA completo
7. ✏️ Optimización con codificación binaria
8. ✏️ Optimización con codificación real

---

## 🚀 Tutorial 2: Operadores Genéticos Avanzados

**Nivel:** Intermedio | **Duración estimada:** 6-8 horas

### Contenido

- ✅ Selección por ranking
- ✅ Stochastic Universal Sampling (SUS)
- ✅ Selección de Boltzmann
- ✅ Cruce aritmético y BLX-α
- ✅ Simulated Binary Crossover (SBX)
- ✅ Mutación polinomial
- ✅ Parámetros adaptativos
- ✅ Elitismo y estrategias de reemplazo
- ✅ Manejo de restricciones
- ✅ Mantenimiento de diversidad

### Archivos Incluidos

```
GA_Tutorial_2_Intermediate/
├── GA_Advanced_Operators.ipynb  # Notebook principal
├── ga_utils_intermediate.py     # Utilidades avanzadas
├── public_tests.py              # Tests
├── testCases.py                 # Casos de prueba
├── README.md                    # Documentación
├── datasets/                    # Problemas de benchmark
└── images/                      # Visualizaciones
```

### Técnicas Avanzadas

#### Selección Avanzada
- **Rank-based:** Basada en ranking, no en fitness absoluto
- **SUS:** Muestreo universal estocástico
- **Boltzmann:** Selección basada en temperatura
- **Niching:** Mantenimiento de nichos

#### Cruce Avanzado
- **Aritmético:** Combinación lineal de padres
- **BLX-α:** Blend crossover con extensión
- **SBX:** Simulated Binary Crossover (usado en NSGA-II)

#### Mutación Avanzada
- **Polinomial:** Distribución polinomial de probabilidad
- **Adaptativa:** Tasa que decrece con generaciones
- **Auto-adaptativa:** Parámetros que evolucionan con la solución

---

## 🎯 Tutorial 3: Optimización Avanzada

**Nivel:** Avanzado | **Duración estimada:** 8-10 horas

### Contenido Planificado

- 🚧 Algoritmos Genéticos Multiobjetivo
- 🚧 NSGA-II (Non-dominated Sorting GA)
- 🚧 Problema del Viajante (TSP)
- 🚧 Optimización combinatoria
- 🚧 Algoritmos híbridos (GA + Búsqueda Local)
- 🚧 Modelos de islas y computación paralela
- 🚧 Análisis de convergencia

---

## 🏆 Tutorial 4: Aplicaciones Reales

**Nivel:** Aplicado | **Duración estimada:** 10-12 horas

### Contenido

- ✅ Optimización de hiperparámetros en ML
- ✅ Feature selection con GAs
- ✅ Scheduling y planificación de trabajos
- ✅ Optimización de portafolios financieros
- ✅ Neural Architecture Search (NAS) básico
- ✅ Casos de uso con datos reales

### Aplicaciones Implementadas

- **Hyperparameter Optimization:** Búsqueda de mejores hiperparámetros para modelos ML
- **Feature Selection:** Selección automática de características relevantes
- **Job Scheduling:** Minimización de makespan en scheduling de trabajos
- **Portfolio Optimization:** Balance riesgo-retorno en portafolios de inversión
- **Neural Architecture Search:** Diseño automático de arquitecturas de redes neuronales

---

## 🚀 Tutorial 5: Algoritmos Especializados

**Nivel:** Especializado | **Duración estimada:** 8-10 horas

### Contenido

- ✅ **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy)
  - Adaptación automática de tamaño de paso
  - Matriz de covarianza para dirección de búsqueda
  - Interfaz ask-tell
  - Mejor para optimización continua

- ✅ **Differential Evolution** (DE)
  - Mutación: v = x_r1 + F*(x_r2 - x_r3)
  - Crossover binomial
  - Excelente para problemas multimodales
  - Simple y efectivo

- ✅ **Particle Swarm Optimization** (PSO)
  - Inteligencia de enjambre
  - Componentes cognitiva y social
  - Convergencia rápida
  - Pocos parámetros

### Comparaciones de Rendimiento

Incluye benchmarks completos comparando:
- Velocidad de convergencia
- Tasa de éxito
- Escalabilidad con dimensionalidad
- Complejidad computacional

---

## 📊 Casos de Estudio de Data Science

**Ubicación:** `DataScience_Case_Studies/`

### Case Study 1: AutoML con Algoritmos Genéticos

**Nivel:** Avanzado | **Archivo:** `automl/Case_Study_1_AutoML.ipynb`

**Contenido:**
- Optimización simultánea de modelo + hiperparámetros + features
- Codificación híbrida (binaria + real)
- Validación cruzada estratificada
- Comparación con baseline methods
- Dataset: Breast Cancer Wisconsin

**Resultados esperados:**
- 2-5% mejora en accuracy vs tuning manual
- Reducción de features 30-50%
- Automatización completa del pipeline

### Case Study 2: Feature Engineering con GAs

**Nivel:** Avanzado | **Archivo:** `feature_engineering/Case_Study_2_Feature_Engineering.ipynb`

**Contenido:**
- Generación automática de features
- Operaciones unarias: square, sqrt, log, exp, abs, inv
- Operaciones binarias: add, subtract, multiply, divide, max, min
- Prevención de overfitting
- Análisis de importancia de features

**Resultados esperados:**
- 5-15% mejora en accuracy
- Descubrimiento de interacciones no obvias
- Reducción de esfuerzo manual

### Case Study 3: Ensemble Optimization

**Nivel:** Avanzado | **Archivo:** `ensemble_optimization/Case_Study_3_Ensemble_Optimization.ipynb`

**Contenido:**
- Optimización de pesos en voting ensembles
- Selección automática de modelos
- Configuración de meta-learners (stacking)
- Balance diversidad vs accuracy
- Dataset: Breast Cancer Wisconsin

**Resultados esperados:**
- 2-5% mejora vs mejor modelo individual
- 1-3% mejora vs ensemble uniforme
- 30-50% reducción de modelos

### Case Study 4: Time Series Forecasting

**Nivel:** Avanzado | **Archivo:** `time_series/Case_Study_4_Time_Series_Optimization.ipynb`

**Contenido:**
- Selección de features de lag
- Optimización de ventanas rolling
- Tuning de hiperparámetros
- Validación temporal correcta
- Métricas: RMSE, MAE, MAPE

**Resultados esperados:**
- 10-25% reducción RMSE vs lag-1 simple
- 30-60% mejora vs persistencia naive
- Feature engineering automático

---

## 📈 Interactive Dashboard

**Ubicación:** `Interactive_Dashboard/`

### Características

**Dashboard con Plotly Dash** (`ga_dashboard.py`):
- Visualización en tiempo real de optimización GA
- 4 problemas benchmark (Sphere, Rastrigin, Rosenbrock, Ackley)
- Controles interactivos de parámetros
- Gráficos actualizados en vivo:
  - Evolución de fitness
  - Diversidad de población
  - Distribución 2D de individuos
  - Tasa de convergencia

**Visualizador Simple** (`simple_visualizer.py`):
- Visualización con matplotlib (sin servidor web)
- Modo estático y animado
- Análisis de importancia de features
- Comparación de algoritmos

### Uso

```bash
# Dashboard interactivo
cd Interactive_Dashboard
python ga_dashboard.py
# Abrir http://localhost:8050

# Visualizador simple
python simple_visualizer.py
```

---

## 🤖 Integración con scikit-learn

**Ubicación:** `sklearn_integration/`

### Componentes

#### GAClassifier y GARegressor
```python
from ga_sklearn import GAClassifier
from sklearn.ensemble import RandomForestClassifier

param_space = {
    'n_estimators': (10, 200, 'int'),
    'max_depth': (3, 20, 'int')
}

ga_clf = GAClassifier(
    estimator=RandomForestClassifier(),
    param_space=param_space,
    pop_size=30,
    max_generations=20
)

ga_clf.fit(X_train, y_train)
print(f"Best params: {ga_clf.best_params_}")
print(f"Best score: {ga_clf.best_score_:.4f}")
```

#### GAFeatureSelector
```python
from ga_sklearn import GAFeatureSelector
from sklearn.linear_model import LogisticRegression

selector = GAFeatureSelector(
    estimator=LogisticRegression(),
    max_generations=20
)

selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
print(f"Selected features: {selector.selected_features_}")
```

### Ventajas

- **Compatible con pipelines de sklearn**
- **Interfaz familiar** (fit/predict/transform)
- **Funciona con cross_val_score**
- **Drop-in replacement** para GridSearchCV
- **Más rápido** que grid search exhaustivo

---

## 🏁 Benchmarking Suite

**Ubicación:** `Benchmarking_Suite/`

### Características

- **Comparación automatizada** de 4 algoritmos (GA, CMA-ES, DE, PSO)
- **4 funciones benchmark** estándar
- **Múltiples dimensiones** (5D, 10D, 20D)
- **Análisis estadístico** completo
- **10 ejecuciones independientes** por configuración

### Uso

```bash
cd Benchmarking_Suite
python ga_benchmark.py
```

### Salidas Generadas

1. **benchmark_report.txt**: Análisis detallado con estadísticas
2. **benchmark_results.csv**: Datos crudos para análisis posterior
3. **benchmark_results.json**: Resultados estructurados
4. **benchmark_plots.png**: Visualizaciones de alta calidad

### Métricas Reportadas

- Best, Mean, Median, Worst fitness
- Standard deviation (consistencia)
- Success rate (% alcanzando óptimo)
- Average time (tiempo de ejecución)
- Scalability analysis (rendimiento vs dimensión)

### Ejemplo de Resultados

```
Algorithm     Mean Fitness    Success Rate    Avg Time
CMA-ES        1.234e-05      78.3%           2.1s
DE            2.456e-04      65.0%           1.8s
PSO           3.789e-04      58.3%           1.2s
Standard_GA   5.123e-03      45.0%           2.5s
```

---

## 🛠️ Instalación y Requisitos

### Requisitos del Sistema

- Python 3.7 o superior
- Jupyter Notebook
- Librerías: NumPy, Matplotlib, SciPy

### Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd GNETIC-ALGORITHMS

# Instalar dependencias
pip install numpy matplotlib scipy jupyter

# Iniciar Jupyter
jupyter notebook
```

### Inicio Rápido

1. **Tutorial 1 (Básico):**
   ```bash
   cd GA_Tutorial_1_Basics
   jupyter notebook GA_Introduction.ipynb
   ```

2. **Ejecutar Tests:**
   ```bash
   cd GA_Tutorial_1_Basics
   python public_tests.py
   ```

---

## 📖 Guía de Uso

### Para Principiantes

1. Comienza con **Tutorial 1** - Fundamentos
2. Completa todos los ejercicios en orden
3. Ejecuta los tests para validar tu código
4. Experimenta con diferentes parámetros

### Para Intermedios

1. Revisa rápidamente **Tutorial 1**
2. Enfócate en **Tutorial 2** - Operadores Avanzados
3. Compara el rendimiento de diferentes técnicas
4. Implementa tus propias variaciones

### Para Avanzados

1. **Tutorial 3** - Optimización Avanzada
2. **Tutorial 4** - Aplicaciones Reales
3. Contribuye con nuevos ejemplos y casos de uso

---

## 🎓 Objetivos de Aprendizaje

Al completar esta serie de tutoriales, serás capaz de:

✅ Comprender los fundamentos teóricos de los algoritmos genéticos
✅ Implementar GAs desde cero en Python
✅ Seleccionar operadores apropiados para diferentes problemas
✅ Ajustar parámetros para optimizar el rendimiento
✅ Manejar restricciones y múltiples objetivos
✅ Aplicar GAs a problemas reales de optimización
✅ Analizar y visualizar el comportamiento de GAs
✅ Diseñar algoritmos híbridos y variantes personalizadas

---

## 📊 Características Destacadas

### ✨ Calidad Profesional

- Material diseñado con estándares académicos
- Código limpio, documentado y probado
- Tests automatizados para cada componente
- Visualizaciones interactivas

### 📚 Contenido Completo

- Desde conceptos básicos hasta técnicas avanzadas
- 40+ ejercicios prácticos
- 15+ problemas de benchmark
- Casos de uso del mundo real

### 🔬 Enfoque Práctico

- Implementaciones desde cero (no black-box)
- Ejemplos ejecutables y modificables
- Comparaciones de rendimiento
- Experimentos reproducibles

### 🎯 Pedagógicamente Estructurado

- Progresión lógica de conceptos
- Ejercicios graduados en dificultad
- Retroalimentación inmediata mediante tests
- Referencias a literatura académica

---

## 🔧 Estructura de Cada Tutorial

Cada tutorial sigue una estructura consistente:

```
Tutorial_X/
│
├── README.md                    # Guía completa del tutorial
├── Notebook.ipynb               # Tutorial interactivo principal
│
├── Código Base:
│   ├── ga_utils_*.py           # Funciones de utilidad
│   ├── public_tests.py         # Tests públicos
│   └── testCases.py            # Generadores de datos
│
├── Recursos:
│   ├── datasets/               # Datos de ejemplo
│   └── images/                 # Diagramas y visualizaciones
│
└── Ejercicios:
    ├── Implementaciones guiadas
    ├── Tests automatizados
    └── Soluciones esperadas
```

---

## 📈 Progreso y Roadmap

### ✅ Completado

**Tutoriales Principales:**
- [x] Tutorial 1: Fundamentos completos (8 ejercicios, tests, visualizaciones)
- [x] Tutorial 2: Operadores avanzados completo (17 tests, SBX, NSGA-II concepts)
- [x] Tutorial 3: TSP, híbridos, multi-objetivo (8 tests, 2-opt, Pareto fronts)
- [x] Tutorial 4: Aplicaciones reales (hyperparameters, features, scheduling, portfolio)
- [x] Tutorial 5: Algoritmos especializados (CMA-ES, DE, PSO con 8 tests)

**Casos de Estudio:**
- [x] Case Study 1: AutoML completo con GA
- [x] Case Study 2: Feature Engineering automático
- [x] Case Study 3: Ensemble Optimization
- [x] Case Study 4: Time Series Forecasting

**Herramientas y Utilidades:**
- [x] Interactive Dashboard con Plotly Dash
- [x] Simple Visualizer con matplotlib
- [x] Integración completa con scikit-learn (GAClassifier, GARegressor, GAFeatureSelector)
- [x] Benchmarking Suite automatizado

**Infraestructura:**
- [x] Módulo de utilidades para cada tutorial
- [x] Sistema de tests automatizados (60+ tests totales)
- [x] Funciones benchmark y problemas reales
- [x] Visualizaciones interactivas y Gantt charts
- [x] Documentación profesional completa (READMEs detallados)

### 🎯 Próximas Características Potenciales

- [ ] Ejemplos de GA paralelo con multiprocessing
- [ ] MCP server para Claude Code integration
- [ ] Videos tutoriales complementarios
- [ ] Notebooks interactivos con widgets
- [ ] Más casos de estudio (NLP, Computer Vision)
- [ ] Integración con frameworks de deep learning (PyTorch, TensorFlow)

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Formas de contribuir:

1. 🐛 Reportar bugs o errores
2. 💡 Sugerir mejoras o nuevos tutoriales
3. 📝 Mejorar documentación
4. 🎨 Añadir visualizaciones
5. 🧪 Agregar casos de prueba
6. 📚 Compartir aplicaciones reales

---

## 📚 Referencias y Recursos

### Libros Fundamentales

1. **Goldberg, D. E.** (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*
2. **Mitchell, M.** (1998). *An Introduction to Genetic Algorithms*
3. **Eiben, A. E., & Smith, J. E.** (2015). *Introduction to Evolutionary Computing*

### Papers Clave

- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*
- Deb, K., et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*
- Haupt, R. L., & Haupt, S. E. (2004). *Practical Genetic Algorithms*

### Recursos Online

- [Genetic Algorithms Tutorial - MIT OpenCourseWare](https://ocw.mit.edu)
- [Evolutionary Computation - Introduction](https://www.nature.com/subjects/evolutionary-computation)
- [DEAP Documentation](https://deap.readthedocs.io/)

---

## 📜 Licencia

Este material educativo está disponible para uso académico y de aprendizaje. Siéntete libre de usar, modificar y compartir con propósitos educativos.

---

## 📧 Contacto y Soporte

Para preguntas, sugerencias o problemas:

1. Revisa la documentación de cada tutorial
2. Consulta los ejemplos y tests
3. Abre un issue en el repositorio
4. Consulta las referencias bibliográficas

---

## 🎉 Comienza Ahora

```bash
# 1. Navega al Tutorial 1
cd GA_Tutorial_1_Basics

# 2. Abre el notebook
jupyter notebook GA_Introduction.ipynb

# 3. Sigue las instrucciones y completa los ejercicios

# 4. Ejecuta los tests
python public_tests.py
```

**¡Feliz aprendizaje! 🚀**

---

*Última actualización: 2025
Versión: 1.0
Estado: En desarrollo activo*
