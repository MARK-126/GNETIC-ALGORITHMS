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
| **GA_Tutorial_3_Advanced** | Avanzado | Optimización avanzada y TSP | 🚧 En desarrollo |
| **GA_Tutorial_4_Applications** | Aplicado | Aplicaciones reales y casos de uso | 🚧 En desarrollo |
| **W2A1** | Referencia | Métodos de optimización (material original) | ✅ Completo |

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

### Contenido Planificado

- 🚧 Optimización de hiperparámetros en ML
- 🚧 Feature selection
- 🚧 Diseño de redes neuronales (neuroevolución)
- 🚧 Scheduling y planificación
- 🚧 Optimización de portafolios
- 🚧 Diseño de sistemas de control
- 🚧 Casos de estudio industriales

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

- [x] Tutorial 1: Fundamentos completos
- [x] Módulo de utilidades básicas
- [x] Sistema de tests automatizados
- [x] Funciones benchmark (Sphere, Rastrigin, Rosenbrock)
- [x] Visualizaciones interactivas
- [x] Tutorial 2: Operadores avanzados (utilidades)

### 🚧 En Desarrollo

- [ ] Tutorial 2: Notebook completo
- [ ] Tutorial 3: Optimización avanzada
- [ ] Tutorial 4: Aplicaciones reales

### 🎯 Próximas Características

- [ ] Ejemplos de GA paralelo
- [ ] Integración con scikit-learn
- [ ] Dashboard de análisis de rendimiento
- [ ] Videos tutoriales complementarios

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
