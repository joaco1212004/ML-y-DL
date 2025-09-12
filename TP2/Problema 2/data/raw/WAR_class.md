## **Dataset: Predicción de Rendimiento de Jugadores de Basketball**

Este conjunto de datos reúne información estadística de jugadores de basketball recopilada a lo largo de varias temporadas.  
El objetivo es predecir el impacto de un jugador en el rendimiento de su equipo a partir de métricas individuales de juego, clasificándolo en una de las tres categorías de **WAR\_class** (*Wins Above Replacement*):  
- **Negative WAR** → impacto negativo.  
- **Null WAR** → impacto neutro.  
- **Positive WAR** → impacto positivo.  

---

## **Descripción de las Variables**

### **Variables Numéricas (estadísticas de juego por temporada)**

- **poss** *(entero)* → Número total de posesiones jugadas en la temporada.  
- **mp** *(minutos)* → Minutos jugados durante la temporada.  
- **raptor_total** *(puntos/100 posesiones)* → Puntos por encima del promedio aportados por el jugador cada 100 posesiones, considerando tanto ofensiva como defensa. Incluye componentes de caja estadística (*box score*) y de diferencia con el jugador en cancha vs. fuera de cancha (*on-off*).  
- **pace_impact** *(impacto por 48 min)* → Medida del impacto del jugador en el número de posesiones de su equipo por cada 48 minutos de juego.  

### **Variable Objetivo**

- **war_class** *(categórica)* → Clasificación del jugador según su contribución al equipo.  
  - `1` → Negative WAR  
  - `2` → Null WAR  
  - `3` → Positive WAR  