# bootstrap-backtester

## Contexto

Este repositorio busca generar las funciones necesarias para realizar el backtesting de estrategias de inversión con un enfoque alternativo al convencional. Al igual que el enfoque clásico, este enfoque busca comparar el caso de haber utilizado la estrategia versus la opción de no haberlo hecho. Sin embargo, a diferencia del enfoque convencional que compara la estrategia con un benchmark global, este nuevo enfoque realiza una selección aleatoria de activos y períodos y busca identificar si la estrategia es sólida en distintos contextos para resultar sobreperformante.

## El sesgo de selección y el nuevo enfoque

El objetivo del nuevo enfoque es atenuar el impacto del sesgo de selección, que puede generar conclusiones equivocadas en el backtesting. El sesgo de selección es la realización de un backtesting que no hubiera sido posible en tiempo real, en particular por los activos de los que se dispone. El ejemplo más convencional es el del SP500, que regularmente tiene 500 activos pero que es incorrecto backtestear los activos que hoy forman parte del SP500, ya que algunos hoy no están y en el pasado sí (y algunos en el pasado estaban y hoy no).

El nuevo enfoque quita importancia a los activos en cuestión, y pone como eje principal que la estrategia será satisfactoria en caso de que en diferentes contextos, logre superar a un benchmark que no usa la estrategia. Apoyado sobre esta idea, la estrategia toma N sets de M activos aleatorios, y en un período de tiempo también aleatorio evalúa la estrategia. El benchmark con el que compara la estrategia es, por cada N set, el promedio de esos M activos en ese mismo período de tiempo. Esto permite salir de la comparación con un benchmark fijo y concentrarse en el efecto timing de la estrategia. Los resultados los devuelve en forma de boxplot, comparando específicamente los resultados que da en las distintas iteraciones de la estrategia y del benchmark.

## Componentes

Los componentes de este repositorio son dos: 

- bootstrap_backtester.py: El conjunto de funciones necesarias para evaluar las estrategias.
- example.ipynb: Una jupyter notebook de ejemplo, con los activos correspondientes al SP500 y la evaluación de la estrategia del RSI, comprando por debajo de 30 en el índice y vendiendo sobre los 70.
