Final Other versions
SIN data augmentation
Final Indiv DataAug inicial: --> 
Final Indiv DataAug v1: -->  Usa las imágenes ya cargadas en arrays y luego Genera las imágenes (.flow) aumentadas en memoria           MEJOR ACCURACY (TEST) 82,609%   LOSS = 0,4917. Es la base del main.ipynb definitivo

Final Indiv DataAug v2: --> Usa las imágenes desde los directorios (train, valid y test) y luego 
                    genera las imágenes (.flowfromdirectory) aumentadas en memoria
                    Convierte a ByN y redimensiona en memoria las imágenes
                    Lo malo de esta opcion es que hay que dividir explicita y previamente los conjuntos
                    de train, valid y  test en un directorio


NOTA: Debido que para los resultados y pruebas preliminares se obtuvo mejor accuracy con las imágenes en grayscale que en color, se usará esta modalidad. De igual manera con el tamano de las imágenes dado mejor resultados con redimensionando las mismas a 48 x48.


El seleccionado para seguir trabajándolo  fué el que esta en la carpeta Final Indiv DataAug v1


Cuando se usa generación aumentada de imágenes en cada EPOCH se genera una imagen diferente con lo cual si usamos un 
conjunto de 210 imagenes de entrenamiento, en 10 epochs habremos entrenado el modelo con 2100 imágenes distintas (variaciones de las originales segun lo especificado )
