# Como usar
## Convertir varias imagenes
Se puede usar la aplicación con el siguiente comando:
```bash
python3 main.py <nombre carpeta imagenes> <nombre directorio salida>
```

En este modo se convertirán todas las imágenes como descrito en el informe y se guardarán en el directorio de salida, junto con un csv de las métricas relevantes.

## Usar solo método propuesto
Para usar solo el nuevo método analizado, se puede usar la función `new_his_eq(image)` del archivo `new_hist.py`. Esta función recibe una imagen y devuelve la imagen ecualizada con el nuevo método. También se puede cambiar `DEBUG` a `True` para ver los histogramas de las imágenes, con su los límites de plateaus, histogramas y punto de separación.