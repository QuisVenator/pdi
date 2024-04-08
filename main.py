import cv2
import numpy as np
import os
import sys
import re
import matplotlib.pyplot as plt

from new_hist import new_his_eq
from metrics import calculate_ambe, calculate_psnr, calculate_entropy, calculate_contrast

DEBUG = False

def usage():
    """Muestra cómo usar el script."""
    print("El directorio de entrada es escáneado y todas las imágenes (en escala de grises) serán procesadas.")
    print("Las imágenes procesadas se guardarán en el directorio de salida, junto con un archivo CSV con las métricas calculadas.")
    print("Uso: python3 main.py <directorio_imagenes> <directoria_salida>")
    print("Ejemplo: python3 main.py images/ output/")

if __name__ == '__main__':
    # Asegurar que se pasaron los argumentos correctos
    if len(sys.argv) != 3:
        usage()
        exit(1)
    input_dir = sys.argv[1]
    if not os.path.exists(input_dir):
        print("Error: el directorio de entrada no existe.")
        exit(1)
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Error: el directorio de salida ya existe. Por favor, elige otro.")
        exit(1)
    
    # Crear un subdirectorio en salida para histogramas
    histogram_dir = os.path.join(output_dir, 'histograms')
    if not os.path.exists(histogram_dir):
        os.makedirs(histogram_dir)

    # Crear el archivo CSV de salida
    output_csv = os.path.join(output_dir, 'metrics.csv')
    f = open(output_csv, 'w')
    f.write(',histograma paper,,,,histograma tradicional,,,,CLAHE\n')
    f.write('imagen,AMBE,PSNR,entropia,contraste,AMBE,PSNR,entropia,contraste,AMBE,PSNR,entropia,contraste\n')
    
    images = os.listdir(input_dir)
    images = [image for image in images if image.endswith('.jpg')]
    # Iterar sobre las imágenes en el directorio de entrada
    for i in range(len(images)):
        image = images[i]
        print(f"Procesando imagen {image} ({i/len(images)*100:.0f}%)")

        # Verificar si se trata de una imagen jpg
        if not image.endswith('.jpg'):
            if DEBUG:
                print(f"La imagen {image} no es un archivo JPG. Se omitirá.")
            continue
        image_name = re.sub(r'\.jpg$', '', image)
        image_path = os.path.join(input_dir, image)
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Verificar si la imagen se ha cargado correctamente
        if image_gray is None:
            print(f"Error: la imagen {image} no se pudo cargar. Verifica la ruta del archivo.")
            continue

        new_hist_eq = new_his_eq(image_gray)
        standard_hist_eq = cv2.equalizeHist(image_gray)
        clahe_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_gray)

        # Calcular métricas
        metrics = []
        metrics.append(calculate_ambe(image_gray, new_hist_eq))
        metrics.append(calculate_psnr(image_gray, new_hist_eq))
        metrics.append(calculate_entropy(new_hist_eq))
        metrics.append(calculate_contrast(new_hist_eq))
        metrics.append(calculate_ambe(image_gray, standard_hist_eq))
        metrics.append(calculate_psnr(image_gray, standard_hist_eq))
        metrics.append(calculate_entropy(standard_hist_eq))
        metrics.append(calculate_contrast(standard_hist_eq))
        metrics.append(calculate_ambe(image_gray, clahe_eq))
        metrics.append(calculate_psnr(image_gray, clahe_eq))
        metrics.append(calculate_entropy(clahe_eq))
        metrics.append(calculate_contrast(clahe_eq))
        f.write(f"{image_name},{','.join(map(str, metrics))}\n")

        # Guardar las imágenes procesadas
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_original.png"), image_gray)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_new_hist_eq.png"), new_hist_eq)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_standard_hist_eq.png"), standard_hist_eq)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_clahe_eq.png"), clahe_eq)

        # Guardar también los histogramas con matplotlib
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
        plt.bar(np.arange(256), hist.ravel())
        plt.savefig(os.path.join(histogram_dir, f"{image_name}_hist.png"))
        plt.close()
        hist = cv2.calcHist([new_hist_eq], [0], None, [256], [0, 256])
        plt.bar(np.arange(256), hist.ravel())
        plt.savefig(os.path.join(histogram_dir, f"{image_name}_new_hist_eq_hist.png"))
        plt.close()
        hist = cv2.calcHist([standard_hist_eq], [0], None, [256], [0, 256])
        plt.bar(np.arange(256), hist.ravel())
        plt.savefig(os.path.join(histogram_dir, f"{image_name}_standard_hist_eq_hist.png"))
        plt.close()
        hist = cv2.calcHist([clahe_eq], [0], None, [256], [0, 256])
        plt.bar(np.arange(256), hist.ravel())
        plt.savefig(os.path.join(histogram_dir, f"{image_name}_clahe_eq_hist.png"))
        plt.close()
    f.close()