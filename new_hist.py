
import cv2
import matplotlib.pyplot as plt
import numpy as np

DEBUG = False

def get_separate_point(hist, total_pixels):
    """Obtiene el punto de separación de un histograma.

    Args:
        hist (np array): Histograma de la imagen.
        total_pixels (int): Número total de pixeles de la imagen.

    Returns:
        int: Punto de separación del histograma.
    """
    gray_level_maximum_range = hist.shape[0]
    total_sum = sum([ hist[i][0] * i for i in range(gray_level_maximum_range)])
    
    if DEBUG:
        # Ploteamos el histograma con el punto de separación
        plt.plot(hist)
        plt.axvline(x=total_sum / total_pixels, color='r', linestyle='--')
        plt.title('Histograma')
        plt.show()

    return round(total_sum / total_pixels)

def get_min_gray_level(hist):
    """Retorna el nivel de gris mínimo del histograma.

    Args:
        hist (np array): Histograma de la imagen.

    Returns:
        int: Nivel de gris mínimo.
    """
    return np.nonzero(hist)[0][0]

def get_max_gray_level(hist):
    """Retorna el nivel de gris máximo del histograma.

    Args:
        hist (np array): Histograma de la imagen.

    Returns:
        int: Nivel de gris máximo.
    """
    return np.nonzero(hist)[0][-1]

def quantify_hist(hist, pl_1, pl_2, pl_3, start=0, end=256):
    """Cuantifica un histograma a partir de los límites de plateau.

    Args:
        hist (np array): Histograma de la imagen.
        pl_1 (float): Límite de plateau 1.
        pl_2 (float): Límite de plateau 2.
        pl_3 (float): Límite de plateau 3.
        start (int, optional): Punto de inicio para la cuantificación (incluyente). Por defecto 0.
        end (int, optional): Punta final para la cuantificación (excluyente). Por defecto 256.

    Returns:
        np array: El histograma cuantificado.
    """
    quantified_hist = np.zeros_like(hist)
    for i in range(start, min(end, len(hist))):
        count = hist[i][0]
        if count <= pl_1:
            quantified_hist[i][0] = pl_1
        elif pl_1 < count and count <= pl_3:
            quantified_hist[i][0] = pl_2
        else:
            quantified_hist[i][0] = pl_3
    return quantified_hist


image_gray = cv2.imread('40.png', cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen se ha cargado correctamente
if image_gray is None:
    print("Error: la imagen no se pudo cargar. Verifica la ruta del archivo.")
    exit(1)

def new_his_eq(image_gray):
    """Aplica la ecualización nueva de histograma a una imagen en escala de grises.

    Args:
        image_gray (np array): Imagen en escala de grises.

    Returns:
        np array: Imagen ecualizada.
    """
    # Obtengo el histograma de la imagen
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256]).astype(int)

    # Obtenemos el nivel de gris mínimo y máximo del histograma
    min_gray_level = get_min_gray_level(hist)
    max_gray_level = get_max_gray_level(hist)
    if DEBUG:
        print("MIN GRAY LEVEL", min_gray_level)
        print("MAX GRAY LEVEL", max_gray_level)

    # Obtenemos la cantidad total de pixeles de la imagen
    total_pixels = image_gray.shape[0] * image_gray.shape[1]

    sp = get_separate_point(hist, total_pixels)

    # Subdivimos el histograma original en dos, uno donde niveles de grises son menores a sp
    # y otro donde los niveles de grises son mayores a sp
    low_sub_hist = hist[:sp + 1]
    low_sub_hist_total_pixels = np.sum(low_sub_hist)
    low_sub_hist_sp = get_separate_point(low_sub_hist, low_sub_hist_total_pixels)
    if DEBUG:
        print("LOW SUB HIST SP", low_sub_hist_sp)

    high_sub_hist = hist[sp + 1:]
    high_sub_hist_total_pixels = np.sum(high_sub_hist)
    high_sub_shift = sp + 1
    high_sub_hist_sp = get_separate_point(high_sub_hist, high_sub_hist_total_pixels) + high_sub_shift
    if DEBUG:
        print("HIGH SUB HIST SP", high_sub_hist_sp)


    # Calculamos los ratios de niveles de gris (gr) del subhistograma inferior y superior
    gr_low_2 = (sp - low_sub_hist_sp) / (sp - min_gray_level)
    gr_high_2 = (max_gray_level - high_sub_hist_sp) / (max_gray_level - sp)
    if DEBUG:
        print("GRL2:", gr_low_2)
        print("GRH2:",gr_high_2)

    # Calculamos la diferencia de niveles de gris (d) para el subhistograma inferior y superior
    d_low = (1 - gr_low_2) / 2 if gr_low_2 > 0.5 else gr_low_2 / 2
    d_high = (1 - gr_high_2) / 2 if gr_high_2 > 0.5 else gr_high_2 / 2
    if DEBUG:
        print("DL", d_low)
        print("DH", d_high)

    # Calculamos todos los ratios de niveles de gris del subhistograma inferior
    gr_low_1 = gr_low_2 - d_low
    gr_low_3 = gr_low_2 + d_low
    if DEBUG:
        print("GRL1", gr_low_1)
        print("GRL3", gr_low_3)

    # Calculamos todos los ratios de niveles de gris del subhistograma superior
    gr_high_1 = gr_high_2 - d_high
    gr_high_3 = gr_high_2 + d_high
    if DEBUG:
        print("GRH1", gr_high_1)
        print("GRH3", gr_high_3)

    # Obtenemos los picos de cantidad de pixeles de cada subhistograma
    pk_low = low_sub_hist.max()
    pk_high = high_sub_hist.max()
    if DEBUG:
        print("PKLOW", low_sub_hist.max())
        print("PKHigh", high_sub_hist.max())

    # Calculamos los límites de plateau (pl) de cada subhistrograma
    pl_low_1 = gr_low_1 * pk_low
    pl_low_2 = gr_low_2 * pk_low
    pl_low_3 = gr_low_3 * pk_low
    if DEBUG:
        print("PL1", pl_low_1)
        print("PL2", pl_low_2)
        print("PL3", pl_low_3)

    pl_high_1 = gr_high_1 * pk_high
    pl_high_2 = gr_high_2 * pk_high
    pl_high_3 = gr_high_3 * pk_high
    if DEBUG:
        print("PH1", pl_high_1)
        print("PH2", pl_high_2)
        print("PH3", pl_high_3)

    # Cuantificamos los subhistogramas a partir de los limites de plateau obtenidos
    low_sub_hist_quantified = quantify_hist(low_sub_hist, pl_low_1, pl_low_2, pl_low_3, start=min_gray_level)
    high_sub_hist_quantified = quantify_hist(high_sub_hist, pl_high_1, pl_high_2, pl_high_3, end=max_gray_level - sp)

    low_sub_hist_quantified_total_pixels = low_sub_hist_quantified[min_gray_level:].sum()
    high_sub_hist_quantified_total_pixels = high_sub_hist_quantified[:max_gray_level+1].sum()

    #creo que debe ser high_sub_hist_quantified_total_pixels = high_sub_hist_quantified[:max_gray_level - sp].sum()

    if DEBUG:
        # Mosramos el histograma original, los límites de plateau y el histograma cuantificado de cada subhistograma
        plt.plot(hist)
        sp_percentage = sp/plt.xlim()[1] + 0.03
        plt.axvline(x=sp, color='r', linestyle='--')

        plt.text(low_sub_hist_sp, 0, "SPL")
        plt.axhline(y=pl_low_1, color='b', linestyle='--', xmax=sp_percentage)
        plt.text(0, pl_low_1 + 0.1, "Pl1")
        plt.axhline(y=pl_low_2, color='b', linestyle='--', xmax=sp_percentage)
        plt.text(0, pl_low_2 + 0.1, "PL2")
        plt.axhline(y=pl_low_3, color='b', linestyle='--', xmax=sp_percentage)
        plt.text(0, pl_low_3 + 0.1, "PL3")


        plt.text(high_sub_hist_sp, 0, "SPH")

        plt.axhline(y=pl_high_1, color='b', linestyle='--', xmin=sp_percentage)
        plt.text(250, pl_high_1 + 0.1, "PH1")
        plt.axhline(y=pl_high_2, color='b', linestyle='--', xmin=sp_percentage)
        plt.text(250, pl_high_2 + 0.1, "PH2")
        plt.axhline(y=pl_high_3, color='b', linestyle='--', xmin=sp_percentage)
        plt.text(250, pl_high_3 + 0.1, "PH3")
        plt.title('Histograma Original y Cuantificado')
        plt.plot(low_sub_hist_quantified)
        x = [i + high_sub_shift for i in range(len(high_sub_hist_quantified))]
        plt.plot(x, high_sub_hist_quantified)
        plt.show()

    # Procedemos a calcular la imagen ecualizada
    cdf_low = lambda k: sum([ low_sub_hist_quantified[i][0] / low_sub_hist_quantified_total_pixels  for i in range(min_gray_level, k + 1)])
    cdf_high = lambda k : sum([ high_sub_hist_quantified[i][0] / high_sub_hist_quantified_total_pixels  for i in range(0, k - sp)])


    # Calculamos la transformación de los niveles de grises aplicar a la imagen
    trans_map = {}
    for k in range(min_gray_level, max_gray_level + 1):
        if(min_gray_level <= k <= sp):
            trans_map[k] = (sp * cdf_low(k))
        else:
            trans_map[k] = (sp + 1 + (255 - sp - 1) * cdf_high(k))

    # Transformamos los pixeles de la imagen
    transformed_image = image_gray.copy()
    for i in range(transformed_image.shape[0]):
        for j in range(transformed_image.shape[1]):
            new_gray_level = trans_map.get(image_gray[i][j])
            transformed_image[i][j] = new_gray_level

    if DEBUG:
        # Mostramos el histograma de la imagen ecualizada
        histogram_transformed = cv2.calcHist([transformed_image], [0], None, [256], [0, 256]).astype(int)
        plt.bar([i for i in range(256)], histogram_transformed.ravel())
        plt.title('Histograma Ecualizado')
        plt.show()

        cv2.imshow("Imagen original", image_gray)
        cv2.imshow('Imagen ecualizada', transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return transformed_image
