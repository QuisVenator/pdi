import numpy as np
from skimage.measure import shannon_entropy

def calculate_ambe(image1, image2):
    """
    Calcula el Error Medio Absoluto del Brillo (AMBE) entre dos imágenes.
    """
    return abs(np.mean(image1) - np.mean(image2))

def calculate_psnr(image1, image2):
    """
    Calcula la Relación Señal a Ruido de Pico (PSNR) entre dos imágenes.
    """
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_entropy(image):
    """
    Calcula la entropía de Shannon de una imagen.
    """
    return shannon_entropy(image)

def calculate_contrast(image):
    """
    Calcula el contraste de una imagen como su desviación estándar.
    """
    return np.std(image)