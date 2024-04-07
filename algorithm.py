
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np



def get_separate_point(hist, total_pixels):
    gray_level_maximum_range = hist.shape[0]
    total_sum = sum([ hist[i][0] * i for i in range(gray_level_maximum_range)])
    return math.floor(total_sum / total_pixels)

def get_min_gray_level(hist):
    for i in range(hist.shape[0]):
        if hist[i][0] != 0:
            return i

    return -1

def get_max_gray_level(hist):
    for i in range(hist.shape[0] - 1, 0, -1):
        if hist[i][0] != 0:
            return i

    return -1

def quantify_hist(hist, pl_1, pl_2, pl_3, start, end):
    quantified_hist = np.zeros_like(hist)
    for i in range(start, end):
        count = hist[i][0]
        if count <= pl_1:
            quantified_hist[i][0] = pl_1
        elif pl_1 < count and count <= pl_3:
            quantified_hist[i][0] = pl_2
        else:
            quantified_hist[i][0] = pl_3
    return quantified_hist





image_gray = cv2.imread('40.png', cv2.IMREAD_GRAYSCALE)

gif_kiss = cv2.VideoCapture('kiss.gif')

# Verificar si la imagen se ha cargado correctamente
if image_gray is None:
    print("Error: la imagen no se pudo cargar. Verifica la ruta del archivo.")
    exit(1)

# Obtengo el histograma de la imagen
hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

# Obtenemos el nivel de gris mínimo y máximo del histograma
min_gray_level = get_min_gray_level(hist)
max_gray_level = get_max_gray_level(hist)
print("MIN GRAY LEVEL", min_gray_level)
print("MAX GRAY LEVEL", max_gray_level)

# Obtenemos la cantidad total de pixeles de la imagen
total_pixels = image_gray.shape[0] * image_gray.shape[1]

sp = get_separate_point(hist, total_pixels)

print("SP", sp)

# Subdivimos el histograma original en dos, uno donde niveles de grises son menores a sp
# y otro donde los niveles de grises son mayores a sp
low_sub_hist = hist[:sp + 1]
low_sub_hist_total_pixels = np.sum(low_sub_hist)
low_sub_hist_sp = get_separate_point(low_sub_hist, low_sub_hist_total_pixels)

high_sub_hist = hist[sp + 1:]
high_sub_hist_total_pixels = np.sum(high_sub_hist)
# TODO: Definir si hay que sumar sp o sp + 1 a 
high_sub_shift = sp + 1
high_sub_hist_sp = get_separate_point(high_sub_hist, high_sub_hist_total_pixels) + high_sub_shift

print(high_sub_hist.shape)
print(low_sub_hist.shape)


print("SPL", low_sub_hist_sp)
print("SPH", high_sub_hist_sp)

# Calculamos los ratios de niveles de gris (gr) del subhistograma inferior y superior
gr_low_2 = (sp - low_sub_hist_sp) / (sp - min_gray_level)
gr_high_2 = (max_gray_level - high_sub_hist_sp) / (max_gray_level - sp)
print("GRL2:", gr_low_2)
print("GRH2:",gr_high_2)

# Calculamos la diferencia de niveles de gris (d) para el subhistograma inferior y superior
d_low = (1 - gr_low_2) / 2 if gr_low_2 > 0.5 else gr_low_2 / 2
d_high = (1 - gr_high_2) / 2 if gr_high_2 > 0.5 else gr_high_2 / 2

print("DL", d_low)
print("DH", d_high)

# Calculamos todos los ratios de niveles de gris del subhistograma inferior
gr_low_1 = gr_low_2 - d_low
gr_low_3 = gr_low_2 + d_low
print("GRL1", gr_low_1)
print("GRL3", gr_low_3)

# Calculamos todos los ratios de niveles de gris del subhistograma superior
gr_high_1 = gr_high_2 - d_high
gr_high_3 = gr_high_2 + d_high
print("GRH1", gr_high_1)
print("GRH3", gr_high_3)

# Obtenemos los picos de cantidad de pixeles de cada subhistograma
pk_low = low_sub_hist.max()
pk_high = high_sub_hist.max()

print("PKLOW", low_sub_hist.max())
print("PKHigh", high_sub_hist.max())

# Calculamos los límites de plateau (pl) de cada subhistrograma
pl_low_1 = gr_low_1 * pk_low
pl_low_2 = gr_low_2 * pk_low
pl_low_3 = gr_low_3 * pk_low

print("PL1", pl_low_1)
print("PL2", pl_low_2)
print("PL3", pl_low_3)

pl_high_1 = gr_high_1 * pk_high
pl_high_2 = gr_high_2 * pk_high
pl_high_3 = gr_high_3 * pk_high

print("PH1", pl_high_1)
print("PH2", pl_high_2)
print("PH3", pl_high_3)

# Cuantificamos los subhistogramas a partir de los limites de plateau obtenidos
low_sub_hist_quantified = quantify_hist(low_sub_hist, pl_low_1, pl_low_2, pl_low_3, min_gray_level, sp)
high_sub_hist_quantified = quantify_hist(high_sub_hist, pl_high_1, pl_high_2, pl_high_3, 0, max_gray_level - high_sub_shift)

low_sub_hist_quantified_total_pixels = low_sub_hist_quantified.sum()
high_sub_hist_quantified_total_pixels = high_sub_hist_quantified.sum()

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
plt.title('Histograma Original')

plt.plot(low_sub_hist_quantified)

x = [i + high_sub_shift for i in range(len(high_sub_hist_quantified))]

plt.plot(x, high_sub_hist_quantified)


plt.show()

# Procedemos a calcular la imagen ecualizada

# Acá no tengo claro si esta función está bien, ya que en el paper en ningún momento se utiliza la variable k
cdf_low = lambda k: sum([ low_sub_hist_quantified[i][0] / low_sub_hist_quantified_total_pixels  for i in range(k + 1)])
cdf_high = lambda k : sum([ high_sub_hist_quantified[i][0] / high_sub_hist_quantified_total_pixels  for i in range(k + 1 - high_sub_shift)])


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


cv2.imwrite("40_gray_transformed.png", transformed_image)
cv2.imshow("Imagen original", image_gray)
cv2.imshow('Imagen ecualizada', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()






