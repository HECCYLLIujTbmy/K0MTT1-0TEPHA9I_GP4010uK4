# Task 1 - Создать программу, которая рисует отрезок между двумя точками, заданными пользователем 
```
import matplotlib.pyplot as plt
def common_grafic(x1,y1,x2,y2):
    plt.plot([x1, x2], [y1, y2], marker='o', linestyle='-', color='b')# Рисуем отрезок между точками синем цветом

    # Настраиваем отображение графика
    plt.xlim(min(x1, x2) - 1, max(x1, x2) + 1)
    plt.ylim(min(y1, y2) - 1, max(y1, y2) + 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Отрезок между двумя точками обычным алгоритмом')
    plt.grid(True)

    # Отображаем график
    plt.show()

def bresenham_line(x1, y1, x2, y2):
    points = []  # Список для хранения точек линии
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))  # Добавляем текущую точку в список
        if x1 == x2 and y1 == y2:  # Если достигли конечной точки, выходим
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points

# Разделяем точки на координаты X и Y для визуализации




x1, y1 = map(float, input("Введите координаты первой точки (x1, y1): ").split()) #ввод координат
x2, y2 = map(float, input("Введите координаты второй точки (x2, y2): ").split())
common_grafic(x1, y1, x2, y2)

line_points = bresenham_line(x1, y1, x2, y2)
x_coords, y_coords = zip(*line_points)
plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
plt.xlim(min(x_coords) - 1, max(x_coords) + 1)
plt.ylim(min(y_coords) - 1, max(y_coords) + 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Отрезок по алгоритму Брезенхема')
plt.grid(True)

# Отображаем график
plt.show()
```

> Алгоритм Брезенхема позволяет наглядно увидеть процесс построения прямой. Для примера, графики отрезков с координатами **(x1, y1): 100 200** и **(x2, y2): 700 50**
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/Figure_1.png)
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/Figure_2.png)

# 2 Создать программу, которая рисует окружность с заданным пользователем радиусом
```
import matplotlib.pyplot as plt
import math

# Обычное построение окружности
def common_circle(radius):
    theta = [i for i in range(361)]  # Углы в градусах (от 0 до 360)
    x_vals = [radius * math.cos(math.radians(t)) for t in theta]  # Вычисляем координаты x
    y_vals = [radius * math.sin(math.radians(t)) for t in theta]  # Вычисляем координаты y

    plt.plot(x_vals, y_vals, linestyle='-', color='b')  # Рисуем окружность
    plt.gca().set_aspect('equal', adjustable='box')  # Соотношение осей 1:1
    plt.title("Окружность обычным алгоритмом")
    plt.grid(True)
    plt.savefig('Krug_common'+str(radius)+'.png')
    plt.show()
    
# Алгоритм Брезенхема для окружности
def bresenham_circle(radius):
    points = []
    x = 0
    y = radius
    d = 3 - 2 * radius

    def plot_circle_points(x_center, y_center, x, y):
        points.extend([(x_center + x, y_center + y), (x_center - x, y_center + y),
                       (x_center + x, y_center - y), (x_center - x, y_center - y),
                       (x_center + y, y_center + x), (x_center - y, y_center + x),
                       (x_center + y, y_center - x), (x_center - y, y_center - x)])

    x_center, y_center = 0, 0  # Центр окружности в (0,0)
    plot_circle_points(x_center, y_center, x, y)

    while y >= x:
        x += 1
        if d > 0:
            y -= 1
            d = d + 4 * (x - y) + 10
        else:
            d = d + 4 * x + 6
        plot_circle_points(x_center, y_center, x, y)

    return points

# Ввод радиуса от пользователя
radius = int(input("Введите радиус окружности: "))

# Обычный алгоритм рисования окружности
common_circle(radius)

# Алгоритм Брезенхема для рисования окружности
circle_points = bresenham_circle(radius)

# Разделяем точки на координаты X и Y для визуализации
x_coords, y_coords = zip(*circle_points)

plt.scatter(x_coords, y_coords, color='r')  # Рисуем окружность по алгоритму Брезенхема
plt.gca().set_aspect('equal', adjustable='box')  # Соотношение осей 1:1
plt.title("Окружность по алгоритму Брезенхема")
plt.grid(True)
plt.savefig('Krug_6pezen'+str(radius)+'.png')
plt.show()
```

> Ситуация, что была с прямой, при постройке окружности не поменялась. При маленьком радиусе,  **R=5**, видны точки постройки фигуры.



![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/Krug_6pezen5.png)
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/Krug_common5.png)


>  При **R=300**, точность повышается



![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/Krug_6pezen300.png)
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/Krug_common300.png)

> Весьма занятно, построение окружности обычной функцией будет занимать 0.22 секунды при **R∈(10, 10000000)**. Брезенхемом потребовалось *10 минут* на обрабатку **R=10000000** 


# Реализация алгоритма алгоритма Сезерленда-Коэна
> Алгоритм  Сазерленда-Коэна используется для отсечения линий прямоугольным окном на плоскости. Идея алгоритма заключается в классификации концов отрезка относительно сторон окна и применении побитовой логики для определения видимых частей отрезка.
```
import matplotlib.pyplot as plt

def compute_code(x, y, x_min, x_max, y_min, y_max):
    code = 0b0000  # Изначально все биты равны 0

    if x < x_min:       # слева от окна
        code |= 0b0001  # Устанавливает первый бит в 1, если точка слева от окна
    elif x > x_max:     # справа от окна
        code |= 0b0010  # Устанавливает второй бит в 1, если точка справа от окна
    if y < y_min:       # ниже окна
        code |= 0b0100  # Устанавливает третий бит в 1, если точка ниже окна
    elif y > y_max:     # выше окна
        code |= 0b1000  # Устанавливает четвертый бит в 1, если точка выше окна

    return code


def cohen_sutherland_clip(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
    code1 = compute_code(x1, y1, x_min, x_max, y_min, y_max)
    code2 = compute_code(x2, y2, x_min, x_max, y_min, y_max)
    accept = False
    clipped_line = (x1, y1, x2, y2)

    while True:
        if code1 == 0 and code2 == 0:  # оба конца внутри окна
            accept = True
            clipped_line = (x1, y1, x2, y2)
            break
        elif (code1 & code2) != 0:     # оба конца вне окна и на одной стороне
            break
        else:
            x, y = 0, 0
            code_out = code1 if code1 != 0 else code2

            #### поиск пересечения с границей окна
            if code_out & 8:           # точка выше окна
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif code_out & 4:         # точка ниже окна
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif code_out & 2:         # точка справа от окна
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif code_out & 1:         # точка слева от окна
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min

            ##### обновляем точку и код
            if code_out == code1:
                x1, y1 = x, y
                code1 = compute_code(x1, y1, x_min, x_max, y_min, y_max)
            else:
                x2, y2 = x, y
                code2 = compute_code(x2, y2, x_min, x_max, y_min, y_max)

    if accept:
        return (x1, y1, x2, y2)
    else:
        return None

def visualize_multiple_lines(x_min, y_min, x_max, y_max, lines):
    fig, ax = plt.subplots()


    window = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='blue', facecolor='none')
    ax.add_patch(window)

 
    for i, (x1, y1, x2, y2) in enumerate(lines):
        ##### ориг отрезок
        ax.plot([x1, x2], [y1, y2], color='red', linestyle='--', label='Оригинальный отрезок' if i == 0 else "")


        clipped_line = cohen_sutherland_clip(x1, y1, x2, y2, x_min, y_min, x_max, y_max)
        if clipped_line:
            cx1, cy1, cx2, cy2 = clipped_line
            ax.plot([cx1, cx2], [cy1, cy2], color='green', label='Обрезанный отрезок' if i == 0 else "")

    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_min - 5, y_max + 5)
    ax.axhline(0, color='black',linewidth=0.5, ls='--')
    ax.axvline(0, color='black',linewidth=0.5, ls='--')
    ax.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.title('Алгоритм Сазерленда-Коэна для нескольких отрезков')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

x_min, y_min, x_max, y_max = 1, 1, 5, 5
lines = [
    (-5, 5, 15, 5),
    (-23, -1, 8, 12),
    (-5, 7, 7, -3),
    (6, 6, 6, -3)
]
visualize_multiple_lines(x_min, y_min, x_max, y_max, lines)

```
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/алгорит-%20Сазерленда-Коэна.png)

# Task  - Сравнение производительности алгоритма заполнения многоугольников с затравкой и метода из библиотеки pygame
> Прихожу к выводу, что создавать велосипед в таких вопросах не следует. Способы из библиотек весьма хорошо оптимизированы, в то время как созданный руками код будет иметь множество изьянов. Когда pygame позволяет весьма быстро заполнить большие фигуры на любом железе, стандартная реализация затравки на фоне выглядит как что то страшное по времени.
> **Условие общее: 5 углов и 100 радиус. Pygame 0.067, Common func 0.684**

```
#pygame
import pygame
import numpy as np
import sys
import time

def create_regular_polygon(num_sides, radius, center):
    """
    Создание правильного многоугольника по количеству углов и радиусу.
    """
    angle = 2 * np.pi / num_sides  # Угол между вершинами
    vertices = [(int(center[0] + radius * np.cos(i * angle)),
                 int(center[1] + radius * np.sin(i * angle))) for i in range(num_sides)]
    return vertices

def boundary_fill(screen, x, y, fill_color):
    width, height = screen.get_size()
    background_color = screen.get_at((x, y))
    
    if background_color != (255, 255, 255):  # Проверяем, что не закрашиваем уже закрашенный пиксель
        return
    
    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()
        
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            continue
        
        current_color = screen.get_at((cx, cy))
        
        if current_color == background_color:
            screen.set_at((cx, cy), fill_color)

            # Проверяем соседние пиксели
            stack.append((cx - 1, cy))
            stack.append((cx + 1, cy))
            stack.append((cx, cy - 1))
            stack.append((cx, cy + 1))

def main():
    pygame.init()
    width, height = 400, 400
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Заполнение многоугольника")

    #количествo углов и радиус
    num_sides = int(input("Введите количество углов многоугольника: "))
    radius = int(input("Введите радиус многоугольника: "))

    # Генерация вершин правильного многоугольника
    center = (width // 2, height // 2)

    # Начинаем отсчет времени
    start_time = time.time()
    vertices = create_regular_polygon(num_sides, radius, center)
    polygon_creation_time = time.time() - start_time
    ###print(f"Время создания многоугольника: {polygon_creation_time:.6f} секунд")

    # Рисуем многоугольник
    pygame.draw.polygon(screen, (0, 0, 0), vertices)  # Черный цвет для границы
    pygame.draw.polygon(screen, (255, 255, 255), vertices)  # Белый цвет для внутренней части

    # Заполняем многоугольник
    start_fill_time = time.time()
    fill_color = (139, 0, 0)  # Цвет заливки (цвет крови!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
    boundary_fill(screen, center[0], center[1], fill_color)
    fill_execution_time = time.time() - start_fill_time
    print(f"Время заполнения многоугольника: {fill_execution_time:.6f} секунд")

    # Основной цикл
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()

if __name__ == "__main__":
    main()
```
```
#Common
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
def create_regular_polygon(num_sides, radius):
    """
    Создание правильного многоугольника по количеству углов и радиусу.
    """
    angle = 2 * np.pi / num_sides  # Угол между вершинами
    vertices = [(radius * np.cos(i * angle) + 200, radius * np.sin(i * angle) + 200) for i in range(num_sides)]
    return vertices

def create_polygon_image(vertices, shape=(400, 400)):
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[0] / fig.dpi, shape[1] / fig.dpi)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.invert_yaxis()
    ax.axis('off')

    # Рисуем многоугольник
    polygon = Polygon(vertices, closed=True, edgecolor='black', facecolor='white')
    ax.add_patch(polygon)

    # Преобразуем в массив
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(shape[0], shape[1], 4)
    plt.close(fig)

    return image[:, :, :3].copy()

def is_background(color, threshold=68):
    # Считаем белыми пиксели с яркостью выше 68
    return np.mean(color) > threshold

def boundary_fill(image, x, y, fill_color):
    
    if not is_background(image[x, y]):
        return

    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()
        if is_background(image[cx, cy]):
            image[cx, cy] = fill_color

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and is_background(image[nx, ny]):
                    stack.append((nx, ny))

# Запрос количества углов и радиуса
num_sides = int(input("Введите количество углов многоугольника: "))
radius = int(input("Введите радиус многоугольника: "))

# Генерация вершин правильного многоугольника
vertices = create_regular_polygon(num_sides, radius)

# Создание изображения многоугольника
image_shape = (1680, 1920)  # Размер изображения
image = create_polygon_image(vertices, shape=image_shape)

fill_color = np.array([139, 0, 0], dtype=np.uint8)  # Цвет заливки (цвет крови)

# Убираем темные серые пиксели между границей и заливкой
gray_threshold = 100
image[np.all((image[:, :, 0] < gray_threshold) & 
             (image[:, :, 1] < gray_threshold) & 
             (image[:, :, 2] < gray_threshold), axis=-1)] = [255, 255, 255]

# Отображаем исходное изображение
plt.subplot(1, 2, 1)
plt.title("Исходное изображение")
plt.imshow(image)

# Применяем Boundary Fill с начальной точкой внутри многоугольника
start_fill_time = time.time()
boundary_fill(image, image_shape[0] // 2, image_shape[1] // 2, fill_color)
fill_execution_time = time.time() - start_fill_time
print(f"Время заполнения многоугольника: {fill_execution_time:.6f} секунд")
# Отображаем результат
plt.subplot(1, 2, 2)
plt.title("После Boundary Fill")
plt.imshow(image)
plt.show()
```
