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

# Циферблат
```

import matplotlib.pyplot as plt
import numpy as np

class CircleDrawer:
    

    
    def __init__(self, radius): 
        self.radius = radius
        self.points = []


    def symmetry(self, x, y):  # Добавляет симметричные точки
            self.points += [
                (x, y), (-x, y), (x, -y), (-x, -y),
                (y, x), (-y, x), (y, -x), (-y, -x)
            ]

    def raschet(self):  # Рассчитывает координаты точек для построения круга
        x, y = 0, self.radius
        d = 3 - 2 * self.radius
        while x <= y:
            self.symmetry(x, y)
            if d <= 0:
                d += 3 * x + 7
            else:
                d += 4 * (x - y) + 10
                y -= 1
            x += 1
        self.points = self.sortirovka(set(self.points))

    

    def sortirovka(self, points):  # Сортирует точки
        points = sorted(points, key=lambda p: np.arctan2(p[1], p[0]))
        return list(points) + [list(points)[0]]

    def c4ferblat(self):  # Строит круг с 12 линиями
        x_coords, y_coords = zip(*self.points)
        plt.plot(x_coords, y_coords, color='black')
        
        # Добавляем 12 линий
        for i in range(12):
            angle = 2 * np.pi * i / 12  # Угол для каждой линии
            start_x = (self.radius * 0.8) * np.cos(angle)  # настройка линий
            start_y = (self.radius * 0.8) * np.sin(angle)
            end_x = self.radius * np.cos(angle)  # Конец линии на окружности
            end_y = self.radius * np.sin(angle)
            
            plt.plot([start_x, end_x], [start_y, end_y], color='red', lw=1.5)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'радиус {self.radius} и 12 линий')
        plt.grid(True)
        plt.show()

def main():

        radius = int(input("Введите радиус: "))
        if radius <= 0:
            print("Радиус натуральное число.")
            return

        drawer = CircleDrawer(radius)
        drawer.raschet()
        drawer.c4ferblat()

if __name__ == "__main__":
    main()
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/cyfer.png)
```
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
# Task Алгоритм Цирруса-Бека 

> Алгоритм Кируса-Бека — это эффективный алгоритм отсечения отрезков прямой произвольным выпуклым многоугольником. Он основан на использовании параметрического представления отрезка и анализа взаимного расположения концов отрезка относительно граней многоугольника.
```
from PIL import Image
from random import randint
from Bresenham import draw_line, draw

class point2:
	def __init__(self, x, y):
		self.x = x
		self.y = y

class polygon:
	def __init__(self, vertexes):
		self.vertexes = vertexes

	def cyruse_beck(self, a, b):
		normal = lambda p1, p2: point2(- (p1.y - p2.y), - (p2.x - p1.x))
		direction = lambda p1, p2: point2(p2.x - p1.x, p2.y - p1.y)
		scalar = lambda p1, p2: p1.x * p2.x + p1.y * p2.y
		
		t_begin = 0
		t_end = 1
		ab_vec = direction(a, b)

		new_a = point2(a.x, a.y)
		new_b = point2(b.x, b.y)
		for i in range(len(vertexes)):
			p1 = vertexes[i]
			p2 = vertexes[(i + 1) % len(vertexes)]
			p12_vec = direction(p1, p2)
			p12_norm = normal(p1, p2)

			p1a = direction(p1, a)

			scalar_abn = scalar(ab_vec, p12_norm)
			scalar_p1an = scalar(p1a, p12_norm)

			if scalar_abn == 0:
				if scalar_p1an > 0:
					return None, None
				else:
					return new_a, new_b
			elif scalar_abn > 0:
				t =  -  scalar_p1an / scalar_abn
				if t > t_end:
					continue
				t_begin = max(t_begin, t)
			elif scalar_abn < 0:
				t =   - scalar_p1an / scalar_abn
				if t < t_begin:
					continue
				t_end = min(t_end, t)

		if t_end > t_begin:
			if t_begin > 0:
				new_a.x = int(a.x + t_begin * ab_vec.x)
				new_a.y = int(a.y + t_begin * ab_vec.y)
			if t_end < 1:
				new_b.x = int(a.x + t_end * ab_vec.x)
				new_b.y = int(a.y + t_end * ab_vec.y)
		else:
			return None, None

		return new_a, new_b

if __name__ == '__main__':
	vertexes = [
		point2(250, 250),
		point2(250, 350),
	 	point2(350, 450),
	 	point2(450, 450),
	 	point2(550, 350),
	 	point2(550, 250),
	 	point2(450, 150),
	 	point2(350, 150)
	]

	
	center = point2(400, 300)
	segment = polygon(vertexes)
	pol_points = []

	with Image.open('NUR.png') as im:
		im.paste((0, 0, 0), (0, 0, im.size[0], im.size[1]))

		for i in range(len(vertexes)):
			pol_points += draw_line(vertexes[i].x, vertexes[i].y,
									vertexes[(i + 1) % len(vertexes)].x,
									vertexes[(i + 1) % len(vertexes)].y)
		draw(im, pol_points, (0, 255, 0))

		for i in range(500):
			b = point2(randint(0, 799), randint(0, 599))
			a, b = segment.cyruse_beck(center, b)
			if a is not None and b is not None:
				line_points = draw_line(a.x, a.y, b.x, b.y)
				draw(im, line_points, (255, 0, 0))

		im.save('CyrusBeck.png')
```
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/CyrusBeck.png)

# Алгоритм заполнения фигуры

> Этот алгоритм основан на методе затравочного заполнения (Flood Fill), который используется для заполнения замкнутых областей в двумерных массивах (например, изображениях). Данный код позволил мне заполнить фигуру (какой-то осколок) краснмым цветом относительно точно). Хотя другие реализации данного алгоритма порой не могли увенчаться успехом

```
import matplotlib.pyplot as plt
import numpy as np

def fill_polygon(img, seed):
    stack = [seed]

    while stack:
        x, y = stack.pop()

        if img[x, y] == 0:
            img[x, y] = 1
            neighbors = [(x-1, y), (x, y-1), (x+1, y), (x, y+1)]

            for nx, ny in neighbors:
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                    stack.append((nx, ny))

image_size = 10
img = np.zeros((image_size, image_size))


polygon_points = [
    (3, 2),  # Нижний левый угол
    (4, 1),  # Верхний острый угол
    (5, 2),  # Нижний правый угол
    (5, 3),  # Верхний правый угол
    (8, 10),
    (2, 2)
]
polygon = np.array(polygon_points)
plt.fill(polygon[:, 0], polygon[:, 1], color='red')
seed_point = (4, 4)
fill_polygon(img, seed_point)

plt.imshow(img, cmap='gray', origin='lower')
plt.show()
```
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/fill.png)
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/FIGASE.png)

# Заполнение сканированием
```
import numpy as np
import matplotlib.pyplot as plt

def scanline_fill(img, seed):
    """
    Заполнение замкнутой области посредством горизонтального сканирования.

    :param img: Двумерный массив (изображение)
    :param seed: Начальная точка (x, y)
    """
    stack = [seed]
    rows, cols = img.shape

    while stack:
        x, y = stack.pop()

        # Перейти влево, пока не найдем границу
        left = y
        while left >= 0 and img[x, left] == 0:
            left -= 1
        left += 1

        # Перейти вправо, пока не найдем границу
        right = y
        while right < cols and img[x, right] == 0:
            right += 1
        right -= 1

        # Закрасить строку от left до right
        for col in range(left, right + 1):
            img[x, col] = 1

        # Проверить строки выше и ниже
        for col in range(left, right + 1):
            if x > 0 and img[x - 1, col] == 0:  # Строка сверху
                stack.append((x - 1, col))
            if x < rows - 1 and img[x + 1, col] == 0:  # Строка снизу
                stack.append((x + 1, col))

# Размер изображения
image_size = 20
img = np.zeros((image_size, image_size))

# Рисуем замкнутую область
img[5:15, 5] = 1
img[5:15, 15] = 1
img[5, 5:20] = 1
img[15, 5:16] = 1

# Точка внутри замкнутой области
seed_point = (10, 10)

# Заполняем область
scanline_fill(img, seed_point)

# Визуализация результата
plt.imshow(img, cmap='gray', origin='lower')
plt.title("Scanline Fill")
plt.show()

```

![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/scanfile.png)




# Вращение фигуры
```
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

# Параметры GIF
gif_filename = 'rotating_spiral_240fps.gif'
frames = []
num_frames = 240  # Количество кадров

# Функция для поворота точки в 3D
def rotate(point, angle_x, angle_y, angle_z):
    ax, ay, az = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx @ point

# Создание спирали
num_points = 1000
z = np.linspace(-2, 2, num_points)
radius = 0.5 * (1 + z)  # Радиус спирали изменяется вдоль оси Z
x = radius * np.cos(10 * z)  # Угловая частота вращения
y = radius * np.sin(10 * z)
points = np.vstack((x, y, z)).T

# Цвета для спирали
colors = plt.cm.viridis(np.linspace(0, 1, num_points))

# Создание фигуры
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Генерация анимации вращения
for i in range(num_frames):
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.axis('off')

    # Углы поворота
    angle_x = i * 1.5  # Скорость вращения по X
    angle_y = i * 0.75  # Скорость вращения по Y
    angle_z = i * 0.5  # Скорость вращения по Z

    # Поворачиваем точки
    rotated_points = np.array([rotate(p, angle_x, angle_y, angle_z) for p in points])

    # Отрисовка спирали
    ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2],
               c=colors, s=1, alpha=0.8)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)

# Сохранение в GIF с 240 fps
imageio.mimsave(gif_filename, frames, fps=240)
print(f'GIF сохранен в файл: {gif_filename}')

```
![alt text](https://github.com/HECCYLLIujTbmy/K0MTT1-0TEPHA9I_GP4010uK4/blob/main/rotating_spiral_240fps.gif)
