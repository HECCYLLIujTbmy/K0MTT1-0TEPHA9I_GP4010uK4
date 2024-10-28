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
