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
