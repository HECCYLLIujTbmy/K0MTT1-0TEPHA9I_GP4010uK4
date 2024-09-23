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

