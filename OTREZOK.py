import matplotlib.pyplot as plt


x1, y1 = map(float, input("Введите координаты первой точки (x1, y1): ").split()) #ввод координат
x2, y2 = map(float, input("Введите координаты второй точки (x2, y2): ").split())


plt.plot([x1, x2], [y1, y2], marker='o', linestyle='-', color='b')# Рисуем отрезок между точками

# Настраиваем отображение графика
plt.xlim(min(x1, x2) - 1, max(x1, x2) + 1)
plt.ylim(min(y1, y2) - 1, max(y1, y2) + 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Отрезок между двумя точками')
plt.grid(True)

# Отображаем график
plt.show()
