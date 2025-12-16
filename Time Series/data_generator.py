import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv
import argparse
import os
def generate_time_series(n_points=1000, has_trend=True, has_seasonality=True, seed=None):
    """
    Генерирует временной ряд со случайными параметрами без использования
    Pandas
    Параметры:
    ----------
    n_points : int
    Количество точек данных
    has_trend : bool
    Добавлять ли тренд
    has_seasonality : bool
    Добавлять ли сезонность
    seed : int или None
    Seed для воспроизводимости
    Возвращает:
    -----------
    tuple: (dates, values) где dates - список строк дат, values - список
    значений
    """
    if seed is not None:
        np.random.seed(seed)
    # Генерируем временные метки (ежедневные данные начиная с 2020-01-01)
    start_date = datetime(2020, 1, 1)
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_points)]
    # Базовый шум (случайная компонента)
    noise = np.random.normal(0, 5, n_points)
    # Инициализируем компоненты
    trend_component = np.zeros(n_points)
    seasonal_component = np.zeros(n_points)
    # Генерируем тренд, если требуется
    if has_trend:
    # Случайно выбираем тип тренда: линейный, квадратичный или логарифмический
        trend_type = np.random.choice(['linear', 'quadratic', 'logarithmic'])
        if trend_type == 'linear':
            # Линейный тренд: y = mx + b
            slope = np.random.uniform(-0.5, 0.5) # случайный наклон
            intercept = np.random.uniform(50, 100) # случайное начальное значение
            trend_component = slope * np.arange(n_points) + intercept

        elif trend_type == 'quadratic':
            # Квадратичный тренд: y = ax^2 + bx + c
            a = np.random.uniform(-0.001, 0.001)
            b = np.random.uniform(-0.1, 0.1)
            c = np.random.uniform(50, 100)
            x = np.arange(n_points)
            trend_component = a * x**2 + b * x + c

        else: # logarithmic
            # Логарифмический тренд: y = a*log(x+1) + b
            a = np.random.uniform(10, 30)
            b = np.random.uniform(50, 100)
            trend_component = a * np.log(np.arange(n_points) + 1) + b

    # Генерируем сезонность, если требуется
    if has_seasonality:
        # Случайное количество сезонных компонент (1-3)
        n_seasons = np.random.randint(1, 4)

        for _ in range(n_seasons):
            # Случайные параметры для каждой сезонной компоненты
            amplitude = np.random.uniform(10, 30) # амплитуда
            period = np.random.uniform(30, 365) # период в днях
            phase = np.random.uniform(0, 2*np.pi) # фаза

            # Создаем сезонную компоненту (синусоида)
            seasonal_component += amplitude * np.sin(2*np.pi*np.arange(n_points)/period + phase)

    # Комбинируем все компоненты
    values = noise + trend_component + seasonal_component

    return dates, values


def save_to_csv(dates, values, filename=None):
    """
    Сохраняет временной ряд в CSV файл без использования Pandas

    Параметры:
    ----------
    dates : list
    Список дат в строковом формате
    values : list или np.array
    Список значений временного ряда
    filename : str или None
    Имя файла. Если None, генерируется автоматически
    """
    if filename is None:
        # Генерируем имя файла с текущей датой и временем
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"tsdata_{current_time}.csv"
    
    # Сохраняем в CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Записываем заголовок
        writer.writerow(['date', 'value'])
        # Записываем данные
        for date, value in zip(dates, values):
            writer.writerow([date, f"{value:.6f}"])

        print(f"Данные сохранены в файл: {filename}")
        print(f"Размер данных: {len(dates)} строк")

    return filename

def plot_time_series(dates, values, figsize=(14, 6)):
    """
    Визуализирует временной ряд без использования Pandas

    Параметры:
    ----------
    dates : list
    Список дат в строковом формате
    values : list или np.array
    Список значений временного ряда
    figsize : tuple
    Размер фигуры
    """
    # Преобразуем строки дат в объекты datetime для matplotlib
    date_objects = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in dates]

    plt.figure(figsize=figsize)

    # Строим график временного ряда
    plt.plot(date_objects, values, label='Временной ряд', color='blue',
    linewidth=1.5)
    plt.title('Сгенерированный временной ряд', fontsize=16,
    fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Автоматически форматируем ось X для дат
    plt.gcf().autofmt_xdate()

    # Добавляем информацию о данных
    values_array = np.array(values)
    stats_text = f"Всего точек: {len(values)}\nСреднее: {np.mean(values_array):.2f}\nСтандартное отклонение: {np.std(values_array):.2f}"
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    plt.tight_layout()
    plt.show()

def generate_statistics(values):
    """Генерирует статистику по данным"""
    values_array = np.array(values)

    return {
    'count': len(values_array),
    'mean': np.mean(values_array),
    'std': np.std(values_array),
    'min': np.min(values_array),
    'max': np.max(values_array),
    '25%': np.percentile(values_array, 25),
    '50%': np.percentile(values_array, 50),
    '75%': np.percentile(values_array, 75)
    }


def main():
    """Основная функция для запуска из командной строки"""

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Генерация случайного временного ряда без Pandas')
    parser.add_argument('--n_points', type=int, default=1000, help='Количество точек данных (по умолчанию: 1000)')
    parser.add_argument('--no_trend', action='store_true', help='Исключить тренд из данных')
    parser.add_argument('--no_seasonality', action='store_true', help='Исключить сезонность из данных')
    parser.add_argument('--seed', type=int, default=None, help='Seed для воспроизводимости случайной генерации')
    parser.add_argument('--plot', action='store_true', help='Показать график после генерации')
    parser.add_argument('--output', type=str, default=None, help='Имя выходного файла (по умолчанию генерируется автоматически)')
    parser.add_argument('--stats', action='store_true', help='Показать подробную статистику')

    args = parser.parse_args()

    # Генерация данных
    print("Генерация временного ряда...")
    dates, values = generate_time_series(
        n_points=args.n_points,
        has_trend=not args.no_trend,
        has_seasonality=not args.no_seasonality,
        seed=args.seed
    )

    # Сохранение в файл
    filename = save_to_csv(dates, values, args.output)

    # Вывод основной информации о данных
    print("\nИнформация о данных:")
    print(f"Количество точек: {len(values)}")
    print(f"Наличие тренда: {not args.no_trend}")
    print(f"Наличие сезонности: {not args.no_seasonality}")
    print(f"Первый день: {dates[0]}")
    print(f"Последний день: {dates[-1]}")

    # Подробная статистика, если требуется
    if args.stats:
        stats = generate_statistics(values)
        print("\nПодробная статистика:")
        print(f"Среднее значение: {stats['mean']:.4f}")
        print(f"Стандартное отклонение: {stats['std']:.4f}")
        print(f"Минимальное значение: {stats['min']:.4f}")
        print(f"25-й процентиль: {stats['25%']:.4f}")
        print(f"Медиана (50-й процентиль): {stats['50%']:.4f}")
        print(f"75-й процентиль: {stats['75%']:.4f}")
        print(f"Максимальное значение: {stats['max']:.4f}")
    else:
        # Краткая статистика
        values_array = np.array(values)
        print(f"Среднее значение: {np.mean(values_array):.4f}")
        print(f"Стандартное отклонение: {np.std(values_array):.4f}")
        print(f"Минимальное значение: {np.min(values_array):.4f}")
        print(f"Максимальное значение: {np.max(values_array):.4f}")

    # Построение графика, если требуется
    if args.plot:
       plot_time_series(dates, values)
    
    return dates, values, filename

if __name__ == "__main__":
    # Примеры использования:
    # 1. Запуск с параметрами по умолчанию
    # python script_no_pandas.py --plot --stats

    # 2. Генерация без тренда
    # python script_no_pandas.py --no_trend --n_points 500 --plot

    # 3. Генерация с конкретным seed
    # python script_no_pandas.py --seed 42 --plot --stats

    # 4. Генерация с собственным именем файла
    # python script_no_pandas.py --output my_timeseries.csv --plot

    dates, values, filename = main()


