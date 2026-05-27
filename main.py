import numpy as np

class GamePayoffMatrix:
    def __init__(self):
        self.matrix = [
            [7, 19, 1, 19, 8],
            [7, 18, 5, 2, 6],
            [15, 3, 16, 19, 4],
            [5, 12, 19, 14, 18]
        ]
        self.rows = 4
        self.cols = 5

    def get_matrix(self):
        return self.matrix

    def get_dimensions(self):
        return self.rows, self.cols


class GameSolver:
    def __init__(self, game):
        self.game = game
        self.m = game.rows
        self.n = game.cols
        self.matrix = game.matrix

    def _print_table(self, table, basis, Cb, title, m, n, problem_type):
        rows, cols = table.shape
        print("\n" + "="*90)
        print(title)
        print("="*90)

        # Шапка
        if problem_type == 'A':
            print("|  C  |", end="")
            for _ in range(m): print("  1  |", end="")
            for _ in range(n): print("  0  |", end="")
            print("  0  |")
            print("| базис |", end="")
            for j in range(m): print(f" u{j+1} |", end="")
            for i in range(n): print(f" s{i+1} |", end="")
            print("  b  |")
        else:
            print("|  C  |", end="")
            for _ in range(m): print(" -1  |", end="")
            for _ in range(n): print("  0  |", end="")
            print("  0  |")
            print("| базис |", end="")
            for j in range(m): print(f" t{j+1} |", end="")
            for i in range(n): print(f" r{i+1} |", end="")
            print("  b  |")
        print("-" * (8 + 6 * (m + n + 2)))

        # Строки таблицы
        for i in range(rows):
            if i < rows-1:
                print(f"| {basis[i]:>4} |", end="")
            else:
                print("|  Z/W  |", end="")
            for j in range(cols):
                val = table[i][j]
                if j == cols-1:
                    print(f" {val:8.4f} |", end="")
                else:
                    print(f" {val:8.4f} ", end="")
            print()
        print("="*90)

    def _build_table(self, problem_type):
        m, n = self.m, self.n
        if problem_type == 'A':
            # Прямая задача (минимизация)
            table = np.zeros((n + 1, m + n + 1))
            for i in range(n):
                for j in range(m):
                    table[i][j] = -self.matrix[j][i]
                table[i][m + n] = -1
                table[i][m + i] = 1
            for j in range(m):
                table[n][j] = 1
        else:  # 'B'
            # Двойственная задача (максимизация)
            table = np.zeros((m + 1, n + m + 1))
            for i in range(m):
                for j in range(n):
                    table[i][j] = self.matrix[i][j]
                table[i][n + i] = 1
                table[i][-1] = 1
            for j in range(n):
                table[m][j] = -1
        return table

    def _simplex(self, table, problem_type, max_iter=100):
        if problem_type == 'A':
            m, n = self.m, self.n
            basis = [f"s{i+1}" for i in range(n)]
            Cb = [0] * n
            rows, cols = table.shape
        else:
            m, n = self.n, self.m   # здесь m - число переменных (t), n - число ограничений (r)
            basis = [f"r{i+1}" for i in range(self.m)]
            Cb = [0] * self.m
            rows, cols = table.shape

        iteration = 0
        while True:
            # Фаза 1 – устранение отрицательных свободных членов
            if problem_type == 'A':
                neg_rows = [i for i in range(self.n) if table[i][-1] < -1e-8]
            else:
                neg_rows = [i for i in range(self.m) if table[i][-1] < -1e-8]

            if neg_rows:
                pivot_row = min(neg_rows, key=lambda i: table[i][-1])
                pivot_col = -1
                for j in range(cols - 1):
                    if table[pivot_row][j] < -1e-8:
                        pivot_col = j
                        break
                if pivot_col == -1:
                    break
                print(f"\n--- Итерация {iteration+1} (Фаза 1) ---")
                print(f"Отрицательный b в строке {pivot_row} = {table[pivot_row][-1]:.4f}")
                print(f"Ведущий столбец: {pivot_col}, элемент = {table[pivot_row][pivot_col]:.4f}")
            else:
                # Фаза 2 – оптимизация
                # Вычисляем дельты
                deltas = []
                if problem_type == 'A':
                    lim = self.n
                    for j in range(cols - 1):
                        s = sum(Cb[i] * table[i][j] for i in range(lim))
                        Cj = 1 if j < self.m else 0
                        deltas.append(s - Cj)
                    pos = [j for j, d in enumerate(deltas) if d > 1e-8]
                    if not pos:
                        break
                    pivot_col = max(pos, key=lambda j: deltas[j])
                    # Выбор ведущей строки
                    min_ratio = float('inf')
                    pivot_row = -1
                    for i in range(lim):
                        if table[i][pivot_col] > 1e-8:
                            ratio = table[i][-1] / table[i][pivot_col]
                            if ratio < min_ratio:
                                min_ratio = ratio
                                pivot_row = i
                    if pivot_row == -1:
                        break
                    print(f"\n--- Итерация {iteration+1} (Фаза 2) ---")
                    print(f"Дельты: {[f'{d:.4f}' for d in deltas]}")
                    print(f"Ведущий столбец: {pivot_col}, дельта = {deltas[pivot_col]:.4f}")
                    print(f"Ведущая строка: {pivot_row}, отношение = {min_ratio:.4f}")
                else:  # problem_type == 'B'
                    lim = self.m
                    for j in range(cols - 1):
                        s = sum(Cb[i] * table[i][j] for i in range(lim))
                        Cj = -1 if j < self.n else 0
                        deltas.append(s - Cj)
                    pos = [j for j, d in enumerate(deltas) if d > 1e-8]
                    if not pos:
                        break
                    pivot_col = max(pos, key=lambda j: deltas[j])
                    min_ratio = float('inf')
                    pivot_row = -1
                    for i in range(lim):
                        if table[i][pivot_col] > 1e-8:
                            ratio = table[i][-1] / table[i][pivot_col]
                            if ratio < min_ratio:
                                min_ratio = ratio
                                pivot_row = i
                    if pivot_row == -1:
                        break
                    print(f"\n--- Итерация {iteration+1} (Фаза 2) ---")
                    print(f"Дельты: {[f'{d:.4f}' for d in deltas]}")
                    print(f"Ведущий столбец: {pivot_col}, дельта = {deltas[pivot_col]:.4f}")
                    print(f"Ведущая строка: {pivot_row}, отношение = {min_ratio:.4f}")

            # Выполняем преобразование
            table, basis, Cb = self._pivot_operation(
                table, pivot_row, pivot_col, basis, Cb, rows, problem_type
            )
            iteration += 1
            # Печатаем таблицу после итерации
            if problem_type == 'A':
                self._print_table(table, basis, Cb, f"ТАБЛИЦА ПОСЛЕ ИТЕРАЦИИ {iteration}", self.m, self.n, 'A')
            else:
                self._print_table(table, basis, Cb, f"ТАБЛИЦА ПОСЛЕ ИТЕРАЦИИ {iteration}", self.n, self.m, 'B')
            if iteration >= max_iter:
                break

        return table, basis, Cb

    def _pivot_operation(self, table, pivot_row, pivot_col, basis, Cb, rows, problem_type):
        elem = table[pivot_row][pivot_col]
        table[pivot_row] = table[pivot_row] / elem

        if problem_type == 'A':
            m, n = self.m, self.n
            if pivot_col < m:
                basis[pivot_row] = f"u{pivot_col+1}"
                Cb[pivot_row] = 1
            else:
                basis[pivot_row] = f"s{pivot_col - m + 1}"
                Cb[pivot_row] = 0
        else:  # 'B'
            n_vars = self.n   # число переменных t
            if pivot_col < n_vars:
                basis[pivot_row] = f"t{pivot_col+1}"
                Cb[pivot_row] = -1
            else:
                basis[pivot_row] = f"r{pivot_col - n_vars + 1}"
                Cb[pivot_row] = 0

        for i in range(rows):
            if i != pivot_row:
                factor = table[i][pivot_col]
                if abs(factor) > 1e-10:
                    table[i] = table[i] - factor * table[pivot_row]
        return table, basis, Cb

    def _extract_solution(self, table, basis, problem_type):
        if problem_type == 'A':
            m = self.m
            sol = np.zeros(m)
            for j in range(m):
                name = f"u{j+1}"
                if name in basis:
                    idx = basis.index(name)
                    sol[j] = table[idx][-1]
            W = table[-1][-1]
            return sol, W
        else:
            n = self.n
            sol = np.zeros(n)
            for i in range(n):
                name = f"t{i+1}"
                if name in basis:
                    idx = basis.index(name)
                    sol[i] = table[idx][-1]
            T = -table[-1][-1]   # значение целевой функции
            return sol, T

    def _solve(self, problem_type):
        print("\n" + "="*90)
        if problem_type == 'A':
            print("РЕШЕНИЕ ДЛЯ ИГРОКА A (МИНИМИЗАЦИЯ)")
        else:
            print("РЕШЕНИЕ ДЛЯ ИГРОКА B (МАКСИМИЗАЦИЯ)")
        print("="*90)

        table = self._build_table(problem_type)
        if problem_type == 'A':
            self._print_table(table, [f"s{i+1}" for i in range(self.n)], [0]*self.n,
                              "НАЧАЛЬНАЯ СИМПЛЕКС-ТАБЛИЦА", self.m, self.n, 'A')
        else:
            self._print_table(table, [f"r{i+1}" for i in range(self.m)], [0]*self.m,
                              "НАЧАЛЬНАЯ СИМПЛЕКС-ТАБЛИЦА", self.n, self.m, 'B')

        table, basis, Cb = self._simplex(table, problem_type)
        sol, val = self._extract_solution(table, basis, problem_type)

        print("\n" + "="*90)
        if problem_type == 'A':
            print("РЕЗУЛЬТАТ ДЛЯ ИГРОКА A")
            print(f"Минимальное значение W = {val:.6f}")
            v = 1.0 / val
            print(f"Цена игры v = {v:.6f}")
            print("Оптимальная стратегия A:")
            for i in range(self.m):
                print(f"  x{i+1} = {sol[i]:.6f}")
            return sol, val
        else:
            print("РЕЗУЛЬТАТ ДЛЯ ИГРОКА B")
            print(f"Максимальное значение T = {val:.6f}")
            v = 1.0 / val
            print(f"Цена игры v = {v:.6f}")
            print("Оптимальная стратегия B:")
            for i in range(self.n):
                print(f"  y{i+1} = {sol[i]:.6f}")
            return sol, val

    def get_strategy_for_minimizer(self): # A
        u, W = self._solve('A')
        return u, W

    def get_strategy_for_maximizer(self):
        t, T = self._solve('B')
        v = 1.0 / T                # цена игры
        y = t * v                  # оптимальная стратегия B
        return y, v


def main():
    game = GamePayoffMatrix()
    solver = GameSolver(game)

    u, W = solver.get_strategy_for_minimizer()
    v_A = 1.0 / W
    x = u * v_A

    y, v_B = solver.get_strategy_for_maximizer()

    print("=" * 50)
    print("ВАРИАНТ 23")
    print("=" * 50)
    print(f"\nЦена игры (из A): v = {v_A:.6f}")
    print(f"Цена игры (из B): v = {v_B:.6f}\n")
    print("Стратегия A (игрок, минимизирующий):")
    for i in range(game.rows):
        print(f"  x{i+1} = {x[i]:.6f}")
    print("\nСтратегия B (игрок, максимизирующий):")
    for i in range(game.cols):
        print(f"  y{i+1} = {y[i]:.6f}")


if __name__ == "__main__":
    main()