"""
Сервер проекта: вся нейросеть живёт здесь, на Python.

Браузер ничего не считает — он только рисует то, что пришло отсюда, и
отправляет обратно действия пользователя. Заодно этот же сервер раздаёт
сами страницы, чтобы запуск был одной командой:

    python3 python/server.py

Потом открыть http://localhost:8000

Из зависимостей только numpy. Ни Flask, ни чего-то ещё ставить не нужно —
HTTP-сервер берётся из стандартной библиотеки.
"""

import json
import base64
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np

КОРЕНЬ = Path(__file__).resolve().parent          # html лежат рядом
ПОРТ = 8000
rng = np.random.default_rng()

# ============================================================================
# ОБЩАЯ МАТЕМАТИКА
# ============================================================================

def релу(z):     return np.maximum(0.0, z)
def д_релу(z):   return (z > 0).astype(z.dtype)
def тангенс(z):  return np.tanh(z)
def д_тангенс(z):return 1.0 - np.tanh(z) ** 2
def сигмоида(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
def д_сигмоида(z):
    s = сигмоида(z)
    return s * (1.0 - s)

АКТИВАЦИИ = {
    "tanh":    (тангенс, д_тангенс),
    "relu":    (релу, д_релу),
    "sigmoid": (сигмоида, д_сигмоида),
    "linear":  (lambda z: z, lambda z: np.ones_like(z)),
}


class Сеть:
    """Полносвязная сеть. Скрытые слои — выбранная активация, выход — задаётся."""

    def __init__(self, размеры, активация="tanh", выход="sigmoid"):
        self.размеры = list(размеры)
        self.имя_активации = активация
        self.выход = выход
        self.W, self.b = [], []
        for вх, вых in zip(размеры, размеры[1:]):
            масштаб = np.sqrt(2.0 / вх) if активация == "relu" else np.sqrt(1.0 / вх) * 1.3
            self.W.append(rng.normal(0, масштаб, (вх, вых)))
            self.b.append(np.zeros(вых))

    def вперёд(self, X, всё=False):
        """Прямой проход. всё=True — вернуть активации каждого слоя."""
        f, _ = АКТИВАЦИИ[self.имя_активации]
        a = X
        слои, суммы = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            последний = i == len(self.W) - 1
            if последний:
                a = сигмоида(z) if self.выход == "sigmoid" else softmax(z)
            else:
                a = f(z)
            суммы.append(z)
            слои.append(a)
        return (слои, суммы) if всё else a

    def шаг(self, X, Y, скорость, батч, рег="none", коэф=0.0):
        """Одна эпоха: перемешать, пройти мини-батчами, поправить веса."""
        _, дf = АКТИВАЦИИ[self.имя_активации]
        порядок = rng.permutation(len(X))
        for н in range(0, len(порядок), батч):
            idx = порядок[н:н + батч]
            xb, yb = X[idx], Y[idx]
            слои, суммы = self.вперёд(xb, всё=True)

            # --- ОБРАТНОЕ РАСПРОСТРАНЕНИЕ ---
            # Выход + его функция потерь подобраны так, что производная
            # схлопывается до простой разности (предсказание - правильный ответ).
            дельта = (слои[-1] - yb) / len(idx)

            for сл in range(len(self.W) - 1, -1, -1):
                gW = слои[сл].T @ дельта
                gb = дельта.sum(axis=0)
                if рег == "L2":
                    gW += коэф * self.W[сл]
                elif рег == "L1":
                    gW += коэф * np.sign(self.W[сл])
                if сл > 0:
                    дельта = (дельта @ self.W[сл].T) * дf(суммы[сл - 1])
                self.W[сл] -= скорость * gW
                self.b[сл] -= скорость * gb


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ============================================================================
# ЧАСТЬ 1 — ИГРОВАЯ СЕТЬ (точки на плоскости)
# ============================================================================

ПРИЗНАКИ = [
    ("X₁",     lambda x, y: x),
    ("X₂",     lambda x, y: y),
    ("X₁²",    lambda x, y: x * x),
    ("X₂²",    lambda x, y: y * y),
    ("X₁X₂",   lambda x, y: x * y),
    ("sinπX₁", lambda x, y: np.sin(np.pi * x)),
    ("sinπX₂", lambda x, y: np.sin(np.pi * y)),
]
СЕТКА = 72          # разрешение карты решения
МИНИ = 24           # разрешение миниатюры нейрона
ОБЗОР = 1.1         # видимая область [-ОБЗОР, ОБЗОР]


def сделать_точки(вид, шум, доля, n=300):
    ш = шум / 40 * 0.55
    дрожь = lambda k: rng.normal(0, ш * 0.5, k)
    if вид == "gauss":
        c = np.arange(n) % 2
        cx = np.where(c == 1, 0.45, -0.45)
        cy = np.where(c == 1, 0.40, -0.40)
        x = cx + rng.normal(0, 0.2, n) + дрожь(n)
        y = cy + rng.normal(0, 0.2, n) + дрожь(n)
    elif вид == "circle":
        c = np.arange(n) % 2
        r = np.where(c == 1, np.sqrt(rng.random(n)) * 0.4, 0.6 + rng.random(n) * 0.35)
        t = rng.random(n) * 2 * np.pi
        x = r * np.cos(t) + дрожь(n)
        y = r * np.sin(t) + дрожь(n)
    elif вид == "xor":
        x = rng.random(n) * 2 - 1
        y = rng.random(n) * 2 - 1
        x += np.where(x > 0, 0.06, -0.06)
        y += np.where(y > 0, 0.06, -0.06)
        c = (x * y >= 0).astype(int)
        x = x * 0.92 + дрожь(n)
        y = y * 0.92 + дрожь(n)
    else:  # spiral
        половина = n // 2
        i = np.arange(половина) / половина
        r = 0.05 + i * 0.92
        t = i * 2.3 * np.pi
        x = np.concatenate([r * np.sin(t), r * np.sin(t + np.pi)]) + дрожь(половина * 2)
        y = np.concatenate([r * np.cos(t), r * np.cos(t + np.pi)]) + дрожь(половина * 2)
        c = np.concatenate([np.zeros(половина, int), np.ones(половина, int)])

    x = np.clip(x, -1, 1)
    y = np.clip(y, -1, 1)
    порядок = rng.permutation(len(x))
    x, y, c = x[порядок], y[порядок], c[порядок]
    тест = np.zeros(len(x), bool)
    тест[int(len(x) * доля / 100):] = True
    return x, y, c, тест


def закодировать(x, y, маска):
    столбцы = [ф(x, y) for (_, ф), вкл in zip(ПРИЗНАКИ, маска) if вкл]
    return np.stack(столбцы, axis=-1)


class ИграСостояние:
    def __init__(self):
        self.сброс({})

    def сброс(self, п, новые=True):
        self.вид     = п.get("dataset", "circle")
        self.маска   = п.get("features", [True, True, False, False, False, False, False])
        self.скрытые = п.get("hidden", [4, 2])
        self.акт     = п.get("act", "tanh")
        self.шум     = п.get("noise", 0)
        self.доля    = п.get("ratio", 50)

        if новые or not hasattr(self, "x"):
            self.x, self.y, self.c, self.тест = сделать_точки(self.вид, self.шум, self.доля)

        n_вх = sum(self.маска)
        self.сеть = Сеть([n_вх] + list(self.скрытые) + [1], self.акт, "sigmoid")
        self.эпоха = 0

        # сетка для карт активаций — считается один раз
        ось = np.linspace(-ОБЗОР, ОБЗОР, СЕТКА)
        gx, gy = np.meshgrid(ось, -ось)
        self.сетка_xy = (gx.ravel(), gy.ravel())

    @property
    def X(self):
        return закодировать(self.x, self.y, self.маска)

    def обучить(self, эпох, п):
        учеб = ~self.тест
        X, Y = self.X[учеб], self.c[учеб].reshape(-1, 1).astype(float)
        for _ in range(эпох):
            self.сеть.шаг(X, Y, п.get("lr", 0.03), п.get("batch", 10),
                          п.get("reg", "none"), п.get("regRate", 0.0))
            self.эпоха += 1

    def метрики(self):
        p = np.clip(self.сеть.вперёд(self.X).ravel(), 1e-9, 1 - 1e-9)
        ц = self.c
        потери = -(ц * np.log(p) + (1 - ц) * np.log(1 - p))
        учеб = ~self.тест
        return {
            "epoch": self.эпоха,
            "lossTrain": float(потери[учеб].mean()) if учеб.any() else None,
            "lossTest": float(потери[self.тест].mean()) if self.тест.any() else None,
            "acc": float((((p > 0.5).astype(int)) == ц).mean()),
        }

    def карты(self):
        """Карта выхода в полном разрешении + миниатюры скрытых нейронов."""
        gx, gy = self.сетка_xy
        X = закодировать(gx, gy, self.маска)
        слои, _ = self.сеть.вперёд(X, всё=True)

        выход = (слои[-1].ravel() * 255).astype(np.uint8)

        мини = []
        шаг = СЕТКА // МИНИ
        for сл in range(1, len(слои) - 1):
            a = слои[сл]
            for j in range(a.shape[1]):
                кадр = a[:, j].reshape(СЕТКА, СЕТКА)[::шаг, ::шаг]
                пик = max(abs(кадр).max(), 1e-6)
                мини.append(((кадр / пик + 1) * 127.5).astype(np.uint8))

        # карты самих признаков (для входной колонки)
        призн = []
        for (_, ф), вкл in zip(ПРИЗНАКИ, [True] * 7):
            кадр = ф(gx, gy).reshape(СЕТКА, СЕТКА)[::шаг, ::шаг]
            пик = max(abs(кадр).max(), 1e-6)
            призн.append(((кадр / пик + 1) * 127.5).astype(np.uint8))

        return {
            "out": b64(выход),
            "hidden": b64(np.concatenate([м.ravel() for м in мини])) if мини else "",
            "feats": b64(np.concatenate([п.ravel() for п in призн])),
            "res": СЕТКА, "mini": МИНИ,
        }

    def карта_узла(self, слой, номер):
        """Карта одного нейрона в полном разрешении — для режима наведения."""
        gx, gy = self.сетка_xy
        if слой == 0:
            _, ф = ПРИЗНАКИ[номер]
            кадр = ф(gx, gy)
        else:
            слои, _ = self.сеть.вперёд(закодировать(gx, gy, self.маска), всё=True)
            кадр = слои[слой][:, номер]
        пик = max(abs(кадр).max(), 1e-6)
        return b64(((кадр / пик + 1) * 127.5).astype(np.uint8))

    def полное(self):
        д = self.метрики()
        д.update(self.карты())
        д["weights"] = [W.tolist() for W in self.сеть.W]
        д["arch"] = self.сеть.размеры
        return д

    def точки(self):
        return {
            "x": [round(float(v), 4) for v in self.x],
            "y": [round(float(v), 4) for v in self.y],
            "c": [int(v) for v in self.c],
            "test": [bool(v) for v in self.тест],
        }


# ============================================================================
# ЧАСТЬ 2 — РАСПОЗНАВАНИЕ ЦИФР
# ============================================================================

class ЦифрыСостояние:
    def __init__(self):
        д = np.load(Path(__file__).with_name("mnist14.npz"))
        # В файле пиксели лежат как uint8 0..255. Их обязательно надо привести
        # к 0..1: с входами в сотни ReLU разносит суммы, softmax насыщается,
        # и сеть намертво застревает на 10% — то есть на чистом угадывании.
        self.Xtr = д["Xtr"].astype(np.float64) / 255.0
        self.Xte = д["Xte"].astype(np.float64) / 255.0
        self.ytr = д["ytr"].astype(int)
        self.yte = д["yte"].astype(int)
        self.сторона = int(np.sqrt(self.Xtr.shape[1]))
        self.сброс()

    def сброс(self):
        n_вх = self.Xtr.shape[1]
        self.сеть = Сеть([n_вх, 64, 10], "relu", "softmax")
        self.эпоха = 0
        self.потеря = None
        self.Ytr = np.eye(10)[self.ytr]

    def обучить(self, эпох, скорость=0.15, батч=64):
        for _ in range(эпох):
            self.сеть.шаг(self.Xtr, self.Ytr, скорость, батч)
            self.эпоха += 1
        p = self.сеть.вперёд(self.Xtr)
        self.потеря = float(-np.log(np.clip(p[np.arange(len(self.ytr)), self.ytr], 1e-9, 1)).mean())

    def точность(self):
        pred = self.сеть.вперёд(self.Xte).argmax(axis=1)
        return float((pred == self.yte).mean())

    def узнать(self, пиксели):
        x = np.array(пиксели, dtype=float).reshape(1, -1)
        return self.сеть.вперёд(x).ravel().tolist()

    def веса_картинками(self):
        """Веса первого слоя, разложенные обратно в картинки."""
        W = self.сеть.W[0]                       # (196, 64)
        кадры = []
        for j in range(W.shape[1]):
            к = W[:, j]
            пик = max(abs(к).max(), 1e-6)
            кадры.append(((к / пик + 1) * 127.5).astype(np.uint8))
        return b64(np.concatenate(кадры)), W.shape[1]


def b64(массив):
    return base64.b64encode(np.asarray(массив, dtype=np.uint8).tobytes()).decode()


# ============================================================================
# HTTP
# ============================================================================

игра = ИграСостояние()
цифры = ЦифрыСостояние()


class Обработчик(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(КОРЕНЬ), **kw)

    def log_message(self, *a):
        pass                                       # не засорять консоль

    def _ответ(self, данные):
        тело = json.dumps(данные).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(тело)))
        self.end_headers()
        self.wfile.write(тело)

    def do_POST(self):
        длина = int(self.headers.get("Content-Length", 0))
        п = json.loads(self.rfile.read(длина) or "{}")
        путь = self.path

        if путь == "/api/pg/reset":
            игра.сброс(п, новые=п.get("newData", True))
            о = игра.полное()
            о["points"] = игра.точки()
            return self._ответ(о)

        if путь == "/api/pg/step":
            игра.обучить(int(п.get("epochs", 1)), п)
            return self._ответ(игра.полное())

        if путь == "/api/pg/previews":
            о = {}
            for вид in ("circle", "xor", "gauss", "spiral"):
                x, y, c, _ = сделать_точки(вид, 0, 100, n=160)
                о[вид] = {"x": [round(float(v), 3) for v in x],
                          "y": [round(float(v), 3) for v in y],
                          "c": [int(v) for v in c]}
            return self._ответ(о)

        if путь == "/api/pg/node":
            return self._ответ({"grid": игра.карта_узла(int(п.get("layer", 0)),
                                                        int(п.get("index", 0)))})

        if путь == "/api/dg/reset":
            цифры.сброс()
            w, n = цифры.веса_картинками()
            return self._ответ({"epoch": 0, "acc": цифры.точность(),
                                "loss": None, "w1": w, "hid": n,
                                "side": цифры.сторона})

        if путь == "/api/dg/step":
            цифры.обучить(int(п.get("epochs", 1)))
            w, n = цифры.веса_картинками()
            return self._ответ({"epoch": цифры.эпоха, "acc": цифры.точность(),
                                "loss": цифры.потеря, "w1": w, "hid": n})

        if путь == "/api/dg/predict":
            return self._ответ({"probs": цифры.узнать(п.get("pixels", []))})

        self.send_error(404)


if __name__ == "__main__":
    print(f"Нейросеть считается здесь, на Python.")
    print(f"Открой http://localhost:{ПОРТ}\n")
    print(f"Обучающих цифр: {len(цифры.ytr)}, отложенных: {len(цифры.yte)}")
    print(f"Ctrl+C — остановить\n")
    ThreadingHTTPServer(("127.0.0.1", ПОРТ), Обработчик).serve_forever()
