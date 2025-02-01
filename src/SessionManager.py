import os
import pickle
import hashlib
from functools import wraps
from pathlib import Path


class Session:
    PATH = Path("cache")
    SESSIONS_FILE = PATH / "sessions.txt"

    def __init__(self, new=True, name=None):
        self.PATH.mkdir(parents=True, exist_ok=True)
        self.SESSIONS_FILE.touch(exist_ok=True)  # Создаём sessions.txt, если его нет

        with open(self.SESSIONS_FILE, "r") as f:
            sessions = f.read().strip().split("\n")
        sessions = [s for s in sessions if s]  # Убираем пустые строки

        if new:
            if name is None:
                self.id = 0
                while str(self.id) in sessions:
                    self.id += 1
                self.id = str(self.id)
            else:
                self.id = str(name)
        else:
            if name is None:
                if sessions:
                    self.id = sessions[-1]  # Берём последнюю существующую сессию
                else:
                    raise ValueError("No existing sessions found.")
            elif isinstance(name, (str, int)):
                name = str(name)
                if name not in sessions:
                    raise ValueError(f"Session '{name}' not found.")
                self.id = name
            else:
                raise ValueError("Session name must be a string or an integer.")

        if new and str(self.id) not in sessions:
            with open(self.SESSIONS_FILE, "a") as f:
                print(self.id, file=f)

    def clear(self):
        session_dir = self.PATH / self.id
        if session_dir.exists():
            for file in session_dir.iterdir():
                file.unlink()
            session_dir.rmdir()

        # Удаляем сессию из sessions.txt
        with open(self.SESSIONS_FILE, "r") as f:
            sessions = f.read().strip().split("\n")

        sessions = [s for s in sessions if s and s != self.id]  # Фильтруем
        with open(self.SESSIONS_FILE, "w") as f:
            f.write("\n".join(sessions) + "\n" if sessions else "")  # Оставляем файл пустым, если нет сессий

        print(f"Session {self.id} cleared and removed from sessions.txt.")

    def __str__(self):
        return f"Session({self.id})"

    def __repr__(self):
        return f"Session({self.id})"


class DiskCache:
    _instances = {}

    def __new__(cls, cache_dir="cache", session_id="default"):
        if (cache_dir, session_id) not in cls._instances:
            instance = super().__new__(cls)
            instance.cache_dir = Path(cache_dir) / str(session_id)  # Отдельная папка для сессии
            instance.cache_file = instance.cache_dir / "session.pkl"
            instance.session_id = session_id
            instance.cache = instance._load_cache()
            cls._instances[(cache_dir, session_id)] = instance
        return cls._instances[(cache_dir, session_id)]

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # Создаем папку, если её нет
        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache, f)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        self._save_cache()

    def clear(self):
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.cache = {}


def disk_memoize(cache_dir="cache", session_id=None):
    if session_id is None:
        raise ValueError("SessionId must be specified")
    cache = DiskCache(cache_dir, session_id)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = hashlib.sha256(pickle.dumps((args, kwargs))).hexdigest()
            if (result := cache.get(key)) is not None:
                return result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        wrapper.clear_cache = cache.clear
        return wrapper

    return decorator