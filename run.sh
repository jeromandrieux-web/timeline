#!/usr/bin/env bash
set -euo pipefail

# Se placer dans le dossier du script (utile si lancé depuis ailleurs)
cd "$(dirname "$0")"

echo "=== Timeline · Run ==="

# --------- Chargement des variables d’environnement ------------
if [ -f ".env" ]; then
  echo "[INFO] Chargement du fichier .env"
  # charge sans exécuter de code arbitraire
  set -a
  source .env
  set +a
else
  echo "[WARN] Aucun fichier .env trouvé, lancement avec variables système."
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# --------- Création & activation du venv -----------------------
if [ ! -d ".venv" ]; then
  echo "[INFO] Création de l'environnement virtuel .venv"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# --------- Installation des requirements (si besoin) -----------
if [ ! -f "requirements.txt" ]; then
  echo "[ERROR] requirements.txt introuvable."
  exit 1
fi

# Option simple: installer si fastapi absent (plus fiable qu'un flag)
if ! python -c "import fastapi" >/dev/null 2>&1; then
  echo "[INFO] Installation des dépendances (fastapi manquant)"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
else
  echo "[INFO] Dépendances OK."
fi

echo "[INFO] Lancement de Uvicorn sur http://${HOST}:${PORT}"

# --------- Vérifie si le port est déjà utilisé -----------------
if command -v lsof >/dev/null 2>&1; then
  if lsof -i :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    PID=$(lsof -i :"$PORT" -sTCP:LISTEN -t | head -n 1)
    echo "[ERROR] Le port ${PORT} est déjà utilisé (PID=${PID})."
    echo "        Ferme l'autre serveur ou lance: kill -9 ${PID}"
    exit 1
  fi
fi

# --------- Ouvre le navigateur --------------------------------
open_browser() {
  local url="$1"
  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  else
    echo "[INFO] Ouvre manuellement: $url"
  fi
}

# --------- Démarrage serveur -----------------------------------
# IMPORTANT: on force le uvicorn du venv
python -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload &
UVICORN_PID=$!

sleep 1
open_browser "http://${HOST}:${PORT}/"

# arrêt propre si Ctrl+C
trap 'echo; echo "[INFO] Arrêt..."; kill "$UVICORN_PID" >/dev/null 2>&1 || true' INT TERM

wait "$UVICORN_PID"
