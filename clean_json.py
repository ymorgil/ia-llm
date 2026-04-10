# Fase 1: Script de limpieza (clean_json.py): limpia el JSON original y produce train_clean.json

import re                           # para buscar y eliminar patrones de texto (regex)
import json                         # para leer y escribir archivos JSON
from datasets import load_dataset   # para descargar el dataset de HuggingFace
from tqdm import tqdm               # para mostrar una barra de progreso

MAX_REGISTROS   = 300   # Subconjunto para desarrollo (ajusta según necesites)
MIN_PALABRAS    = 50    # Registros con menos palabras se descartan
OUTPUT_FILE     = "train_clean.json"

# Cuenta el número de palabras en un texto.
def contar_palabras(texto: str) -> int:
    return len(texto.split())

def limpiar_transcripcion(texto: str) -> str:
    if not texto:
        return ""
    # Eliminar marcadores de tiempo SRT: 00:00:01,000 --> 00:00:04,000
    texto = re.sub(r"\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}", "", texto)
    # Eliminar líneas que solo contienen un número (índice SRT)
    texto = re.sub(r"^\s*\d+\s*$", "", texto, flags=re.MULTILINE)
    # Eliminar etiquetas entre corchetes: [música], [aplausos], [MUSIC], etc.
    texto = re.sub(r"\[.*?\]", "", texto)
    # Eliminar etiquetas HTML residuales
    texto = re.sub(r"<[^>]+>", "", texto)
    # Eliminar caracteres especiales problemáticos (mantener puntuación básica)
    texto = re.sub(r"[^\w\s.,;:!?áéíóúüñÁÉÍÓÚÜÑ\-\'\"()]", " ", texto)
    # Normalizar espacios múltiples y saltos de línea
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def procesar_registro(registro: dict) -> dict | None:
    contenido_raw = registro.get("content", "") or ""
    titulo        = registro.get("title", "Sin título") or "Sin título"
    url           = registro.get("url", "") or ""
    idioma        = registro.get("language", "") or ""

    # 1. Eliminar o filtrar registros con transcripciones vacías o demasiado cortas (< 50 palabras)
    if contar_palabras(contenido_raw) < MIN_PALABRAS:
        return None

    # 2. Eliminar etiquetas de subtítulos y marcadores de tiempo
    # 3. Normalizar espacios en blanco y saltos de línea
    contenido_limpio = limpiar_transcripcion(contenido_raw)

    # 4. Seleccionar y conservar únicamente los campos relevantes para el RAG (title, content, url)
    return {
        "title":    titulo.strip(),
        "content":  contenido_limpio,
        "url":      url.strip(),
        "language": idioma,
    }


# 5. Opcionalmente: dividir las transcripciones largas en fragmentos (chunks) coherentes
def dividir_en_chunks(texto: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    palabras = texto.split()
    chunks   = []
    inicio   = 0

    while inicio < len(palabras):
        fin = inicio + chunk_size
        fragmento = " ".join(palabras[inicio:fin])
        chunks.append(fragmento)
        inicio += chunk_size - overlap  # avanzar con solapamiento

    return chunks

# Programa Principal
def main():
    print("   ")
    print("Fase 1 — Descarga y Limpieza del Dataset")
    print("=" * 60)

    # ── 1. Carga del dataset desde HuggingFace ────────────────────────────────
    print(f"\n Descargando dataset (primeros {MAX_REGISTROS} registros)...")
    dataset = load_dataset(
        "iblai/ibl-khanacademy-transcripts",
        split=f"train[:{MAX_REGISTROS}]",
        trust_remote_code=True,
    )
    print(f"   Registros descargados: {len(dataset)}")

    # ── 2. Limpieza y filtrado ────────────────────────────────────────────────
    print("\n Limpiando y filtrando registros...")
    registros_limpios = []
    descartados = 0

    for registro in tqdm(dataset, desc="Procesando"):
        resultado = procesar_registro(dict(registro))
        if resultado is None:
            descartados += 1
            continue

        # Dividir en chunks si la transcripción es muy larga (> 600 palabras)
        if contar_palabras(resultado["content"]) > 600:
            chunks = dividir_en_chunks(resultado["content"])
            for i, chunk in enumerate(chunks):
                registros_limpios.append({
                    "title":    f"{resultado['title']} (parte {i + 1})",
                    "content":  chunk,
                    "url":      resultado["url"],
                    "language": resultado["language"],
                })
        else:
            registros_limpios.append(resultado)

    # ── 3. Eliminar duplicados por contenido ──────────────────────────────────
    print("\n Eliminando duplicados...")
    vistos      = set()
    sin_duplicados = []
    for r in registros_limpios:
        clave = r["content"][:100]  # primeros 100 chars como huella
        if clave not in vistos:
            vistos.add(clave)
            sin_duplicados.append(r)

    # ── 4. Guardar resultado ──────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sin_duplicados, f, ensure_ascii=False, indent=2)

    # ── 5. Resumen ────────────────────────────────────────────────────────────
    print("\n Limpieza completada:")
    print(f"   Registros originales : {len(dataset)}")
    print(f"   Descartados          : {descartados}")
    print(f"   Duplicados eliminados: {len(registros_limpios) - len(sin_duplicados)}")
    print(f"   Registros finales    : {len(sin_duplicados)}")
    print(f"   Archivo guardado     : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
