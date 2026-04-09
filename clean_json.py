"""
clean_json.py
=============
Fase 1: Descarga y limpieza del dataset de Khan Academy desde HuggingFace.
Produce un archivo train_clean.json listo para ser indexado.
"""

import re
import json
from datasets import load_dataset
from tqdm import tqdm

# ── Configuración ─────────────────────────────────────────────────────────────
MAX_REGISTROS   = 300   # Subconjunto para desarrollo (ajusta según necesites)
MIN_PALABRAS    = 50    # Registros con menos palabras se descartan
OUTPUT_FILE     = "train_clean.json"
# ──────────────────────────────────────────────────────────────────────────────


def limpiar_transcripcion(texto: str) -> str:
    """
    Limpia el texto de una transcripción:
    - Elimina etiquetas de subtítulos [música], [aplausos], etc.
    - Elimina marcadores de tiempo (00:00:01,000 --> 00:00:04,000)
    - Elimina líneas de índice de subtítulo (solo números)
    - Normaliza espacios en blanco y saltos de línea
    """
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


def contar_palabras(texto: str) -> int:
    """Cuenta el número de palabras en un texto."""
    return len(texto.split())


def procesar_registro(registro: dict) -> dict | None:
    """
    Procesa un registro del dataset:
    - Extrae solo los campos relevantes (title, content, url)
    - Limpia la transcripción
    - Devuelve None si el registro no es apto
    """
    contenido_raw = registro.get("content", "") or ""
    titulo        = registro.get("title", "Sin título") or "Sin título"
    url           = registro.get("url", "") or ""
    idioma        = registro.get("language", "") or ""

    # Filtrar por idioma (opcional: quedarse solo con inglés o español)
    # if idioma not in ("en", "es"):
    #     return None

    # Limpiar la transcripción
    contenido_limpio = limpiar_transcripcion(contenido_raw)

    # Descartar registros demasiado cortos
    if contar_palabras(contenido_limpio) < MIN_PALABRAS:
        return None

    return {
        "title":    titulo.strip(),
        "content":  contenido_limpio,
        "url":      url.strip(),
        "language": idioma,
    }


def dividir_en_chunks(texto: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Divide un texto largo en fragmentos (chunks) con solapamiento.
    - chunk_size: número de palabras por fragmento
    - overlap:    palabras compartidas entre fragmentos consecutivos
    """
    palabras = texto.split()
    chunks   = []
    inicio   = 0

    while inicio < len(palabras):
        fin = inicio + chunk_size
        fragmento = " ".join(palabras[inicio:fin])
        chunks.append(fragmento)
        inicio += chunk_size - overlap  # avanzar con solapamiento

    return chunks


def main():
    print("=" * 60)
    print("  Fase 1 — Limpieza del dataset Khan Academy")
    print("=" * 60)

    # ── 1. Carga del dataset desde HuggingFace ────────────────────────────────
    print(f"\n📥 Descargando dataset (primeros {MAX_REGISTROS} registros)...")
    dataset = load_dataset(
        "iblai/ibl-khanacademy-transcripts",
        split=f"train[:{MAX_REGISTROS}]",
        trust_remote_code=True,
    )
    print(f"   Registros descargados: {len(dataset)}")

    # ── 2. Limpieza y filtrado ────────────────────────────────────────────────
    print("\n🧹 Limpiando y filtrando registros...")
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
    print("\n🔍 Eliminando duplicados...")
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
    print("\n✅ Limpieza completada:")
    print(f"   Registros originales : {len(dataset)}")
    print(f"   Descartados          : {descartados}")
    print(f"   Duplicados eliminados: {len(registros_limpios) - len(sin_duplicados)}")
    print(f"   Registros finales    : {len(sin_duplicados)}")
    print(f"   Archivo guardado     : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
