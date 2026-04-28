"""CLI prediction helper for local model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.predictor import predict_from_answers


def main() -> int:
    parser = argparse.ArgumentParser(description="Predicción local con modelo Random Forest.")
    parser.add_argument(
        "--input-json",
        type=str,
        default="",
        help="Ruta a archivo JSON con respuestas {feature: valor}.",
    )
    args = parser.parse_args()

    if not args.input_json:
        print(
            "Proporciona --input-json con respuestas. "
            "Ejemplo: python predict.py --input-json artifacts/sample_answers.json"
        )
        return 1

    path = Path(args.input_json)
    if not path.exists():
        print(f"No existe el archivo de entrada: {path}")
        return 1

    answers = json.loads(path.read_text(encoding="utf-8"))
    result = predict_from_answers(answers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
