import io
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sympy as sp
from flask import Flask, render_template, request
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

app = Flask(__name__)

PREDEFINED_MODELS: Dict[str, str] = {
    "Recta (1 variable)": "1, x",
    "Polinomio cuadrático (1 variable)": "1, x, x^2",
    "Polinomio cúbico (1 variable)": "1, x, x^2, x^3",
    "Plano (2 variables)": "1, x1, x2",
    "Cuadrático sin interacción (2 variables)": "1, x1, x2, x1^2, x2^2",
    "Cuadrático con interacción (2 variables)": "1, x1, x2, x1^2, x2^2, x1*x2",
}

ALLOWED_FUNCTIONS = {
    "sin": sp.sin,
    "sen": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "cot": sp.cot,
    "sec": sp.sec,
    "csc": sp.csc,
    "asin": sp.asin,
    "arcsin": sp.asin,
    "asen": sp.asin,
    "arcsen": sp.asin,
    "acos": sp.acos,
    "arccos": sp.acos,
    "atan": sp.atan,
    "arctan": sp.atan,
    "acot": sp.acot,
    "arccot": sp.acot,
    "asec": sp.asec,
    "arcsec": sp.asec,
    "acsc": sp.acsc,
    "arccsc": sp.acsc,
    "sinh": sp.sinh,
    "senh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "coth": sp.coth,
    "sech": sp.sech,
    "csch": sp.csch,
    "asinh": sp.asinh,
    "arcsinh": sp.asinh,
    "asenh": sp.asinh,
    "acosh": sp.acosh,
    "arccosh": sp.acosh,
    "atanh": sp.atanh,
    "arctanh": sp.atanh,
    "acoth": sp.acoth,
    "arccoth": sp.acoth,
    "asech": sp.asech,
    "arcsech": sp.asech,
    "acsch": sp.acsch,
    "arccsch": sp.acsch,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
}

ALLOWED_CONSTANTS = {
    "e": sp.E,
    "pi": sp.pi,
}

TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)
SAFE_GLOBAL_DICT = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
}


def split_basis_functions(raw_text: str) -> List[str]:
    tokens = [t.strip() for t in re.split(r"[,;\n]+", raw_text) if t.strip()]
    if not tokens:
        raise ValueError("Debes capturar al menos una funcion base.")
    return tokens


def infer_single_variable_name(model_text: str) -> str:
    has_x1 = re.search(r"(?<![A-Za-z0-9_])x1(?![A-Za-z0-9_])", model_text) is not None
    has_x = re.search(r"(?<![A-Za-z0-9_])x(?![A-Za-z0-9_])", model_text) is not None
    if has_x and not has_x1:
        return "x"
    return "x1"


def parse_basis_expressions(
    model_text: str, n_features: int
) -> Tuple[List[sp.Expr], Tuple[sp.Symbol, ...], List[str]]:
    if n_features == 1:
        single_symbol = sp.Symbol(infer_single_variable_name(model_text))
        x_symbols: Tuple[sp.Symbol, ...] = (single_symbol,)
        # Con una sola variable, x y x1 son alias del mismo dato.
        local_dict = {"x": single_symbol, "x1": single_symbol}
    else:
        x_symbols = sp.symbols(f"x1:{n_features + 1}")
        local_dict = {f"x{i + 1}": x_symbols[i] for i in range(n_features)}
    local_dict.update(ALLOWED_FUNCTIONS)
    local_dict.update(ALLOWED_CONSTANTS)

    basis_strings = split_basis_functions(model_text)
    basis_exprs: List[sp.Expr] = []
    valid_symbols = set(x_symbols)

    for basis in basis_strings:
        normalized_basis = normalize_basis_text(basis)
        try:
            expr = parse_expr(
                normalized_basis,
                global_dict=SAFE_GLOBAL_DICT,
                local_dict=local_dict,
                transformations=TRANSFORMATIONS,
                evaluate=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"No se pudo interpretar '{basis}': {exc}") from exc

        unknown = [sym for sym in expr.free_symbols if sym not in valid_symbols]
        if unknown:
            names = ", ".join(str(sym) for sym in unknown)
            raise ValueError(f"La función '{basis}' usa variables no válidas: {names}")

        basis_exprs.append(expr)

    return basis_exprs, x_symbols, basis_strings


def normalize_basis_text(basis: str) -> str:
    """Normaliza atajos como e^(...) para mapearlos a exp(...)."""
    chars: List[str] = []
    i = 0
    text = basis

    while i < len(text):
        current = text[i]

        is_valid_left_boundary = i == 0 or not (
            text[i - 1].isalnum() or text[i - 1] == "_"
        )
        if current == "e" and is_valid_left_boundary:
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1

            if j < len(text) and text[j] == "^":
                j += 1
                while j < len(text) and text[j].isspace():
                    j += 1

                if j < len(text) and text[j] == "(":
                    depth = 0
                    k = j
                    while k < len(text):
                        if text[k] == "(":
                            depth += 1
                        elif text[k] == ")":
                            depth -= 1
                            if depth == 0:
                                inner = text[j + 1 : k]
                                chars.append(f"exp({inner})")
                                i = k + 1
                                break
                        k += 1
                    else:
                        raise ValueError(
                            f"La función '{basis}' tiene paréntesis desbalanceados."
                        )
                    continue

        chars.append(current)
        i += 1

    return "".join(chars)


def build_design_matrix(
    basis_exprs: Sequence[sp.Expr],
    x_symbols: Sequence[sp.Symbol],
    x_data: np.ndarray,
) -> np.ndarray:
    args = [x_data[:, i] for i in range(x_data.shape[1])]
    columns: List[np.ndarray] = []

    for expr in basis_exprs:
        func = sp.lambdify(x_symbols, expr, modules=["numpy"])
        try:
            values = np.asarray(func(*args), dtype=float)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"No se pudo evaluar la función '{expr}' con los datos: {exc}"
            ) from exc

        if values.ndim == 0:
            values = np.full(x_data.shape[0], float(values))
        values = values.reshape(-1)
        if values.shape[0] != x_data.shape[0]:
            raise ValueError(
                f"La función '{expr}' no produce una columna válida para la matriz."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"La función '{expr}' generó valores no finitos (inf o NaN)."
            )

        columns.append(values)

    return np.column_stack(columns)


def build_latex_equation(coefficients: np.ndarray, basis_exprs: Sequence[sp.Expr]) -> str:
    terms: List[str] = []

    for coef, expr in zip(coefficients, basis_exprs):
        coef = float(coef)
        if abs(coef) < 1e-12:
            continue

        sign = "-" if coef < 0 else "+"
        magnitude = abs(coef)
        coef_text = f"{magnitude:.6g}"
        expr_text = sp.latex(expr)

        if expr == 1:
            term = coef_text
        elif np.isclose(magnitude, 1.0):
            term = expr_text
        else:
            term = f"{coef_text}\\,{expr_text}"

        if not terms:
            terms.append(term if sign == "+" else f"-{term}")
        else:
            terms.append(f" {sign} {term}")

    rhs = "".join(terms) if terms else "0"
    return f"\\hat{{y}} = {rhs}"


def read_tabular_file(raw_bytes: bytes, source_name: Optional[str] = None) -> pd.DataFrame:
    source_name = (source_name or "").lower()
    try_excel_first = source_name.endswith((".xlsx", ".xls", ".xlsm", ".xlsb", ".ods"))

    if try_excel_first:
        try:
            return pd.read_excel(io.BytesIO(raw_bytes), header=None)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"No se pudo leer el archivo Excel: {exc}") from exc

    try:
        return pd.read_csv(io.BytesIO(raw_bytes), header=None)
    except Exception as csv_exc:  # noqa: BLE001
        try:
            return pd.read_excel(io.BytesIO(raw_bytes), header=None)
        except Exception as excel_exc:  # noqa: BLE001
            raise ValueError(
                f"No se pudo leer el archivo como CSV ni Excel: {csv_exc}; {excel_exc}"
            ) from excel_exc


def parse_numeric_table(raw_bytes: bytes, source_name: Optional[str] = None) -> pd.DataFrame:
    raw_df = read_tabular_file(raw_bytes, source_name=source_name)

    if raw_df.shape[0] < 1 or raw_df.shape[1] < 2:
        raise ValueError(
            "El archivo debe contener al menos 2 columnas: variables x y la columna y."
        )

    first_cell = raw_df.iat[0, 0]
    if isinstance(first_cell, str):
        first_cell = first_cell.strip()

    first_cell_numeric = pd.notna(pd.to_numeric(first_cell, errors="coerce"))

    if first_cell_numeric:
        df = raw_df.copy()
        df.columns = [f"col{i + 1}" for i in range(df.shape[1])]
    else:
        headers = [str(value).strip() for value in raw_df.iloc[0].tolist()]
        df = raw_df.iloc[1:].reset_index(drop=True)
        df.columns = headers

    if df.shape[0] < 2:
        raise ValueError("El archivo debe contener al menos 2 filas de datos.")

    try:
        numeric_df = df.apply(pd.to_numeric, errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "Todas las columnas del archivo deben ser numéricas para el ajuste."
        ) from exc

    return numeric_df


@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None
    result = None

    selected_model = next(iter(PREDEFINED_MODELS))
    model_text = PREDEFINED_MODELS[selected_model]
    csv_text = ""
    csv_cache = ""
    csv_filename_cache = ""
    csv_source = ""
    use_csv_text_active = False

    if request.method == "POST":
        selected_model = request.form.get("predefined_model", selected_model)
        model_text = request.form.get("model_text", "").strip()
        csv_text = request.form.get("csv_text", "")
        csv_cache = request.form.get("csv_cache", "")
        csv_filename_cache = request.form.get("csv_filename_cache", "")
        use_csv_text_active = request.form.get("use_csv_text") == "on"

        if not model_text:
            model_text = PREDEFINED_MODELS.get(selected_model, "")

        upload = request.files.get("csv_file")
        raw_table_bytes = None
        uploaded_bytes = None
        uploaded_filename = ""
        selected_source = ""

        if upload is not None and upload.filename:
            uploaded_bytes = upload.read()
            if uploaded_bytes and uploaded_bytes.strip():
                uploaded_filename = upload.filename

        if use_csv_text_active and csv_text.strip():
            raw_table_bytes = csv_text.encode("utf-8")
            selected_source = "texto"

        if raw_table_bytes is None and uploaded_bytes is not None:
            raw_table_bytes = uploaded_bytes
            selected_source = "archivo"

        if raw_table_bytes is None and csv_cache.strip():
            raw_table_bytes = csv_cache.encode("utf-8")
            selected_source = "memoria"

        if raw_table_bytes is None:
            error_message = (
                "Debes seleccionar un archivo CSV/Excel o pegar el contenido CSV en texto."
            )
        else:
            try:
                parse_source_name = uploaded_filename if selected_source == "archivo" else None
                df = parse_numeric_table(raw_table_bytes, source_name=parse_source_name)

                # Cache normalizado como texto CSV para recálculos sin re-subir archivo.
                csv_cache = df.to_csv(index=False)
                if selected_source == "archivo":
                    csv_filename_cache = uploaded_filename
                    csv_source = f"archivo ({uploaded_filename})"
                elif selected_source == "texto":
                    csv_filename_cache = ""
                    csv_source = "texto capturado"
                else:
                    if csv_filename_cache:
                        csv_source = f"CSV en memoria ({csv_filename_cache})"
                    else:
                        csv_source = "CSV en memoria"

                x_df = df.iloc[:, :-1]
                y_series = df.iloc[:, -1]
                x_data = x_df.to_numpy(dtype=float)
                y_data = y_series.to_numpy(dtype=float)

                basis_exprs, x_symbols, basis_strings = parse_basis_expressions(
                    model_text=model_text,
                    n_features=x_data.shape[1],
                )

                design_matrix = build_design_matrix(basis_exprs, x_symbols, x_data)

                coefficients, residuals, rank, singular_values = np.linalg.lstsq(
                    design_matrix, y_data, rcond=None
                )

                y_hat = design_matrix @ coefficients
                rmse = float(np.sqrt(np.mean((y_data - y_hat) ** 2)))
                plot_data = None

                if x_data.shape[1] == 1:
                    x_values = x_data[:, 0].astype(float)
                    y_values = y_data.astype(float)
                    x_min = float(np.min(x_values))
                    x_max = float(np.max(x_values))

                    if np.isclose(x_min, x_max):
                        x_plot = np.linspace(x_min - 1.0, x_max + 1.0, 200)
                    else:
                        x_plot = np.linspace(x_min, x_max, 200)

                    x_plot_matrix = x_plot.reshape(-1, 1)
                    design_plot = build_design_matrix(
                        basis_exprs=basis_exprs,
                        x_symbols=x_symbols,
                        x_data=x_plot_matrix,
                    )
                    y_plot = design_plot @ coefficients

                    plot_data = {
                        "x_label": str(x_df.columns[0]),
                        "y_label": str(y_series.name),
                        "observed": [
                            {"x": float(x), "y": float(y)}
                            for x, y in zip(x_values.tolist(), y_values.tolist())
                        ],
                        "fitted": [
                            {"x": float(x), "y": float(y)}
                            for x, y in zip(x_plot.tolist(), y_plot.tolist())
                        ],
                    }

                coef_rows = [
                    {
                        "coef": f"c{i + 1}",
                        "valor": f"{coefficients[i]:.6g}",
                        "funcion": basis_strings[i],
                    }
                    for i in range(len(coefficients))
                ]

                result = {
                    "table_html": df.to_html(
                        classes="data-table", index=False, border=0, float_format="%.6g"
                    ),
                    "equation_latex": build_latex_equation(coefficients, basis_exprs),
                    "coef_rows": coef_rows,
                    "feature_map": [
                        {"symbol_latex": sp.latex(sym), "column": col}
                        for sym, col in zip(x_symbols, x_df.columns.tolist())
                    ],
                    "target_column": y_series.name,
                    "csv_source": csv_source,
                    "plot_data": plot_data,
                    "rank": int(rank),
                    "rmse": f"{rmse:.6g}",
                    "residual_sum": (
                        f"{float(residuals[0]):.6g}" if residuals.size else "0"
                    ),
                    "singular_values": ", ".join(f"{sv:.6g}" for sv in singular_values),
                }
            except ValueError as exc:
                error_message = str(exc)
            except Exception as exc:  # noqa: BLE001
                error_message = f"Ocurrió un error inesperado: {exc}"

    return render_template(
        "index.html",
        predefined_models=PREDEFINED_MODELS,
        selected_model=selected_model,
        model_text=model_text,
        csv_text=csv_text,
        csv_cache=csv_cache,
        csv_filename_cache=csv_filename_cache,
        use_csv_text_active=use_csv_text_active,
        error_message=error_message,
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
